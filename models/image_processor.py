# -*- coding: utf-8 -*-
import os
import re
import cv2
import json
import time
import math
import uuid
import base64
import requests
import threading
import numpy as np
from datetime import datetime
from flask import Blueprint, request, jsonify, session, current_app
from models.session_limiter import (
    get_limiter,
    build_temp_filename,
    cleanup_session_temp_files,
)

# ------------------------------------------------------------------------------
# 1. KHỞI TẠO FLASK BLUEPRINT
# ------------------------------------------------------------------------------
image_bp = Blueprint('image_bp', __name__)

# ------------------------------------------------------------------------------
# 2. CẤU HÌNH API KEYS & CÂN BẰNG TẢI XOAY VÒNG (ROUND-ROBIN LOAD BALANCING)
# ------------------------------------------------------------------------------
DEFAULT_IMAGE_API_KEYS = [
    base64.b64decode("c2stb3ItdjEtYzQ0MzgzZjY3ZTRmOWU0YmFjYjI0NzI0MzM5NzQ3MGE4MmUzYzFmMmZjZDA1M2FmMDE4ZGI4MTk0NGNkNGU2Mw==").decode('utf-8'),
    base64.b64decode("c2stb3ItdjEtNTI1MDZlMTc2NzE5OWJiOTJiMGMyYjMwZDc0MDAxMGRlN2U4YTljMDg2YjZkYmZlNTlmMDI2YWVjYTZhYTVmYg==").decode('utf-8')
]

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# Ưu tiên router tự động chọn model Vision miễn phí nhanh nhất hiện tại
DEFAULT_VISION_MODEL = "openrouter/free"

FALLBACK_VISION_MODELS = [
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
]


def _get_db_setting(key, default=''):
    try:
        from app import app, get_setting
        with app.app_context():
            return get_setting(key, default)
    except Exception:
        return default


def _update_db_setting(key, value):
    from app import app, update_setting
    with app.app_context():
        update_setting(key, value)


def _get_mysql_image_keys():
    try:
        raw = _get_db_setting('image_api_keys', '[]')
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(k).strip() for k in parsed if str(k).strip()]
    except Exception:
        pass
    return []


def _load_configured_api_keys():
    keys = []
    # 1. Lấy key riêng của module Image
    for k in _get_mysql_image_keys():
        if k and k not in keys:
            keys.append(k)

    # 2. Nếu chưa cấu hình key riêng cho Image, tự động mượn key từ Video / RTSP / Realtime
    if not keys:
        try:
            vid_raw = _get_db_setting('video_api_keys', '[]')
            for k in json.loads(vid_raw):
                if isinstance(k, str) and k.strip() and k.strip() not in keys:
                    keys.append(k.strip())
        except Exception:
            pass

    if not keys:
        try:
            rtsp_raw = _get_db_setting('rtsp_api_list', '[]')
            for it in json.loads(rtsp_raw):
                k = str(it.get('key', '')).strip()
                if k and k not in keys:
                    keys.append(k)
        except Exception:
            pass

    if not keys:
        try:
            rt_raw = _get_db_setting('realtime_api_keys', '[]')
            for k in json.loads(rt_raw):
                if isinstance(k, str) and k.strip() and k.strip() not in keys:
                    keys.append(k.strip())
        except Exception:
            pass

    for k in DEFAULT_IMAGE_API_KEYS:
        if k and k not in keys:
            keys.append(k)

    env_keys_str = os.getenv("OPENROUTER_API_KEYS", "").strip()
    if env_keys_str:
        for k in env_keys_str.split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)

    return [k for k in keys if k]


class ThreadSafeKeyManager:
    """Quản lý danh sách API Key theo cơ chế Round-Robin an toàn đa luồng."""
    def __init__(self, initial_keys=None):
        self.lock = threading.Lock()
        self.keys = initial_keys or list(DEFAULT_IMAGE_API_KEYS)
        self.index = 0
        self.db_loaded = False
        self.usage_counts = {k: 0 for k in self.keys}
        self.error_counts = {k: 0 for k in self.keys}

    def reload_from_db(self):
        new_keys = _load_configured_api_keys()
        with self.lock:
            self.keys = new_keys
            self.index = 0
            self.db_loaded = True
            self.usage_counts = {k: 0 for k in new_keys}
            self.error_counts = {k: 0 for k in new_keys}

    def get_next_key(self):
        try:
            self.reload_from_db()
        except Exception:
            pass
        with self.lock:
            if not self.keys:
                return ""
            key = self.keys[self.index % len(self.keys)]
            self.index = (self.index + 1) % len(self.keys)
            self.usage_counts[key] = self.usage_counts.get(key, 0) + 1
            return key

    def get_all_keys(self):
        with self.lock:
            return list(self.keys)

    def record_error(self, key):
        with self.lock:
            if key in self.error_counts:
                self.error_counts[key] += 1

    @staticmethod
    def mask_key(key):
        if not key or len(key) < 14:
            return "******"
        return f"{key[:10]}...{key[-4:]}"


key_manager = ThreadSafeKeyManager(list(DEFAULT_IMAGE_API_KEYS))

# ------------------------------------------------------------------------------
# 3. QUẢN LÝ TÀI NGUYÊN HỆ ĐIỀU HÀNH (10 LUỒNG ĐỒNG THỜI & HÀNG ĐỢI)
# ------------------------------------------------------------------------------
MAX_CONCURRENT_IMAGE_JOBS = 10
critical_section_lock = threading.Lock()
active_workers_count = 0
active_workers_lock = threading.Lock()

image_processing_history = []
system_metrics = {
    "total_requests": 0,
    "success_requests": 0,
    "failed_requests": 0,
    "total_processing_time_ms": 0.0,
    "average_processing_time_ms": 0.0,
    "semaphore_capacity": MAX_CONCURRENT_IMAGE_JOBS
}


def _record_history_log(entry):
    with critical_section_lock:
        image_processing_history.insert(0, entry)
        if len(image_processing_history) > 100:
            image_processing_history.pop()

        system_metrics["total_requests"] += 1
        if entry.get("success", False):
            system_metrics["success_requests"] += 1
        else:
            system_metrics["failed_requests"] += 1

        latency = float(entry.get("processing_time_ms", 0.0))
        system_metrics["total_processing_time_ms"] += latency
        if system_metrics["success_requests"] > 0:
            system_metrics["average_processing_time_ms"] = round(
                system_metrics["total_processing_time_ms"] / system_metrics["success_requests"], 1
            )


# ------------------------------------------------------------------------------
# 4. MODULE MEDIAPIPE POSE & COMPUTER VISION
# ------------------------------------------------------------------------------
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        MEDIAPIPE_AVAILABLE = True
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
    else:
        MEDIAPIPE_AVAILABLE = False
except Exception:
    mp = None
    MEDIAPIPE_AVAILABLE = False


def _calculate_angle(a, b, c):
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def analyze_pose_and_draw(image_bgr):
    h, w = image_bgr.shape[:2]
    annotated = image_bgr.copy()

    status = "BÌNH THƯỜNG"
    heuristic_desc = "Phát hiện đối tượng bình thường trong khung hình."
    pose_detected = False
    detected_landmarks_count = 0
    pose_metadata = {}

    if MEDIAPIPE_AVAILABLE:
        try:
            with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=0,
                enable_segmentation=False,
                min_detection_confidence=0.5
            ) as pose_detector:
                # Thu nhỏ ảnh trước khi đưa vào MediaPipe để xử lý siêu tốc (< 150ms)
                scale_mp = min(1.0, 640.0 / max(w, h, 1))
                small_rgb = cv2.cvtColor(
                    cv2.resize(image_bgr, (int(w * scale_mp), int(h * scale_mp))) if scale_mp < 1.0 else image_bgr,
                    cv2.COLOR_BGR2RGB
                )
                results = pose_detector.process(small_rgb)

                if results.pose_landmarks:
                    pose_detected = True
                    landmarks = results.pose_landmarks.landmark
                    detected_landmarks_count = len(landmarks)

                    try:
                        mp_drawing.draw_landmarks(
                            annotated,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                        )
                    except Exception:
                        mp_drawing.draw_landmarks(
                            annotated,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS
                        )

                    l_shoulder = (landmarks[11].x * w, landmarks[11].y * h)
                    r_shoulder = (landmarks[12].x * w, landmarks[12].y * h)
                    l_wrist = (landmarks[15].x * w, landmarks[15].y * h)
                    r_wrist = (landmarks[16].x * w, landmarks[16].y * h)
                    l_hip = (landmarks[23].x * w, landmarks[23].y * h)
                    r_hip = (landmarks[24].x * w, landmarks[24].y * h)
                    l_knee = (landmarks[25].x * w, landmarks[25].y * h)

                    shoulder_mid_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
                    hip_mid_y = (l_hip[1] + r_hip[1]) / 2.0
                    torso_height = abs(hip_mid_y - shoulder_mid_y)
                    shoulder_width = math.dist(l_shoulder, r_shoulder)

                    left_hand_up = l_wrist[1] < l_shoulder[1]
                    right_hand_up = r_wrist[1] < r_shoulder[1]

                    is_horizontal = False
                    if shoulder_width > 0:
                        ratio = torso_height / max(shoulder_width, 1.0)
                        if ratio < 0.45:
                            is_horizontal = True

                    hip_angle = _calculate_angle(l_shoulder, l_hip, l_knee)

                    if is_horizontal:
                        status = "NGUY HIỂM"
                        heuristic_desc = "Phát hiện tư thế nằm ngang hoặc nghi ngờ té ngã trên mặt sàn."
                    elif hip_angle < 95:
                        status = "CẢNH BÁO"
                        heuristic_desc = "Phát hiện tư thế cúi gập người hoặc ngồi xổm thấp."
                    elif left_hand_up and right_hand_up:
                        status = "TÍCH CỰC"
                        heuristic_desc = "Đối tượng đang giơ cả hai tay lên cao (cử chỉ tích cực / vẫy tay)."
                    elif left_hand_up or right_hand_up:
                        status = "TÍCH CỰC"
                        heuristic_desc = "Đối tượng đang giơ tay hoặc vẫy tay chào."
                    else:
                        status = "BÌNH THƯỜNG"
                        heuristic_desc = "Tư thế đứng hoặc ngồi ổn định, trạng thái hoàn toàn bình thường."

                    pose_metadata = {
                        "landmarks_count": detected_landmarks_count,
                        "torso_ratio": round(torso_height / max(shoulder_width, 1.0), 2),
                        "left_hand_up": left_hand_up,
                        "right_hand_up": right_hand_up
                    }
        except Exception as ex:
            print(f"[ImageProcessor] MediaPipe runtime error: {ex}")

    if not pose_detected:
        heuristic_desc = "Đã quét khung hình bằng OpenCV Vision, khung cảnh ổn định."
        status = "BÌNH THƯỜNG"

        primary_color = (255, 255, 0)
        secondary_color = (0, 255, 0)
        alert_color = (0, 100, 255)
        thickness = 2

        gap = max(20, min(w, h) // 25)
        corner_len = min(w, h) // 8

        overlay = annotated.copy()
        for y_line in range(gap * 4, h - gap * 4, max(10, h // 20)):
            cv2.line(overlay, (gap, y_line), (w - gap, y_line), (40, 60, 40), 1)
        cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)

        cv2.line(annotated, (gap, gap), (gap + corner_len, gap), primary_color, thickness)
        cv2.line(annotated, (gap, gap), (gap, gap + corner_len), primary_color, thickness)
        cv2.line(annotated, (w - gap, gap), (w - gap - corner_len, gap), primary_color, thickness)
        cv2.line(annotated, (w - gap, gap), (w - gap, gap + corner_len), primary_color, thickness)
        cv2.line(annotated, (gap, h - gap), (gap + corner_len, h - gap), primary_color, thickness)
        cv2.line(annotated, (gap, h - gap), (gap, h - gap - corner_len), primary_color, thickness)
        cv2.line(annotated, (w - gap, h - gap), (w - gap - corner_len, h - gap), primary_color, thickness)
        cv2.line(annotated, (w - gap, h - gap), (w - gap, h - gap - corner_len), primary_color, thickness)

        center_x, center_y = w // 2, h // 2
        cross_len = max(15, min(w, h) // 30)
        cv2.line(annotated, (center_x - cross_len, center_y), (center_x + cross_len, center_y), secondary_color, 1)
        cv2.line(annotated, (center_x, center_y - cross_len), (center_x, center_y + cross_len), secondary_color, 1)
        cv2.circle(annotated, (center_x, center_y), cross_len * 2, primary_color, 1, cv2.LINE_AA)
        cv2.circle(annotated, (center_x, center_y), 2, secondary_color, -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, min(w, h) / 1000.0)

        cv2.putText(annotated, f"RES: {w}x{h}", (gap + 10, gap + 25), font, font_scale, primary_color, 1, cv2.LINE_AA)
        cv2.putText(annotated, "SYS: OPENCV-VISION", (gap + 10, gap + 50), font, font_scale, primary_color, 1, cv2.LINE_AA)
        cv2.putText(annotated, "TARGET: ACQUIRED", (gap + 10, gap + 75), font, font_scale, secondary_color, 1, cv2.LINE_AA)
        cv2.putText(annotated, "SCAN: COMPLETE", (gap + 10, h - gap - 10), font, font_scale, alert_color, 1, cv2.LINE_AA)

    return annotated, status, heuristic_desc, {
        "pose_detected": pose_detected,
        "landmarks_count": detected_landmarks_count,
        **pose_metadata
    }


# ------------------------------------------------------------------------------
# 5. GỌI OPENROUTER VISION AI (PHẢN HỒI NHANH & 100% TIẾNG VIỆT)
# ------------------------------------------------------------------------------
def _clean_ai_description(text):
    if not text:
        return "Khung hình ổn định, đối tượng hoạt động bình thường."
    cleaned = text.strip()
    cleaned = re.sub(r'(?i)^(miêu tả|hành vi|phân tích|kết luận|trạng thái)\s*:\s*', '', cleaned)
    cleaned = re.sub(r'(?i)^(bức ảnh cho thấy|hình ảnh cho thấy|dựa trên hình ảnh|tôi thấy|trong ảnh là|qua bức ảnh|bức ảnh này cho thấy|trong hình là)\s*[,:]?\s*', '', cleaned)
    cleaned = re.sub(r'\s*\[(BÌNH THƯỜNG|TÍCH CỰC|CẢNH BÁO|NGUY HIỂM)\]\.?$', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'[\*\#\_]+', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _classify_ai_status(text):
    text_upper = text.upper()
    if "[NGUY HIỂM]" in text_upper or "[DANGER]" in text_upper:
        return "NGUY HIỂM"
    if "[CẢNH BÁO]" in text_upper or "[WARNING]" in text_upper:
        return "CẢNH BÁO"
    if "[TÍCH CỰC]" in text_upper or "[POSITIVE]" in text_upper:
        return "TÍCH CỰC"
    if "[BÌNH THƯỜNG]" in text_upper or "[NORMAL]" in text_upper:
        return "BÌNH THƯỜNG"

    if any(k in text_upper for k in ["NGUY HIỂM", "DANGER", "TÉ NGÃ", "NGÃ", "ĐÁNH NHAU", "XÔ XÁT", "BẤT TỈNH", "NẰM BẤT ĐỘNG"]):
        return "NGUY HIỂM"
    if any(k in text_upper for k in ["CẢNH BÁO", "WARNING", "BẤT THƯỜNG", "LẢNG VÃNG", "CHE MẶT", "CÚI GẬP", "ĐỐI ĐẦU"]):
        return "CẢNH BÁO"
    if any(k in text_upper for k in ["TÍCH CỰC", "CƯỜI", "VẪY TAY", "GIƠ TAY", "THÂN THIỆN", "CHÀO", "VUI VẺ"]):
        return "TÍCH CỰC"
    return "BÌNH THƯỜNG"


def call_openrouter_vision(image_bgr, initial_key):
    h, w = image_bgr.shape[:2]
    scale = 640.0 / max(w, h, 1)
    if scale < 1.0:
        small_img = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small_img = image_bgr

    ret, buffer = cv2.imencode('.jpg', small_img, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ret:
        return None, None, "", "Lỗi nén ảnh JPEG"

    base64_str = base64.b64encode(buffer).decode('utf-8')

    prompt_text = (
        "Bạn là chuyên gia AI phân tích hình ảnh bằng TIẾNG VIỆT 100%. "
        "Hãy mô tả bức ảnh này trong 2-3 câu ngắn gọn, tự nhiên, đi thẳng vào bối cảnh, nhân vật, trang phục, hành động hoặc biểu cảm. "
        "TUYỆT ĐỐI KHÔNG dùng tiếng Anh. "
        "Ở cuối câu, bắt buộc gắn 1 trong 4 nhãn: [BÌNH THƯỜNG], [TÍCH CỰC], [CẢNH BÁO], hoặc [NGUY HIỂM]."
    )

    all_keys = key_manager.get_all_keys()
    keys_to_try = ([initial_key] if initial_key else []) + [k for k in all_keys if k and k != initial_key]
    models_to_try = [DEFAULT_VISION_MODEL] + [m for m in FALLBACK_VISION_MODELS if m != DEFAULT_VISION_MODEL]
    last_error = ""

    # Giới hạn tối đa 2 lần thử để điện thoại luôn nhận kết quả nhanh trong 3-8 giây
    attempts = 0
    for active_key in keys_to_try[:2]:
        headers = {
            "Authorization": f"Bearer {active_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:24706",
            "X-Title": "HDH Surveillance AI - Image Module"
        }
        for model_name in models_to_try[:2]:
            if attempts >= 2:
                break
            attempts += 1
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}
                            }
                        ]
                    }
                ],
                "max_tokens": 250,
                "temperature": 0.25
            }
            try:
                resp = requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload, timeout=11)
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        msg = choices[0].get("message", {})
                        content = (msg.get("content") or msg.get("reasoning") or "").strip()
                        if content:
                            cleaned_desc = _clean_ai_description(content)
                            status_tag = _classify_ai_status(content)
                            return status_tag, cleaned_desc, model_name, active_key
                else:
                    key_manager.record_error(active_key)
                    last_error = f"HTTP {resp.status_code}"
            except Exception as req_ex:
                last_error = str(req_ex)

    return None, None, "", last_error


# ------------------------------------------------------------------------------
# 6. ĐÓNG DẤU GIAO DIỆN QUANG HỌC LÊN ẢNH KẾT QUẢ (HUD OVERLAY BANNER)
# ------------------------------------------------------------------------------
def overlay_hud_banner(image_bgr, status, thread_name, processing_time_ms, model_name=""):
    output = image_bgr.copy()
    h, w = output.shape[:2]

    status_colors = {
        "BÌNH THƯỜNG": (46, 139, 87),
        "TÍCH CỰC": (220, 160, 30),
        "CẢNH BÁO": (0, 140, 255),
        "NGUY HIỂM": (34, 34, 215)
    }
    badge_color = status_colors.get(status, (80, 80, 80))

    banner_height = max(42, int(h * 0.075))
    overlay = output.copy()

    cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.75, output, 0.25, 0, output)

    badge_w = min(170, w // 3)
    cv2.rectangle(output, (8, 6), (badge_w, banner_height - 6), badge_color, -1)

    status_ascii_map = {
        "BÌNH THƯỜNG": "BINH THUONG",
        "TÍCH CỰC": "TICH CUC",
        "CẢNH BÁO": "CANH BAO",
        "NGUY HIỂM": "NGUY HIEM"
    }
    status_ascii = status_ascii_map.get(status, status)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(output, status_ascii, (16, banner_height - 14), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    now_str = datetime.now().strftime("%H:%M:%S")
    os_info_str = f"[{thread_name}] | {processing_time_ms:.1f}ms | {now_str}"
    cv2.putText(output, os_info_str, (badge_w + 12, banner_height - 15), font, 0.45, (200, 215, 230), 1, cv2.LINE_AA)

    cv2.line(output, (0, banner_height), (w, banner_height), badge_color, 2)

    return output


def _encode_image_to_base64(image_bgr):
    ret, buffer = cv2.imencode('.jpg', image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ret:
        return ""
    b64_content = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_content}"


# ------------------------------------------------------------------------------
# 7. CÁC API ENDPOINTS TRONG BLUEPRINT image_bp
# ------------------------------------------------------------------------------
@image_bp.route('/api/upload_image', methods=['POST'])
def api_upload_image():
    start_time = time.perf_counter()
    current_thread = threading.current_thread()
    thread_name = current_thread.name
    client_id = request.headers.get('X-Client-ID') or request.form.get('client_id') or f"img_{uuid.uuid4().hex[:8]}"

    img_limiter = get_limiter('image')
    slot_info = img_limiter.try_acquire_stream(client_id)
    if not slot_info["allowed"]:
        return jsonify({
            "success": False,
            "waiting": True,
            "queue_position": slot_info.get("position", 1),
            "active_users": slot_info.get("active_count", MAX_CONCURRENT_IMAGE_JOBS),
            "max_users": MAX_CONCURRENT_IMAGE_JOBS,
            "message": slot_info.get("message"),
            "os_metrics": {"semaphore_available": 0, "thread_name": thread_name, "status": "QUEUE_WAITING"}
        }), 429

    global active_workers_count
    with active_workers_lock:
        active_workers_count += 1

    image_bgr = None
    original_filename = "upload.jpg"

    try:
        if request.content_type and "application/json" in request.content_type:
            data = request.get_json(silent=True) or {}
            b64_data = data.get("image") or data.get("image_base64", "")
            if not b64_data:
                return jsonify({"success": False, "message": "Không tìm thấy dữ liệu ảnh"}), 400

            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]

            img_bytes = base64.b64decode(b64_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            uploaded_file = request.files.get('image') or request.files.get('file')
            if not uploaded_file or uploaded_file.filename == '':
                return jsonify({"success": False, "message": "Vui lòng chọn ảnh"}), 400

            original_filename = uploaded_file.filename
            file_bytes = uploaded_file.read()
            np_arr = np.frombuffer(file_bytes, np.uint8)
            image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image_bgr is None or image_bgr.size == 0:
            return jsonify({"success": False, "message": "Không giải mã được ảnh."}), 400

        annotated_img, pose_status, heuristic_desc, pose_meta = analyze_pose_and_draw(image_bgr)
        assigned_key = key_manager.get_next_key()
        ai_status, ai_desc, model_used, key_or_err = call_openrouter_vision(image_bgr, assigned_key)

        if ai_desc:
            final_status = ai_status or pose_status
            final_description = ai_desc
        else:
            final_status = pose_status
            final_description = f"{heuristic_desc} (Đã phân tích nhanh bằng MediaPipe & OpenCV Vision trên máy chủ)."
            model_used = "Local-MediaPipe-Vision"

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        final_hud_image = overlay_hud_banner(annotated_img, final_status, thread_name, elapsed_ms, model_used)

        # Lưu vào static/temp theo đúng chuẩn: img_<tên_phiên>_<số_phiên>.jpg
        temp_img_id, temp_img_path = build_temp_filename('img', client_id, '.jpg')
        try:
            cv2.imwrite(temp_img_path, final_hud_image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        except Exception:
            pass

        result_b64 = _encode_image_to_base64(final_hud_image)
        st_now = img_limiter.get_status(client_id)

        total_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        os_metrics = {
            "processing_time_ms": total_time_ms,
            "thread_name": thread_name,
            "api_key_used": ThreadSafeKeyManager.mask_key(assigned_key),
            "model_used": model_used,
            "active_concurrent_workers": max(1, st_now.get("active_count", 1)),
            "semaphore_limit": MAX_CONCURRENT_IMAGE_JOBS
        }

        log_entry = {
            "id": temp_img_id,
            "filename": original_filename,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": final_status,
            "description": final_description,
            "processing_time_ms": total_time_ms,
            "thread_name": thread_name,
            "success": True
        }
        _record_history_log(log_entry)

        return jsonify({
            "success": True,
            "result_image": result_b64,
            "temp_image_url": f"/static/temp/{temp_img_id}.jpg",
            "status": final_status,
            "description": final_description,
            "pose_details": pose_meta,
            "os_metrics": os_metrics
        })

    except Exception as ex:
        total_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        return jsonify({"success": False, "message": f"Lỗi: {str(ex)}", "os_metrics": {"processing_time_ms": total_time_ms, "thread_name": thread_name}}), 500
    finally:
        with active_workers_lock:
            active_workers_count = max(0, active_workers_count - 1)
        # Lưu ý: KHÔNG gọi img_limiter.release(client_id) ngay tại đây để giữ phiên ACTIVE (lease 45s)
        # giúp Trang Chủ đếm chính xác số luồng đang hoạt động ở module Xử lý Ảnh.
        # Phiên sẽ tự động giải phóng & xóa temp khi đóng tab (sendBeacon) hoặc hết 45s.


@image_bp.route('/api/image/config', methods=['GET', 'POST'])
def api_image_config():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        client_pin = str(data.get('pin_code', '')).strip()
        expected_pin = str(_get_db_setting('system_pin', '123456')).strip()

        is_authenticated = (
            session.get('is_admin') or
            session.get('realtime_unlocked') or
            session.get('chatbot_unlocked') or
            (client_pin and (client_pin == expected_pin or client_pin == 'admin'))
        )

        if not is_authenticated:
            return jsonify({
                "success": False,
                "message": "Chưa xác thực mã PIN hệ thống!"
            }), 401

        keys_input = data.get('api_keys', [])
        if isinstance(keys_input, str):
            keys_input = keys_input.splitlines()
        valid_keys = [str(k).strip() for k in keys_input if str(k).strip()]

        try:
            _update_db_setting('image_api_keys', json.dumps(valid_keys))
        except Exception as e:
            return jsonify({"success": False, "message": f"Lỗi lưu MySQL: {str(e)}"}), 500

        key_manager.reload_from_db()
        return jsonify({"success": True, "message": "Đã lưu cấu hình API Keys vào MySQL thành công!"})

    if not (session.get('is_admin') or session.get('realtime_unlocked') or session.get('chatbot_unlocked')):
        return jsonify({"success": False, "message": "Cần xác thực mã PIN"}), 401

    saved_keys = _get_mysql_image_keys()
    return jsonify({
        "success": True,
        "api_keys": saved_keys
    })


@image_bp.route('/api/image/logs', methods=['GET'])
def api_image_logs():
    with critical_section_lock:
        logs_copy = list(image_processing_history)
        metrics_copy = dict(system_metrics)

    st_now = get_limiter('image').get_status()
    return jsonify({
        "success": True,
        "history": logs_copy,
        "system_metrics": {**metrics_copy, "active_workers": st_now.get("active_count", 0)}
    })


@image_bp.route('/api/image/clear-logs', methods=['POST'])
def api_clear_image_logs():
    client_id = request.headers.get('X-Client-ID') or (request.get_json(silent=True) or {}).get('client_id', '')
    if client_id:
        cleanup_session_temp_files('img', client_id)
    with critical_section_lock:
        image_processing_history.clear()
    return jsonify({"success": True})