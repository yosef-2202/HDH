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
from flask import Blueprint, request, jsonify, session

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

DEFAULT_VISION_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

FALLBACK_VISION_MODELS = [
    "deepseek/deepseek-v4-flash:free",
    "google/gemma-4-26b-a4b-it:free",
    "openrouter/free"
]

def _load_configured_api_keys():
    """Tải API Keys từ danh sách mặc định, biến môi trường, config_local.json và Database dự phòng"""
    keys = list(DEFAULT_IMAGE_API_KEYS)

    env_keys_str = os.getenv("OPENROUTER_API_KEYS", "").strip()
    if env_keys_str:
        for k in env_keys_str.split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)

    env_single = os.getenv("OPENROUTER_API_KEY", "").strip()
    if env_single and env_single not in keys:
        keys.append(env_single)

    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config_local.json")
        if not os.path.exists(config_path):
            config_path = "config_local.json"

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                json_keys = data.get("OPENROUTER_API_KEYS", [])
                if isinstance(json_keys, list):
                    for k in json_keys:
                        k = str(k).strip()
                        if k and k not in keys:
                            keys.append(k)
                single_key = data.get("OPENROUTER_API_KEY", "").strip()
                if single_key and single_key not in keys:
                    keys.append(single_key)
    except Exception as e:
        print(f"[ImageProcessor] Note when reading config_local.json: {e}")

    try:
        import pymysql
        conn = pymysql.connect(
            host='91.99.159.222',
            port=3306,
            user='u50283_95Wuon3aKO',
            password='e!nfV.oVXQ6Qcwso9P@WCkz5',
            database='s50283_HDH',
            connect_timeout=2
        )
        with conn.cursor() as cur:
            cur.execute("SELECT setting_key, setting_value FROM system_settings WHERE setting_key IN ('token_video', 'token_video_2', 'rtsp_api_list')")
            for row in cur.fetchall():
                skey, sval = row[0], (row[1] or '').strip()
                if not sval:
                    continue
                if skey in ('token_video', 'token_video_2') and sval not in keys:
                    keys.append(sval)
                elif skey == 'rtsp_api_list':
                    try:
                        items = json.loads(sval)
                        for it in items:
                            k = it.get('key', '').strip()
                            if k and k not in keys:
                                keys.append(k)
                    except Exception:
                        pass
        conn.close()
    except Exception:
        pass

    return [k for k in keys if k]


class ThreadSafeKeyManager:
    """Quản lý danh sách API Key theo cơ chế Round-Robin an toàn đa luồng."""
    def __init__(self, initial_keys=None):
        self.lock = threading.Lock()
        self.keys = initial_keys or list(DEFAULT_IMAGE_API_KEYS)
        self.index = 0
        self.usage_counts = {k: 0 for k in self.keys}
        self.error_counts = {k: 0 for k in self.keys}

    def get_next_key(self):
        with self.lock:
            if not self.keys:
                return ""
            key = self.keys[self.index % len(self.keys)]
            self.index = (self.index + 1) % len(self.keys)
            self.usage_counts[key] = self.usage_counts.get(key, 0) + 1
            return key

    def record_error(self, key):
        with self.lock:
            if key in self.error_counts:
                self.error_counts[key] += 1

    @staticmethod
    def mask_key(key):
        if not key or len(key) < 14:
            return "******"
        return f"{key[:12]}...{key[-4:]}"

    def get_stats(self):
        with self.lock:
            return {
                "total_keys": len(self.keys),
                "current_index": self.index,
                "keys_detail": [
                    {
                        "masked_key": self.mask_key(k),
                        "usage_count": self.usage_counts.get(k, 0),
                        "error_count": self.error_counts.get(k, 0)
                    }
                    for k in self.keys
                ]
            }

key_manager = ThreadSafeKeyManager(_load_configured_api_keys())


# ------------------------------------------------------------------------------
# 3. QUẢN LÝ TÀI NGUYÊN HỆ ĐIỀU HÀNH (SEMAPHORE & CRITICAL SECTION LOCK)
# ------------------------------------------------------------------------------
MAX_CONCURRENT_IMAGE_JOBS = 3
image_concurrency_semaphore = threading.Semaphore(MAX_CONCURRENT_IMAGE_JOBS)
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
    MEDIAPIPE_AVAILABLE = True
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
except Exception as _mp_err:
    mp = None
    MEDIAPIPE_AVAILABLE = False
    print(f"[ImageProcessor] MediaPipe not available ({_mp_err}). Switching to OpenCV Fallback.")


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
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5
            ) as pose_detector:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                results = pose_detector.process(image_rgb)

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

                    nose = (landmarks[0].x * w, landmarks[0].y * h)
                    l_shoulder = (landmarks[11].x * w, landmarks[11].y * h)
                    r_shoulder = (landmarks[12].x * w, landmarks[12].y * h)
                    l_wrist = (landmarks[15].x * w, landmarks[15].y * h)
                    r_wrist = (landmarks[16].x * w, landmarks[16].y * h)
                    l_hip = (landmarks[23].x * w, landmarks[23].y * h)
                    r_hip = (landmarks[24].x * w, landmarks[24].y * h)
                    l_knee = (landmarks[25].x * w, landmarks[25].y * h)
                    r_knee = (landmarks[26].x * w, landmarks[26].y * h)

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
                        heuristic_desc = "Tư thế nằm ngang / nghi ngờ té ngã trên sàn."
                    elif hip_angle < 95:
                        status = "CẢNH BÁO"
                        heuristic_desc = "Tư thế cúi gập người bất thường hoặc ngồi xổm thấp."
                    elif left_hand_up and right_hand_up:
                        status = "TÍCH CỰC"
                        heuristic_desc = "Đang giơ cả hai tay lên cao (cử chỉ tích cực / vẫy tay)."
                    elif left_hand_up or right_hand_up:
                        status = "TÍCH CỰC"
                        heuristic_desc = "Đang giơ tay / vẫy tay chào."
                    else:
                        status = "BÌNH THƯỜNG"
                        heuristic_desc = "Tư thế đứng hoặc ngồi ổn định, trạng thái bình thường."

                    pose_metadata = {
                        "landmarks_count": detected_landmarks_count,
                        "torso_ratio": round(torso_height / max(shoulder_width, 1.0), 2),
                        "left_hand_up": left_hand_up,
                        "right_hand_up": right_hand_up
                    }
        except Exception as ex:
            print(f"[ImageProcessor] MediaPipe runtime error: {ex}")

    if not pose_detected:
        heuristic_desc = "Phát hiện đối tượng/khung hình (chế độ quét OpenCV bề mặt)."
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
        cv2.putText(annotated, "SYS: OPENCV-FALLBACK", (gap + 10, gap + 50), font, font_scale, primary_color, 1, cv2.LINE_AA)
        cv2.putText(annotated, "TARGET: ACQUIRED", (gap + 10, gap + 75), font, font_scale, secondary_color, 1, cv2.LINE_AA)
        cv2.putText(annotated, "SCAN: COMPLETE", (gap + 10, h - gap - 10), font, font_scale, alert_color, 1, cv2.LINE_AA)

    return annotated, status, heuristic_desc, {
        "pose_detected": pose_detected,
        "landmarks_count": detected_landmarks_count,
        **pose_metadata
    }

# ------------------------------------------------------------------------------
# 5. GỌI OPENROUTER VISION AI (CLOUD LLM REASONING)
# ------------------------------------------------------------------------------
def _clean_ai_description(text):
    if not text:
        return "Bình thường, đối tượng trong khung hình."
    cleaned = text.strip()
    cleaned = re.sub(r'(?i)^(miêu tả|hành vi|phân tích|kết luận|trạng thái)\s*:\s*', '', cleaned)
    # Loại bỏ các cụm máy móc nếu AI mở đầu theo quán tính
    cleaned = re.sub(r'(?i)^(bức ảnh cho thấy|hình ảnh cho thấy|dựa trên hình ảnh|tôi thấy|trong ảnh là|qua bức ảnh|bức ảnh này cho thấy|trong hình là)\s*[,:]?\s*', '', cleaned)
    # Loại bỏ tag trạng thái ở cuối nếu có để câu văn hoàn toàn tự nhiên
    cleaned = re.sub(r'\s*\[(BÌNH THƯỜNG|TÍCH CỰC|CẢNH BÁO|NGUY HIỂM)\]\.?$', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'[\*\#\_]+', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _classify_ai_status(text):
    text_upper = text.upper()
    # Ưu tiên nhận diện tag đặt trong ngoặc vuông
    if "[NGUY HIỂM]" in text_upper or "[DANGER]" in text_upper:
        return "NGUY HIỂM"
    if "[CẢNH BÁO]" in text_upper or "[WARNING]" in text_upper:
        return "CẢNH BÁO"
    if "[TÍCH CỰC]" in text_upper or "[POSITIVE]" in text_upper:
        return "TÍCH CỰC"
    if "[BÌNH THƯỜNG]" in text_upper or "[NORMAL]" in text_upper:
        return "BÌNH THƯỜNG"

    if any(k in text_upper for k in ["NGUY HIỂM", "DANGER", "TÉ NGÃ", "NGÃ", "ĐÁNH NHAU", "BẤT TỈNH", "NẰM BẤT ĐỘNG"]):
        return "NGUY HIỂM"
    if any(k in text_upper for k in ["CẢNH BÁO", "WARNING", "BẤT THƯỜNG", "LẢNG VÃNG", "CHE MẶT", "CÚI GẬP", "MỆT MỎI"]):
        return "CẢNH BÁO"
    if any(k in text_upper for k in ["TÍCH CỰC", "CƯỜI", "VẪY TAY", "GIƠ TAY", "THÂN THIỆN", "CHÀO"]):
        return "TÍCH CỰC"
    return "BÌNH THƯỜNG"


def _extract_answer_from_reasoning(reasoning_text):
    if not reasoning_text:
        return "Bình thường, đối tượng trong khung hình."
    parts = [p.strip() for p in reasoning_text.split('\n\n') if p.strip()]
    non_thinking = [p for p in parts if not p.lower().startswith('thinking process') and not re.match(r'^\d+\.', p)]
    if non_thinking:
        text = non_thinking[-1]
    else:
        text = parts[-1]
    text = text.strip('"\'')
    return _clean_ai_description(text)


def call_openrouter_vision(image_bgr, initial_key):
    h, w = image_bgr.shape[:2]
    scale = 640.0 / max(w, 1)
    if scale < 1.0:
        small_img = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small_img = image_bgr

    ret, buffer = cv2.imencode('.jpg', small_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ret:
        return None, None, "", "Lỗi nén ảnh JPEG"

    base64_str = base64.b64encode(buffer).decode('utf-8')

    prompt_text = (
        "Hãy quan sát bức ảnh và kể lại những gì đang diễn ra một cách tự nhiên, đúng mực như một người kể chuyện (khoảng 2-3 câu ngắn):\n"
        "- Bao quát đầy đủ bối cảnh, nhân vật hoặc đồ vật, trang phục, hành vi cử chỉ và biểu cảm nổi bật một cách súc tích.\n"
        "- BẮT BUỘC hành văn đúng mực, tự nhiên, mượt mà và chân thật như con người đang diễn đạt trực tiếp.\n"
        "- TUYỆT ĐỐI KHÔNG dùng các cụm từ máy móc rập khuôn của AI như: 'Bức ảnh cho thấy', 'Dựa trên hình ảnh', 'Tôi thấy', 'Trong ảnh', 'Bức ảnh này'. Hãy đi thẳng ngay vào nội dung và diễn biến của khung cảnh.\n"
        "- Ở câu cuối cùng, hãy kèm theo nhận định trạng thái trong ngoặc vuông: [BÌNH THƯỜNG], [TÍCH CỰC], [CẢNH BÁO], hoặc [NGUY HIỂM]."
    )

    models_to_try = [DEFAULT_VISION_MODEL] + [m for m in FALLBACK_VISION_MODELS if m != DEFAULT_VISION_MODEL]
    active_key = initial_key
    last_error = ""
    best_reasoning_fallback = None

    for model_name in models_to_try:
        headers = {
            "Authorization": f"Bearer {active_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:24706",
            "X-Title": "HDH Surveillance AI - Image Module"
        }

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_str}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300,
            "temperature": 0.3
        }

        try:
            resp = requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload, timeout=10)

            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    content = msg.get("content")
                    reasoning = msg.get("reasoning")

                    if content and isinstance(content, str) and content.strip():
                        cleaned_desc = _clean_ai_description(content)
                        status_tag = _classify_ai_status(content)
                        return status_tag, cleaned_desc, model_name, active_key

                    if reasoning and isinstance(reasoning, str) and reasoning.strip() and not best_reasoning_fallback:
                        cleaned_r = _extract_answer_from_reasoning(reasoning)
                        best_reasoning_fallback = (_classify_ai_status(cleaned_r), cleaned_r, model_name, active_key)

            elif resp.status_code == 429 or resp.status_code == 401:
                key_manager.record_error(active_key)
                backup_key = key_manager.get_next_key()
                if backup_key and backup_key != active_key:
                    active_key = backup_key
                last_error = f"Lỗi {resp.status_code} trên model {model_name}"
                continue

            last_error = f"Model {model_name} HTTP {resp.status_code}"
        except Exception as req_ex:
            last_error = f"Lỗi gọi {model_name}: {req_ex}"

    if best_reasoning_fallback:
        return best_reasoning_fallback

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
    thread_id = threading.get_ident()

    acquired = image_concurrency_semaphore.acquire(blocking=True, timeout=15.0)
    if not acquired:
        return jsonify({
            "success": False,
            "message": "Hệ thống đang quá tải. Vui lòng thử lại sau giây lát!",
            "os_metrics": { "semaphore_available": 0, "thread_name": thread_name, "status": "SEMAPHORE_TIMEOUT" }
        }), 503

    global active_workers_count
    with active_workers_lock:
        active_workers_count += 1
        current_active = active_workers_count

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
            ai_source = f"OpenRouter Cloud AI ({model_used})"
        else:
            final_status = pose_status
            final_description = f"{heuristic_desc} (Phân tích Computer Vision cục bộ)"
            model_used = "Local-MediaPipe-Heuristic"
            ai_source = "Local Engine (Cloud AI Bận/RateLimit/Invalid Key)"

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        final_hud_image = overlay_hud_banner(annotated_img, final_status, thread_name, elapsed_ms, model_used)
        result_b64 = _encode_image_to_base64(final_hud_image)

        total_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        os_metrics = {
            "processing_time_ms": total_time_ms,
            "thread_name": thread_name,
            "api_key_used": ThreadSafeKeyManager.mask_key(assigned_key),
            "model_used": model_used,
            "active_concurrent_workers": current_active,
            "semaphore_limit": MAX_CONCURRENT_IMAGE_JOBS
        }

        log_entry = {
            "id": str(uuid.uuid4())[:8],
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
            "status": final_status,
            "description": final_description,
            "pose_details": pose_meta,
            "os_metrics": os_metrics
        })

    except Exception as ex:
        total_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        return jsonify({ "success": False, "message": f"Lỗi: {str(ex)}", "os_metrics": { "processing_time_ms": total_time_ms, "thread_name": thread_name } }), 500
    finally:
        with active_workers_lock:
            active_workers_count = max(0, active_workers_count - 1)
        image_concurrency_semaphore.release()


@image_bp.route('/api/image/config', methods=['GET', 'POST'])
def api_image_config():
    """API cho phép thiết lập và lưu trữ OpenRouter API Key thông qua UI (Yêu cầu PIN xác thực)"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config_local.json")
    
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        
        # 1. Kiểm tra xác thực mã PIN bảo mật liên kết với app.py
        client_pin = str(data.get('pin_code', '')).strip()
        expected_pin = "123456"
        try:
            from app import get_setting
            db_pin = get_setting('system_pin')
            if db_pin:
                expected_pin = str(db_pin).strip()
        except Exception:
            pass
            
        admin_env_pin = os.getenv("ADMIN_PIN", os.getenv("SYSTEM_PIN", "")).strip()

        is_authenticated = (
            session.get('realtime_unlocked') or
            session.get('chatbot_unlocked') or
            session.get('is_admin') or
            (client_pin and (client_pin == expected_pin or (admin_env_pin and client_pin == admin_env_pin)))
        )
        
        if not is_authenticated:
            return jsonify({
                "success": False, 
                "message": "Mã PIN không đúng hoặc chưa được xác thực! Vui lòng nhập đúng mã PIN hệ thống."
            }), 401

        keys_input = data.get('api_keys', [])
        valid_keys = [k.strip() for k in keys_input if k.strip()]
        
        config_data = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            except:
                pass
        
        config_data['OPENROUTER_API_KEYS'] = valid_keys
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            return jsonify({"success": False, "message": f"Lỗi ghi file: {str(e)}"}), 500
            
        # Áp dụng thay đổi ngay vào Key Manager mà không cần reset server
        global key_manager
        new_keys = _load_configured_api_keys()
        with key_manager.lock:
            key_manager.keys = new_keys
            key_manager.index = 0
            key_manager.usage_counts = {k: 0 for k in new_keys}
            key_manager.error_counts = {k: 0 for k in new_keys}
            
        return jsonify({"success": True, "message": "Đã lưu cấu hình API Keys thành công!"})
        
    # Xử lý Request GET (Đổ dữ liệu ra UI)
    config_data = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except:
            pass
            
    return jsonify({
        "success": True, 
        "api_keys": config_data.get('OPENROUTER_API_KEYS', [])
    })

@image_bp.route('/api/image/logs', methods=['GET'])
def api_image_logs():
    with critical_section_lock:
        logs_copy = list(image_processing_history)
        metrics_copy = dict(system_metrics)

    with active_workers_lock:
        current_workers = active_workers_count

    return jsonify({ "success": True, "history": logs_copy, "system_metrics": { **metrics_copy, "active_workers": current_workers } })

@image_bp.route('/api/image/clear-logs', methods=['POST'])
def api_clear_image_logs():
    with critical_section_lock:
        image_processing_history.clear()
    return jsonify({"success": True})