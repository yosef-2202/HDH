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
from flask import Blueprint, request, jsonify

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

# Danh sách Mô hình Thị giác (Vision Models)
DEFAULT_VISION_MODEL = "dots-studio/dots-3-note-preview:free"
FALLBACK_VISION_MODELS = [
    "deepseek/deepseek-v4-flash:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "openrouter/free"
]


def _load_configured_api_keys():
    """Tải API Keys từ danh sách mặc định, biến môi trường, config_local.json và Database dự phòng"""
    keys = list(DEFAULT_IMAGE_API_KEYS)

    # 1. Biến môi trường
    env_keys_str = os.getenv("OPENROUTER_API_KEYS", "").strip()
    if env_keys_str:
        for k in env_keys_str.split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)

    env_single = os.getenv("OPENROUTER_API_KEY", "").strip()
    if env_single and env_single not in keys:
        keys.append(env_single)

    # 2. File config_local.json (nếu có)
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

    # 3. Tự động nạp thêm các key dự phòng từ DB bằng pymysql trực tiếp (tránh circular import với app.py)
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
    """
    Quản lý danh sách API Key theo cơ chế Round-Robin an toàn đa luồng.
    Áp dụng Mutex Lock để tránh xung đột chỉ số giữa các request đồng thời.
    """
    def __init__(self, initial_keys=None):
        self.lock = threading.Lock()
        self.keys = initial_keys or list(DEFAULT_IMAGE_API_KEYS)
        self.index = 0
        self.usage_counts = {k: 0 for k in self.keys}
        self.error_counts = {k: 0 for k in self.keys}

    def get_next_key(self):
        """Lấy key tiếp theo theo thuật toán Round-Robin"""
        with self.lock:
            if not self.keys:
                return ""
            key = self.keys[self.index % len(self.keys)]
            self.index = (self.index + 1) % len(self.keys)
            self.usage_counts[key] = self.usage_counts.get(key, 0) + 1
            return key

    def record_error(self, key):
        """Ghi nhận lỗi 429 hoặc lỗi kết nối cho key cụ thể"""
        with self.lock:
            if key in self.error_counts:
                self.error_counts[key] += 1

    @staticmethod
    def mask_key(key):
        """Che bớt ký tự key để hiển thị an toàn trên Dashboard / Metrics"""
        if not key or len(key) < 14:
            return "******"
        return f"{key[:12]}...{key[-4:]}"

    def get_stats(self):
        """Thống kê tình trạng sử dụng của các Keys"""
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


# Khởi tạo KeyManager dùng chung toàn module
key_manager = ThreadSafeKeyManager(_load_configured_api_keys())


# ------------------------------------------------------------------------------
# 3. QUẢN LÝ TÀI NGUYÊN HỆ ĐIỀU HÀNH (SEMAPHORE & CRITICAL SECTION LOCK)
# ------------------------------------------------------------------------------
# Semaphore giới hạn tối đa 3 luồng xử lý ảnh đồng thời (tránh quá tải CPU/RAM)
MAX_CONCURRENT_IMAGE_JOBS = 3
image_concurrency_semaphore = threading.Semaphore(MAX_CONCURRENT_IMAGE_JOBS)

# Khóa Lock bảo vệ vùng găng khi cập nhật Lịch sử xử lý và Bộ đếm Thống kê
critical_section_lock = threading.Lock()

# Bộ đếm giám sát luồng đang hoạt động
active_workers_count = 0
active_workers_lock = threading.Lock()

# Lưu trữ lịch sử phân tích và thống kê hệ thống (In-memory buffer)
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
    """Ghi nhận log vào danh sách lịch sử với khóa bảo vệ Vùng Găng (Critical Section)"""
    with critical_section_lock:
        image_processing_history.insert(0, entry)
        if len(image_processing_history) > 100:
            image_processing_history.pop()

        # Cập nhật thống kê
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
# Kiểm tra khả năng nạp MediaPipe an toàn (graceful fallback nếu thiếu thư viện)
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
    """Tính góc giữa 3 điểm a, b, c (tính theo độ)"""
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def analyze_pose_and_draw(image_bgr):
    """
    Sử dụng MediaPipe Pose (static_image_mode=True) để nhận diện người và vẽ khung xương.
    Nếu không phát hiện hoặc không có MediaPipe, sử dụng OpenCV để vẽ HUD và khung cảnh báo.
    
    Trả về:
    - annotated_image (numpy ndarray): Ảnh đã vẽ landmarks / HUD
    - status (str): "BÌNH THƯỜNG", "TÍCH CỰC", "CẢNH BÁO", "NGUY HIỂM"
    - heuristic_desc (str): Mô tả tư thế nhận diện được
    - pose_metadata (dict): Thông số tọa độ chi tiết
    """
    h, w = image_bgr.shape[:2]
    annotated = image_bgr.copy()

    status = "BÌNH THƯỜNG"
    heuristic_desc = "Phát hiện đối tượng bình thường trong khung hình."
    pose_detected = False
    detected_landmarks_count = 0
    pose_metadata = {}

    # 1. Thử nhận diện bằng MediaPipe Pose
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

                    # Vẽ khung xương MediaPipe lên ảnh
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

                    # Lấy các điểm giải phẫu quan trọng
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

                    # Phân tích các trường hợp tư thế:
                    # A. Giơ tay / Vẫy tay (Cổ tay cao hơn vai đáng kể)
                    left_hand_up = l_wrist[1] < l_shoulder[1]
                    right_hand_up = r_wrist[1] < r_shoulder[1]

                    # B. Nằm hoặc ngã (Khoảng cách dọc thân mình quá ngắn so với bề ngang vai)
                    is_horizontal = False
                    if shoulder_width > 0:
                        ratio = torso_height / max(shoulder_width, 1.0)
                        if ratio < 0.45:
                            is_horizontal = True

                    # C. Cúi gập người (Góc vai - hông - gối gập nhỏ)
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

    # 2. Nếu MediaPipe không phát hiện hoặc không khả dụng -> Dùng OpenCV vẽ HUD cảnh báo & khung
    if not pose_detected:
        heuristic_desc = "Phát hiện đối tượng/khung hình (chưa trích xuất được khung xương người)."
        status = "BÌNH THƯỜNG"

        # Vẽ 4 góc khung ngắm quang học (Futuristic Viewfinder Brackets)
        corner_len = min(w, h) // 10
        bracket_color = (0, 220, 255)  # Màu vàng hổ phách
        thickness = 2

        # Góc trên - trái
        cv2.line(annotated, (20, 20), (20 + corner_len, 20), bracket_color, thickness)
        cv2.line(annotated, (20, 20), (20, 20 + corner_len), bracket_color, thickness)
        # Góc trên - phải
        cv2.line(annotated, (w - 20, 20), (w - 20 - corner_len, 20), bracket_color, thickness)
        cv2.line(annotated, (w - 20, 20), (w - 20, 20 + corner_len), bracket_color, thickness)
        # Góc dưới - trái
        cv2.line(annotated, (20, h - 20), (20 + corner_len, h - 20), bracket_color, thickness)
        cv2.line(annotated, (20, h - 20), (20, h - 20 - corner_len), bracket_color, thickness)
        # Góc dưới - phải
        cv2.line(annotated, (w - 20, h - 20), (w - 20 - corner_len, h - 20), bracket_color, thickness)
        cv2.line(annotated, (w - 20, h - 20), (w - 20, h - 20 - corner_len), bracket_color, thickness)

    return annotated, status, heuristic_desc, {
        "pose_detected": pose_detected,
        "landmarks_count": detected_landmarks_count,
        **pose_metadata
    }


# ------------------------------------------------------------------------------
# 5. GỌI OPENROUTER VISION AI (CLOUD LLM REASONING)
# ------------------------------------------------------------------------------
def _clean_ai_description(text):
    """Làm sạch văn bản trả lời từ AI để đạt độ tự nhiên, xúc tích"""
    if not text:
        return "Bình thường, đối tượng trong khung hình."
    cleaned = text.strip()
    # Loại bỏ tiền tố như "Miêu tả:", "Hành vi:", "Kết luận:"
    cleaned = re.sub(r'(?i)^(miêu tả|hành vi|phân tích|kết luận|trạng thái)\s*:\s*', '', cleaned)
    cleaned = re.sub(r'[\*\#\_]+', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _classify_ai_status(text):
    """Phân loại trạng thái an ninh dựa trên nội dung mô tả của AI"""
    text_upper = text.upper()
    if any(k in text_upper for k in ["NGUY HIỂM", "DANGER", "TÉ NGÃ", "NGÃ", "ĐÁNH NHAU", "BẤT TỈNH", "NẰM BẤT ĐỘNG"]):
        return "NGUY HIỂM"
    if any(k in text_upper for k in ["CẢNH BÁO", "WARNING", "BẤT THƯỜNG", "LẢNG VÃNG", "CHE MẶT", "CÚI GẬP", "MỆT MỎI"]):
        return "CẢNH BÁO"
    if any(k in text_upper for k in ["TÍCH CỰC", "CƯỜI", "VẪY TAY", "GIƠ TAY", "THÂN THIỆN", "CHÀO"]):
        return "TÍCH CỰC"
    return "BÌNH THƯỜNG"


def _extract_answer_from_reasoning(reasoning_text):
    """Trích xuất câu miêu tả từ khối reasoning nếu model dùng hết token vào suy luận"""
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
    """
    Gửi ảnh (chiều rộng tối đa 640px, nén JPEG Base64) lên OpenRouter API để phân tích.
    Tự động thử lần lượt các model thị giác và xoay vòng API Keys nếu gặp Rate Limit (429).
    """
    h, w = image_bgr.shape[:2]
    # Thu nhỏ ảnh (chiều rộng tối đa 640px)
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
        "Bạn là chuyên gia thị giác máy tính AI phân tích hành vi và biểu cảm con người qua hình ảnh. "
        "Hãy quan sát kỹ bức ảnh và mô tả chi tiết bằng tiếng Việt tự nhiên (khoảng 2-3 câu):\n"
        "- Nhận diện rõ trong ảnh có những ai/con vật/đồ vật gì, trang phục, tư thế (đứng, ngồi, nằm, giơ tay...).\n"
        "- Hành động cụ thể, cử chỉ và biểu cảm nổi bật (bình thường, vui vẻ, mỉm cười, mệt mỏi, bất thường...).\n"
        "- Bối cảnh xung quanh và nhận định tổng thể trạng thái: [BÌNH THƯỜNG], [TÍCH CỰC], [CẢNH BÁO], hoặc [NGUY HIỂM]."
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

                    # Ưu tiên lấy content văn bản trả lời trực tiếp
                    if content and isinstance(content, str) and content.strip():
                        cleaned_desc = _clean_ai_description(content)
                        status_tag = _classify_ai_status(content)
                        return status_tag, cleaned_desc, model_name, active_key

                    # Nếu content là None nhưng có reasoning
                    if reasoning and isinstance(reasoning, str) and reasoning.strip() and not best_reasoning_fallback:
                        cleaned_r = _extract_answer_from_reasoning(reasoning)
                        best_reasoning_fallback = (_classify_ai_status(cleaned_r), cleaned_r, model_name, active_key)

            elif resp.status_code == 429:
                key_manager.record_error(active_key)
                backup_key = key_manager.get_next_key()
                if backup_key and backup_key != active_key:
                    active_key = backup_key
                last_error = f"Rate Limit (429) trên model {model_name}"
                continue

            last_error = f"Model {model_name} HTTP {resp.status_code}"
        except Exception as req_ex:
            last_error = f"Lỗi gọi {model_name}: {req_ex}"

    # Nếu có kết quả trích xuất từ reasoning của một model 200 OK
    if best_reasoning_fallback:
        return best_reasoning_fallback

    return None, None, "", last_error


# ------------------------------------------------------------------------------
# 6. ĐÓNG DẤU GIAO DIỆN QUANG HỌC LÊN ẢNH KẾT QUẢ (HUD OVERLAY BANNER)
# ------------------------------------------------------------------------------
def overlay_hud_banner(image_bgr, status, thread_name, processing_time_ms, model_name=""):
    """
    Vẽ thanh thông tin trạng thái HUD chuẩn quân sự / AI Vision ở đầu ảnh.
    Sử dụng màu sắc trực quan tương ứng với mức độ an ninh:
    - BÌNH THƯỜNG: Xanh lục
    - TÍCH CỰC: Xanh lam / Cyan
    - CẢNH BÁO: Vàng / Cam
    - NGUY HIỂM: Đỏ thẫm
    """
    output = image_bgr.copy()
    h, w = output.shape[:2]

    # Bảng màu BGR
    status_colors = {
        "BÌNH THƯỜNG": (46, 139, 87),   # Xanh lá biển (SeaGreen)
        "TÍCH CỰC": (220, 160, 30),     # Xanh ngọc / Cyan
        "CẢNH BÁO": (0, 140, 255),      # Cam
        "NGUY HIỂM": (34, 34, 215)      # Đỏ
    }
    badge_color = status_colors.get(status, (80, 80, 80))

    banner_height = max(42, int(h * 0.075))
    overlay = output.copy()

    # Dải băng tối màu trong suốt ở phía trên
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.75, output, 0.25, 0, output)

    # Khối màu trạng thái
    badge_w = min(170, w // 3)
    cv2.rectangle(output, (8, 6), (badge_w, banner_height - 6), badge_color, -1)

    # Chuyển text tiếng Việt sang không dấu cho cv2.putText
    status_ascii_map = {
        "BÌNH THƯỜNG": "BINH THUONG",
        "TÍCH CỰC": "TICH CUC",
        "CẢNH BÁO": "CANH BAO",
        "NGUY HIỂM": "NGUY HIEM"
    }
    status_ascii = status_ascii_map.get(status, status)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(output, status_ascii, (16, banner_height - 14), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Hiển thị thông số Hệ điều hành: Thread, Thời gian xử lý, Model
    now_str = datetime.now().strftime("%H:%M:%S")
    os_info_str = f"[{thread_name}] | {processing_time_ms:.1f}ms | {now_str}"
    cv2.putText(output, os_info_str, (badge_w + 12, banner_height - 15), font, 0.45, (200, 215, 230), 1, cv2.LINE_AA)

    # Đường chỉ viền màu công nghệ phía dưới banner
    cv2.line(output, (0, banner_height), (w, banner_height), badge_color, 2)

    return output


def _encode_image_to_base64(image_bgr):
    """Mã hóa ảnh OpenCV sang chuỗi Data URI JPEG Base64"""
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
    """
    API tiếp nhận và phân tích ảnh tĩnh:
    - Hỗ trợ tải file: multipart/form-data với field 'image' hoặc 'file'
    - Hỗ trợ chuỗi base64: application/json với key 'image' hoặc 'image_base64'
    - Kiểm soát đa luồng: Sử dụng Semaphore(3) giới hạn tối đa 3 tiến trình đồng thời
    - Đo đạc thời gian thực thi (Latency / Processing Time ms)
    - Trả về JSON chứa kết quả ảnh base64, nhãn trạng thái và thông số Hệ điều hành
    """
    start_time = time.perf_counter()
    current_thread = threading.current_thread()
    thread_name = current_thread.name
    thread_id = threading.get_ident()

    # Bước 1: Áp dụng Semaphore để kiểm soát số luồng song song (OS Concurrency Control)
    # Thử xin quyền với thời gian chờ tối đa 15 giây nếu hệ thống đang đầy tải
    acquired = image_concurrency_semaphore.acquire(blocking=True, timeout=15.0)
    if not acquired:
        return jsonify({
            "success": False,
            "message": "Hệ thống đang quá tải (vượt quá 3 luồng xử lý ảnh đồng thời). Vui lòng thử lại sau giây lát!",
            "os_metrics": {
                "semaphore_available": 0,
                "thread_name": thread_name,
                "status": "SEMAPHORE_TIMEOUT"
            }
        }), 503

    # Tăng biến đếm worker đang hoạt động
    global active_workers_count
    with active_workers_lock:
        active_workers_count += 1
        current_active = active_workers_count

    image_bgr = None
    original_filename = "upload.jpg"

    try:
        # Bước 2: Đọc dữ liệu ảnh từ Multipart Form hoặc JSON Payload
        if request.content_type and "application/json" in request.content_type:
            data = request.get_json(silent=True) or {}
            b64_data = data.get("image") or data.get("image_base64", "")
            if not b64_data:
                return jsonify({"success": False, "message": "Không tìm thấy dữ liệu ảnh trong JSON"}), 400

            # Xử lý tiền tố data:image/...;base64,
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]

            try:
                img_bytes = base64.b64decode(b64_data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception as dec_err:
                return jsonify({"success": False, "message": f"Dữ liệu base64 không hợp lệ: {dec_err}"}), 400
        else:
            uploaded_file = request.files.get('image') or request.files.get('file')
            if not uploaded_file or uploaded_file.filename == '':
                return jsonify({"success": False, "message": "Vui lòng chọn một file ảnh để tải lên"}), 400

            original_filename = uploaded_file.filename
            file_bytes = uploaded_file.read()
            np_arr = np.frombuffer(file_bytes, np.uint8)
            image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image_bgr is None or image_bgr.size == 0:
            return jsonify({"success": False, "message": "Không thể giải mã định dạng ảnh này. Hãy thử ảnh JPG hoặc PNG!"}), 400

        # Bước 3: Phân tích khung xương MediaPipe Pose & tính toán Heuristic sơ bộ
        annotated_img, pose_status, heuristic_desc, pose_meta = analyze_pose_and_draw(image_bgr)

        # Bước 4: Lấy API Key xoay vòng (Round-Robin) và gọi OpenRouter Vision AI
        assigned_key = key_manager.get_next_key()
        ai_status, ai_desc, model_used, key_or_err = call_openrouter_vision(image_bgr, assigned_key)

        # Quyết định kết quả cuối cùng (Ưu tiên nhận định sâu từ AI, fallback về Heuristic nếu Cloud bận)
        if ai_desc:
            final_status = ai_status or pose_status
            final_description = ai_desc
            ai_source = f"OpenRouter Cloud AI ({model_used})"
        else:
            final_status = pose_status
            final_description = f"{heuristic_desc} (Phân tích Computer Vision cục bộ)"
            model_used = "Local-MediaPipe-Heuristic"
            ai_source = "Local Engine (Cloud AI Bận/RateLimit)"

        # Bước 5: Đóng dấu HUD trực quan lên ảnh kết quả
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        final_hud_image = overlay_hud_banner(
            annotated_img,
            final_status,
            thread_name,
            elapsed_ms,
            model_name=model_used
        )
        result_b64 = _encode_image_to_base64(final_hud_image)

        # Tính toán thông số Hệ điều hành (OS Metrics)
        total_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        os_metrics = {
            "processing_time_ms": total_time_ms,
            "thread_name": thread_name,
            "thread_id": thread_id,
            "api_key_used": ThreadSafeKeyManager.mask_key(assigned_key),
            "model_used": model_used,
            "ai_source": ai_source,
            "active_concurrent_workers": current_active,
            "semaphore_limit": MAX_CONCURRENT_IMAGE_JOBS
        }

        # Bước 6: Lưu log vào Vùng Găng (Critical Section)
        log_entry = {
            "id": str(uuid.uuid4())[:8],
            "filename": original_filename,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": final_status,
            "description": final_description,
            "processing_time_ms": total_time_ms,
            "thread_name": thread_name,
            "model_used": model_used,
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
        error_log = {
            "id": str(uuid.uuid4())[:8],
            "filename": original_filename,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "LỖI",
            "description": f"Ngoại lệ xử lý: {str(ex)}",
            "processing_time_ms": total_time_ms,
            "thread_name": thread_name,
            "model_used": "None",
            "success": False
        }
        _record_history_log(error_log)
        return jsonify({
            "success": False,
            "message": f"Lỗi trong quá trình xử lý ảnh: {str(ex)}",
            "os_metrics": {
                "processing_time_ms": total_time_ms,
                "thread_name": thread_name
            }
        }), 500

    finally:
        # Giải phóng Semaphore và giảm bộ đếm luồng
        with active_workers_lock:
            active_workers_count = max(0, active_workers_count - 1)
        image_concurrency_semaphore.release()


@image_bp.route('/api/image/logs', methods=['GET'])
def api_image_logs():
    """
    Trả về danh sách lịch sử các bức ảnh đã phân tích và các chỉ số thống kê hệ điều hành.
    Truy xuất được bảo vệ an toàn bằng Khóa (Lock).
    """
    with critical_section_lock:
        logs_copy = list(image_processing_history)
        metrics_copy = dict(system_metrics)

    with active_workers_lock:
        current_workers = active_workers_count

    key_stats = key_manager.get_stats()

    return jsonify({
        "success": True,
        "total_records": len(logs_copy),
        "history": logs_copy,
        "system_metrics": {
            **metrics_copy,
            "active_workers": current_workers,
            "semaphore_available_estimate": max(0, MAX_CONCURRENT_IMAGE_JOBS - current_workers)
        },
        "api_keys_status": key_stats
    })


@image_bp.route('/api/image/stats', methods=['GET'])
def api_image_stats():
    """API nhanh trả về tóm tắt trạng thái tải và thông số hệ thống của module xử lý ảnh"""
    with critical_section_lock:
        metrics_copy = dict(system_metrics)
    with active_workers_lock:
        current_workers = active_workers_count

    return jsonify({
        "success": True,
        "active_workers": current_workers,
        "semaphore_capacity": MAX_CONCURRENT_IMAGE_JOBS,
        "metrics": metrics_copy,
        "keys": key_manager.get_stats()
    })


@image_bp.route('/api/image/clear-logs', methods=['POST'])
def api_clear_image_logs():
    """Xóa lịch sử phân tích trong bộ nhớ RAM"""
    with critical_section_lock:
        image_processing_history.clear()
        system_metrics["total_requests"] = 0
        system_metrics["success_requests"] = 0
        system_metrics["failed_requests"] = 0
        system_metrics["total_processing_time_ms"] = 0.0
        system_metrics["average_processing_time_ms"] = 0.0

    return jsonify({"success": True, "message": "Đã xóa toàn bộ lịch sử phân tích ảnh"})
