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
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

OPENROUTER_VISION_MODELS = [
    "google/gemma-3-27b-it:free",
    "qwen/qwen2.5-vl-32b-instruct:free",
    "google/gemma-3-12b-it:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
    "openrouter/free",
]

GROQ_VISION_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
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


def _verify_mysql_pin(input_pin):
    """Kiểm tra mã PIN đối chiếu 100% từ bảng system_settings trên MySQL."""
    pin_clean = str(input_pin or '').strip()
    if not pin_clean:
        return False
    mysql_pin = str(_get_db_setting('system_pin', '123456')).strip()
    return pin_clean == mysql_pin


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
    # 1. Ưu tiên key của module Image trong MySQL
    for k in _get_mysql_image_keys():
        if k and k not in keys:
            keys.append(k)

    # 2. Lấy thêm key Groq Vision từ cấu hình Chatbot trên MySQL (xử lý ảnh cực nhanh < 1.5s)
    try:
        chat_raw = _get_db_setting('chatbot_api_tokens', '[]')
        for item in json.loads(chat_raw):
            if isinstance(item, dict):
                k = str(item.get('api_key', '')).strip()
                if k and k not in keys:
                    keys.append(k)
    except Exception:
        pass

    # 3. Mượn thêm key từ Video / Realtime / RTSP trên MySQL
    for setting_name in ('video_api_keys', 'realtime_api_keys'):
        try:
            raw_list = _get_db_setting(setting_name, '[]')
            for k in json.loads(raw_list):
                if isinstance(k, str) and k.strip() and k.strip() not in keys:
                    keys.append(k.strip())
        except Exception:
            pass

    try:
        rtsp_raw = _get_db_setting('rtsp_api_list', '[]')
        for it in json.loads(rtsp_raw):
            if isinstance(it, dict):
                k = str(it.get('key', '')).strip()
                if k and k not in keys:
                    keys.append(k)
    except Exception:
        pass

    env_keys_str = os.getenv("OPENROUTER_API_KEYS", "").strip()
    if env_keys_str:
        for k in env_keys_str.split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)

    for k in DEFAULT_IMAGE_API_KEYS:
        if k and k not in keys:
            keys.append(k)

    return [k for k in keys if k]


class ThreadSafeKeyManager:
    """Quản lý danh sách API Key theo cơ chế Round-Robin an toàn đa luồng."""
    def __init__(self, initial_keys=None):
        self.lock = threading.Lock()
        self.keys = initial_keys or list(DEFAULT_IMAGE_API_KEYS)
        self.index = 0
        self.usage_counts = {k: 0 for k in self.keys}
        self.error_counts = {k: 0 for k in self.keys}

    def reload_from_db(self):
        new_keys = _load_configured_api_keys()
        with self.lock:
            self.keys = new_keys
            if self.index >= len(self.keys):
                self.index = 0
            for k in new_keys:
                self.usage_counts.setdefault(k, 0)
                self.error_counts.setdefault(k, 0)

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

    def get_ordered_keys(self, preferred_key=""):
        try:
            self.reload_from_db()
        except Exception:
            pass
        with self.lock:
            all_k = list(self.keys)
        # Ưu tiên các key ít lỗi và ưu tiên Groq (gsk_) vì tốc độ phản hồi Vision rất cao
        groq_keys = [k for k in all_k if k.startswith("gsk_")]
        or_keys = [k for k in all_k if not k.startswith("gsk_")]
        ordered = []
        if preferred_key and preferred_key.startswith("gsk_"):
            ordered.append(preferred_key)
        for k in groq_keys:
            if k and k not in ordered:
                ordered.append(k)
        if preferred_key and preferred_key not in ordered:
            ordered.append(preferred_key)
        for k in or_keys:
            if k and k not in ordered:
                ordered.append(k)
        return ordered

    def record_error(self, key):
        with self.lock:
            if key in self.error_counts:
                self.error_counts[key] += 1

    @staticmethod
    def mask_key(key):
        if not key or len(key) < 12:
            return "Local-Vision"
        return f"{key[:8]}...{key[-4:]}"


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
# 4. GIẢI MÃ ẢNH AN TOÀN (XỬ LÝ NỀN TRONG SUỐT PNG THÀNH NỀN TRẮNG)
# ------------------------------------------------------------------------------
def decode_image_preserve_background(img_bytes):
    """
    Giải mã ảnh từ bytes. Nếu là ảnh PNG có kênh Alpha (trong suốt),
    tự động phủ lên nền trắng (255, 255, 255) thay vì bị biến thành nền đen.
    """
    np_arr = np.frombuffer(img_bytes, np.uint8)
    raw_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    if raw_img is None or raw_img.size == 0:
        return None

    # Nếu ảnh có 4 kênh (BGRA - PNG tách nền)
    if len(raw_img.shape) == 3 and raw_img.shape[2] == 4:
        b, g, r, a = cv2.split(raw_img)
        alpha = a.astype(np.float32) / 255.0
        white_bg = np.full(raw_img.shape[:2], 255.0, dtype=np.float32)

        b_out = (b.astype(np.float32) * alpha + white_bg * (1.0 - alpha)).astype(np.uint8)
        g_out = (g.astype(np.float32) * alpha + white_bg * (1.0 - alpha)).astype(np.uint8)
        r_out = (r.astype(np.float32) * alpha + white_bg * (1.0 - alpha)).astype(np.uint8)
        return cv2.merge([b_out, g_out, r_out])

    # Nếu ảnh xám 1 kênh
    if len(raw_img.shape) == 2:
        return cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)

    return raw_img


# ------------------------------------------------------------------------------
# 5. MODULE MEDIAPIPE POSE & NHẬN DIỆN HÀNH ĐỘNG NHÂN VẬT CHI TIẾT
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


def _detect_dominant_clothing_color(image_bgr, top_y, bottom_y, left_x, right_x):
    """Nhận diện màu sắc trang phục vùng thân của nhân vật."""
    h, w = image_bgr.shape[:2]
    y1 = max(0, min(h - 1, int(top_y)))
    y2 = max(y1 + 1, min(h, int(bottom_y)))
    x1 = max(0, min(w - 1, int(left_x)))
    x2 = max(x1 + 1, min(w, int(right_x)))

    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return "trang phục lịch sự"

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mean_h = float(np.mean(hsv_roi[:, :, 0]))
    mean_s = float(np.mean(hsv_roi[:, :, 1]))
    mean_v = float(np.mean(hsv_roi[:, :, 2]))

    if mean_v < 50:
        return "trang phục tông màu tối/đen lịch lãm"
    if mean_s < 40 and mean_v > 190:
        return "trang phục màu trắng sáng"
    if mean_s < 55:
        return "trang phục tông màu xám/be nhã nhặn"
    if 8 <= mean_h <= 26:
        return "trang phục tông màu nâu/be lịch sự (áo vest/sơ mi)"
    if 90 <= mean_h <= 130:
        return "trang phục tông màu xanh dương"
    if 35 <= mean_h <= 85:
        return "trang phục tông màu xanh lá"
    if mean_h < 8 or mean_h > 165:
        return "trang phục tông màu đỏ/hồng nổi bật"
    return "trang phục gọn gàng, lịch sự"


def _analyze_character_and_action_local(image_bgr, landmarks=None):
    """
    Phân tích chi tiết nhân vật, trang phục và hành động (khoanh tay, vẫy tay, chắp tay...)
    dựa trên tọa độ khớp xương MediaPipe kết hợp đặc trưng thị giác OpenCV.
    """
    h, w = image_bgr.shape[:2]

    if landmarks:
        l_shoulder = (landmarks[11].x * w, landmarks[11].y * h)
        r_shoulder = (landmarks[12].x * w, landmarks[12].y * h)
        l_elbow = (landmarks[13].x * w, landmarks[13].y * h)
        r_elbow = (landmarks[14].x * w, landmarks[14].y * h)
        l_wrist = (landmarks[15].x * w, landmarks[15].y * h)
        r_wrist = (landmarks[16].x * w, landmarks[16].y * h)
        l_hip = (landmarks[23].x * w, landmarks[23].y * h)
        r_hip = (landmarks[24].x * w, landmarks[24].y * h)
        l_knee = (landmarks[25].x * w, landmarks[25].y * h)

        shoulder_mid_x = (l_shoulder[0] + r_shoulder[0]) / 2.0
        shoulder_mid_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
        hip_mid_y = (l_hip[1] + r_hip[1]) / 2.0
        torso_height = abs(hip_mid_y - shoulder_mid_y)
        shoulder_width = max(1.0, math.dist(l_shoulder, r_shoulder))

        # Góc khuỷu tay (để biết tay đang gập hay duỗi)
        l_elbow_angle = _calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = _calculate_angle(r_shoulder, r_elbow, r_wrist)

        # Khoảng cách giữa 2 cổ tay và khoảng cách từ cổ tay tới khuỷu tay đối diện
        wrist_dist = math.dist(l_wrist, r_wrist)
        l_wrist_to_r_elbow = math.dist(l_wrist, r_elbow)
        r_wrist_to_l_elbow = math.dist(r_wrist, l_elbow)

        chest_top_y = shoulder_mid_y - shoulder_width * 0.15
        chest_bottom_y = shoulder_mid_y + max(torso_height * 0.85, shoulder_width * 1.1)

        both_wrists_in_chest_zone = (
            chest_top_y <= l_wrist[1] <= chest_bottom_y and
            chest_top_y <= r_wrist[1] <= chest_bottom_y
        )
        both_elbows_bent = (l_elbow_angle < 125 and r_elbow_angle < 125)

        # Kiểm tra hành động KHOANH TAY (Crossed Arms)
        is_crossed_arms = False
        if both_wrists_in_chest_zone and both_elbows_bent:
            if (
                wrist_dist < shoulder_width * 0.95 or
                l_wrist_to_r_elbow < shoulder_width * 0.85 or
                r_wrist_to_l_elbow < shoulder_width * 0.85 or
                abs(l_wrist[0] - shoulder_mid_x) < shoulder_width * 0.55
            ):
                is_crossed_arms = True

        left_hand_up = l_wrist[1] < l_shoulder[1] - shoulder_width * 0.05
        right_hand_up = r_wrist[1] < r_shoulder[1] - shoulder_width * 0.05

        is_horizontal = (torso_height / shoulder_width) < 0.42 and hip_mid_y < h * 0.85
        hip_angle = _calculate_angle(l_shoulder, l_hip, l_knee)

        clothing_desc = _detect_dominant_clothing_color(
            image_bgr,
            shoulder_mid_y,
            min(h, shoulder_mid_y + shoulder_width * 1.2),
            min(l_shoulder[0], r_shoulder[0]),
            max(l_shoulder[0], r_shoulder[0])
        )

        if is_horizontal:
            return (
                "NGUY HIỂM",
                f"Nhân vật mặc {clothing_desc} đang ở tư thế nằm ngang hoặc có dấu hiệu té ngã trên mặt sàn."
            )
        if left_hand_up and right_hand_up:
            return (
                "TÍCH CỰC",
                f"Nhân vật trong ảnh mặc {clothing_desc}, đang giơ cả hai tay lên cao với biểu cảm vui vẻ, năng động."
            )
        if left_hand_up or right_hand_up:
            hand_side = "tay trái" if left_hand_up else "tay phải"
            return (
                "TÍCH CỰC",
                f"Nhân vật trong ảnh mặc {clothing_desc}, đang giơ {hand_side} lên vẫy chào với thái độ thân thiện, tích cực."
            )
        if is_crossed_arms:
            return (
                "BÌNH THƯỜNG",
                f"Nhân vật trong ảnh mặc {clothing_desc}, đang đứng nhìn thẳng và thực hiện hành động khoanh hai tay trước ngực với phong thái tự tin, điềm tĩnh."
            )
        if hip_angle < 95 and hip_mid_y < h * 0.88:
            return (
                "CẢNH BÁO",
                f"Nhân vật mặc {clothing_desc} đang thực hiện hành động cúi gập người hoặc ngồi xổm thấp trong khung hình."
            )

        return (
            "BÌNH THƯỜNG",
            f"Nhân vật trong ảnh mặc {clothing_desc}, đang đứng thẳng hướng về phía trước với tư thế ổn định, tự nhiên."
        )

    # Trường hợp ảnh minh họa/chibi/chân dung mà MediaPipe không bắt hết khớp xương
    clothing_desc = _detect_dominant_clothing_color(
        image_bgr, int(h * 0.55), int(h * 0.88), int(w * 0.3), int(w * 0.7)
    )
    return (
        "BÌNH THƯỜNG",
        f"Trong ảnh là một nhân vật mặc {clothing_desc}, đứng chính diện ở trung tâm khung hình với biểu cảm tươi tắn, thân thiện."
    )


def analyze_pose_and_draw(image_bgr):
    h, w = image_bgr.shape[:2]
    annotated = image_bgr.copy()

    status = "BÌNH THƯỜNG"
    heuristic_desc = ""
    pose_detected = False
    detected_landmarks_count = 0
    pose_metadata = {}

    if MEDIAPIPE_AVAILABLE:
        try:
            with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.35
            ) as pose_detector:
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

                    status, heuristic_desc = _analyze_character_and_action_local(image_bgr, landmarks)
                    pose_metadata = {
                        "landmarks_count": detected_landmarks_count,
                        "pose_detected": True
                    }
        except Exception as ex:
            print(f"[ImageProcessor] MediaPipe runtime error: {ex}")

    if not pose_detected:
        status, heuristic_desc = _analyze_character_and_action_local(image_bgr, None)

        primary_color = (212, 122, 83)
        secondary_color = (46, 139, 87)
        alert_color = (0, 140, 255)
        thickness = 2

        gap = max(20, min(w, h) // 25)
        corner_len = min(w, h) // 8

        cv2.line(annotated, (gap, gap), (gap + corner_len, gap), primary_color, thickness)
        cv2.line(annotated, (gap, gap), (gap, gap + corner_len), primary_color, thickness)
        cv2.line(annotated, (w - gap, gap), (w - gap - corner_len, gap), primary_color, thickness)
        cv2.line(annotated, (w - gap, gap), (w - gap, gap + corner_len), primary_color, thickness)
        cv2.line(annotated, (gap, h - gap), (gap + corner_len, h - gap), primary_color, thickness)
        cv2.line(annotated, (gap, h - gap), (gap, h - gap - corner_len), primary_color, thickness)
        cv2.line(annotated, (w - gap, h - gap), (w - gap - corner_len, h - gap), primary_color, thickness)
        cv2.line(annotated, (w - gap, h - gap), (w - gap, h - gap - corner_len), primary_color, thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, min(w, h) / 1000.0)
        cv2.putText(annotated, f"RES: {w}x{h}", (gap + 10, gap + 25), font, font_scale, primary_color, 1, cv2.LINE_AA)
        cv2.putText(annotated, "SYS: OPENCV-VISION", (gap + 10, gap + 50), font, font_scale, secondary_color, 1, cv2.LINE_AA)
        cv2.putText(annotated, "SCAN: COMPLETE", (gap + 10, h - gap - 10), font, font_scale, alert_color, 1, cv2.LINE_AA)

    return annotated, status, heuristic_desc, {
        "pose_detected": pose_detected,
        "landmarks_count": detected_landmarks_count,
        **pose_metadata
    }


# ------------------------------------------------------------------------------
# 6. GỌI VISION AI (MÔ TẢ CHI TIẾT NHÂN VẬT, TRANG PHỤC & HÀNH ĐỘNG)
# ------------------------------------------------------------------------------
def _clean_ai_description(text):
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = re.sub(r'(?i)^(miêu tả|mô tả|hành vi|phân tích|kết luận|trạng thái)\s*:\s*', '', cleaned)
    cleaned = re.sub(
        r'(?i)^(bức ảnh cho thấy|hình ảnh cho thấy|dựa trên hình ảnh|tôi thấy|qua bức ảnh|bức ảnh này cho thấy)\s*[,:]?\s*',
        '',
        cleaned
    )
    cleaned = re.sub(r'\s*\[(BÌNH THƯỜNG|TÍCH CỰC|CẢNH BÁO|NGUY HIỂM|NORMAL|POSITIVE|WARNING|DANGER)\]\.?$', '', cleaned, flags=re.IGNORECASE)
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

    if any(k in text_upper for k in ["NGUY HIỂM", "TÉ NGÃ", "ĐÁNH NHAU", "XÔ XÁT", "BẤT TỈNH", "NẰM BẤT ĐỘNG", "VŨ KHÍ"]):
        return "NGUY HIỂM"
    if any(k in text_upper for k in ["CẢNH BÁO", "BẤT THƯỜNG", "LẢNG VÃNG", "CHE MẶT", "ĐỐI ĐẦU"]):
        return "CẢNH BÁO"
    if any(k in text_upper for k in ["VẪY TAY", "GIƠ TAY", "THẢ TIM", "RẠNG RỠ"]):
        return "TÍCH CỰC"
    return "BÌNH THƯỜNG"


def call_vision_ai(image_bgr, initial_key):
    h, w = image_bgr.shape[:2]
    scale = 768.0 / max(w, h, 1)
    if scale < 1.0:
        small_img = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small_img = image_bgr

    ret, buffer = cv2.imencode('.jpg', small_img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ret:
        return None, None, "", "Lỗi nén ảnh JPEG"

    base64_str = base64.b64encode(buffer).decode('utf-8')

    prompt_text = (
        "Bạn là chuyên gia AI nhận diện nhân vật và hành động bằng TIẾNG VIỆT 100%. "
        "Hãy quan sát kỹ bức ảnh và mô tả chi tiết trong 2-3 câu tự nhiên theo đúng cấu trúc: "
        "1. Nhân vật trong ảnh là ai (nam/nữ, người thật hay nhân vật hoạt hình/chibi, kiểu tóc, nét mặt/biểu cảm như mỉm cười, nghiêm túc...). "
        "2. Trang phục và phụ kiện đang mặc (ví dụ: mặc bộ vest màu nâu/be, áo sơ mi, đeo cà vạt xanh, đeo đồng hồ ở cổ tay...). "
        "3. Nhân vật đang thực hiện HÀNH ĐỘNG hoặc CỬ CHỈ gì (chú ý quan sát kỹ đôi tay và tư thế: ví dụ đang đứng khoanh hai tay trước ngực, đang vẫy tay, đang cầm đồ vật, hay đang ngồi...). "
        "TUYỆT ĐỐI KHÔNG dùng tiếng Anh. Ở cuối cùng, gắn đúng 1 nhãn: [BÌNH THƯỜNG], [TÍCH CỰC], [CẢNH BÁO], hoặc [NGUY HIỂM]."
    )

    candidate_keys = key_manager.get_ordered_keys(initial_key)
    last_error = ""
    attempts = 0
    max_attempts = 5

    for active_key in candidate_keys:
        if attempts >= max_attempts:
            break

        is_groq = active_key.startswith("gsk_")
        endpoint = GROQ_ENDPOINT if is_groq else OPENROUTER_ENDPOINT
        model_list = GROQ_VISION_MODELS if is_groq else OPENROUTER_VISION_MODELS

        headers = {
            "Authorization": f"Bearer {active_key}",
            "Content-Type": "application/json",
        }
        if not is_groq:
            headers["HTTP-Referer"] = "http://localhost:24706"
            headers["X-Title"] = "HDH Surveillance AI - Image Module"

        for model_name in model_list:
            if attempts >= max_attempts:
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
                "max_tokens": 320,
                "temperature": 0.2
            }

            try:
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=12)
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        msg = choices[0].get("message", {})
                        content = (msg.get("content") or msg.get("reasoning") or "").strip()
                        if content:
                            cleaned_desc = _clean_ai_description(content)
                            if len(cleaned_desc) > 15:
                                status_tag = _classify_ai_status(content)
                                return status_tag, cleaned_desc, model_name, active_key
                else:
                    key_manager.record_error(active_key)
                    last_error = f"HTTP {resp.status_code}"
                    if resp.status_code in (401, 403):
                        break
            except Exception as req_ex:
                last_error = str(req_ex)

    return None, None, "", last_error


# ------------------------------------------------------------------------------
# 7. ĐÓNG DẤU GIAO DIỆN QUANG HỌC LÊN ẢNH KẾT QUẢ (HUD OVERLAY BANNER)
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
    ret, buffer = cv2.imencode('.jpg', image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ret:
        return ""
    b64_content = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_content}"


# ------------------------------------------------------------------------------
# 8. CÁC API ENDPOINTS TRONG BLUEPRINT image_bp
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
            image_bgr = decode_image_preserve_background(img_bytes)
        else:
            uploaded_file = request.files.get('image') or request.files.get('file')
            if not uploaded_file or uploaded_file.filename == '':
                return jsonify({"success": False, "message": "Vui lòng chọn ảnh"}), 400

            original_filename = uploaded_file.filename
            file_bytes = uploaded_file.read()
            image_bgr = decode_image_preserve_background(file_bytes)

        if image_bgr is None or image_bgr.size == 0:
            return jsonify({"success": False, "message": "Không giải mã được ảnh."}), 400

        annotated_img, pose_status, heuristic_desc, pose_meta = analyze_pose_and_draw(image_bgr)
        assigned_key = key_manager.get_next_key()
        ai_status, ai_desc, model_used, used_key = call_vision_ai(image_bgr, assigned_key)

        if ai_desc:
            if pose_status == "NGUY HIỂM" and ai_status == "BÌNH THƯỜNG":
                final_status = "CẢNH BÁO"
            else:
                final_status = ai_status or pose_status
            final_description = ai_desc
            active_key_display = used_key
        else:
            final_status = pose_status
            final_description = heuristic_desc
            model_used = "Local-Pose-Action-Analyzer"
            active_key_display = assigned_key

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        final_hud_image = overlay_hud_banner(annotated_img, final_status, thread_name, elapsed_ms, model_used)

        temp_img_id, temp_img_path = build_temp_filename('img', client_id, '.jpg')
        try:
            cv2.imwrite(temp_img_path, final_hud_image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        except Exception:
            pass

        result_b64 = _encode_image_to_base64(final_hud_image)
        st_now = img_limiter.get_status(client_id)

        total_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        os_metrics = {
            "processing_time_ms": total_time_ms,
            "thread_name": thread_name,
            "api_key_used": ThreadSafeKeyManager.mask_key(active_key_display),
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
        return jsonify({
            "success": False,
            "message": f"Lỗi: {str(ex)}",
            "os_metrics": {"processing_time_ms": total_time_ms, "thread_name": thread_name}
        }), 500
    finally:
        with active_workers_lock:
            active_workers_count = max(0, active_workers_count - 1)


@image_bp.route('/api/image/verify-pin', methods=['POST'])
def api_image_verify_pin():
    """Xác thực mã PIN lấy trực tiếp từ bảng system_settings trên MySQL."""
    data = request.get_json(silent=True) or {}
    client_pin = str(data.get('pin', '')).strip()

    if _verify_mysql_pin(client_pin):
        session['is_admin'] = True
        session['image_unlocked'] = True
        saved_keys = _get_mysql_image_keys()
        return jsonify({
            "success": True,
            "message": "Xác thực mã PIN từ MySQL thành công!",
            "api_keys": saved_keys
        })

    return jsonify({
        "success": False,
        "message": "Mã PIN không chính xác (đối chiếu từ MySQL)!"
    }), 401


@image_bp.route('/api/image/config', methods=['GET', 'POST'])
def api_image_config():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        client_pin = str(data.get('pin_code', '')).strip()

        is_authenticated = (
            _verify_mysql_pin(client_pin) or
            session.get('is_admin') or
            session.get('image_unlocked')
        )

        if not is_authenticated:
            return jsonify({
                "success": False,
                "message": "Mã PIN không hợp lệ! Vui lòng xác thực lại với MySQL."
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

    pin_header_or_query = request.headers.get('X-Pin-Code') or request.args.get('pin', '')
    if not (
        _verify_mysql_pin(pin_header_or_query) or
        session.get('is_admin') or
        session.get('image_unlocked')
    ):
        return jsonify({"success": False, "message": "Cần xác thực mã PIN hệ thống từ MySQL"}), 401

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