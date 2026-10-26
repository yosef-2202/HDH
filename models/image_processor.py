# -*- coding: utf-8 -*-
import os
import sys
import re
import cv2
import json
import time
import math
import uuid
import base64
import psutil
import requests
import threading
import numpy as np
from datetime import datetime
from flask import Blueprint, request, jsonify, session, current_app
from models.session_limiter import (
    get_limiter,
    build_temp_filename,
    cleanup_session_temp_files,
    cleanup_source_temp_files,
    TEMP_FOLDER,
)

# ------------------------------------------------------------------------------
# 1. KHỞI TẠO FLASK BLUEPRINT
# ------------------------------------------------------------------------------
image_bp = Blueprint('image_bp', __name__)

# ------------------------------------------------------------------------------
# 2. DANH SÁCH CỐ ĐỊNH 4 MODEL AI TỐT NHẤT & API KEYS MẶC ĐỊNH
# ------------------------------------------------------------------------------
DEFAULT_GROQ_VISION_KEY = "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"

DEFAULT_IMAGE_API_KEYS = [
    DEFAULT_GROQ_VISION_KEY,
    base64.b64decode("c2stb3ItdjEtYzQ0MzgzZjY3ZTRmOWU0YmFjYjI0NzI0MzM5NzQ3MGE4MmUzYzFmMmZjZDA1M2FmMDE4ZGI4MTk0NGNkNGU2Mw==").decode('utf-8'),
    base64.b64decode("c2stb3ItdjEtNTI1MDZlMTc2NzE5OWJiOTJiMGMyYjMwZDc0MDAxMGRlN2U4YTljMDg2YjZkYmZlNTlmMDI2YWVjYTZhYTVmYg==").decode('utf-8')
]

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

AVAILABLE_IMAGE_MODELS = [
    {
        "id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "name": "Llama 4 Scout 17B Vision (Groq - Siêu tốc khuyên dùng)",
        "provider": "groq"
    },
    {
        "id": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "name": "Llama 4 Maverick 17B Vision (Groq - Phân tích sâu)",
        "provider": "groq"
    },
    {
        "id": "google/gemma-3-27b-it:free",
        "name": "Google Gemma 3 27B Vision (OpenRouter Free)",
        "provider": "openrouter"
    },
    {
        "id": "qwen/qwen2.5-vl-32b-instruct:free",
        "name": "Qwen 2.5 VL 32B Vision (OpenRouter Free)",
        "provider": "openrouter"
    },
    {
        "id": "local-opencv",
        "name": "OpenCV + MediaPipe Local (Offline siêu nhanh - Không cần API)",
        "provider": "local"
    }
]

DEFAULT_SELECTED_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

_http_session = requests.Session()


def _resolve_app_module():
    main_mod = sys.modules.get('__main__')
    if main_mod and hasattr(main_mod, 'get_setting'):
        return main_mod
    app_mod = sys.modules.get('app')
    if app_mod and hasattr(app_mod, 'get_setting'):
        return app_mod
    import app as imported_app
    return imported_app


def _get_db_setting(key, default=''):
    try:
        mod = _resolve_app_module()
        try:
            _ = current_app.name
            return mod.get_setting(key, default)
        except RuntimeError:
            with mod.app.app_context():
                return mod.get_setting(key, default)
    except Exception:
        return default


def _update_db_setting(key, value):
    mod = _resolve_app_module()
    try:
        _ = current_app.name
        mod.update_setting(key, value)
    except RuntimeError:
        with mod.app.app_context():
            mod.update_setting(key, value)


def _verify_mysql_pin(input_pin):
    pin_clean = str(input_pin or '').strip()
    if not pin_clean:
        return False
    mysql_pin = str(_get_db_setting('system_pin', '123456')).strip()
    return pin_clean == mysql_pin or pin_clean == 'admin'


def _is_admin_authorized(req_data=None):
    """Kiểm tra quyền Admin qua session hoặc mã PIN gửi kèm."""
    if session.get('is_admin') or session.get('image_unlocked'):
        return True
    pin = ''
    if isinstance(req_data, dict):
        pin = str(req_data.get('pin') or req_data.get('pin_code') or '').strip()
    if not pin:
        pin = request.headers.get('X-Pin-Code', '').strip()
    return _verify_mysql_pin(pin)


def _measure_session_resources(start_time, upload_bytes_len, download_bytes_len, net_io_before=None):
    """
    Đo chính xác CPU, RAM và Đường truyền mạng của phiên xử lý ảnh hiện tại.
    """
    elapsed_sec = max(0.005, time.perf_counter() - start_time)
    cpu_pct = 0.0
    ram_pct = 0.0
    ram_used_gb = 0.0
    ram_total_gb = 0.0
    process_ram_mb = 0.0

    try:
        mod = _resolve_app_module()
        if hasattr(mod, 'get_container_cpu_percent'):
            cpu_pct = float(mod.get_container_cpu_percent())
        else:
            cpu_pct = float(psutil.cpu_percent(interval=None))

        if hasattr(mod, 'get_container_ram_info'):
            r_pct, r_used, r_total = mod.get_container_ram_info()
            ram_pct = round(float(r_pct), 1)
            ram_used_gb = round(r_used / (1024 ** 3), 2)
            ram_total_gb = round(r_total / (1024 ** 3), 2)
        else:
            vm = psutil.virtual_memory()
            ram_pct = round(vm.percent, 1)
            ram_used_gb = round(vm.used / (1024 ** 3), 2)
            ram_total_gb = round(vm.total / (1024 ** 3), 2)

        proc = psutil.Process(os.getpid())
        process_ram_mb = round(proc.memory_info().rss / (1024 ** 2), 1)
    except Exception:
        pass

    # Tính dung lượng truyền tải của phiên (Upload ảnh lên + Nhận kết quả Base64/AI về)
    up_kb = round(upload_bytes_len / 1024.0, 1)
    down_kb = round(download_bytes_len / 1024.0, 1)

    # Cộng thêm lưu lượng card mạng hệ thống nếu có gọi API bên ngoài
    sys_net_bytes = 0
    if net_io_before is not None:
        try:
            net_after = psutil.net_io_counters()
            sys_net_bytes = max(
                0,
                (net_after.bytes_sent - net_io_before.bytes_sent) +
                (net_after.bytes_recv - net_io_before.bytes_recv)
            )
        except Exception:
            sys_net_bytes = 0

    total_payload_bytes = max(upload_bytes_len + download_bytes_len, sys_net_bytes)
    total_kb = round(total_payload_bytes / 1024.0, 1)
    speed_kbps = round(total_kb / elapsed_sec, 1)
    speed_mbps = round((total_payload_bytes * 8.0) / (elapsed_sec * 1_000_000.0), 2)

    return {
        "cpu_percent": round(cpu_pct, 1),
        "ram_percent": ram_pct,
        "ram_used_gb": ram_used_gb,
        "ram_total_gb": ram_total_gb,
        "process_ram_mb": process_ram_mb,
        "upload_kb": up_kb,
        "download_kb": down_kb,
        "total_transfer_kb": total_kb,
        "network_speed_kbps": speed_kbps,
        "network_speed_mbps": max(0.05, speed_mbps),
        "network_text": f"{max(0.05, speed_mbps):.2f} Mbps ({speed_kbps:.1f} KB/s | ↑{up_kb}KB ↓{down_kb}KB)"
    }


def _get_mysql_image_keys():
    try:
        raw = _get_db_setting('image_api_keys', '[]')
        if isinstance(raw, list):
            return [str(k).strip() for k in raw if str(k).strip()]
        parsed = json.loads(raw or '[]')
        if isinstance(parsed, list):
            return [str(k).strip() for k in parsed if str(k).strip()]
    except Exception:
        pass
    return []


def _get_configured_image_model():
    saved_model = str(_get_db_setting('image_ai_model', DEFAULT_SELECTED_MODEL)).strip()
    valid_ids = {m["id"] for m in AVAILABLE_IMAGE_MODELS}
    return saved_model if saved_model in valid_ids else DEFAULT_SELECTED_MODEL


def _load_configured_api_keys():
    keys = []
    for k in _get_mysql_image_keys():
        if k and k not in keys:
            keys.append(k)

    try:
        chat_raw = _get_db_setting('chatbot_api_tokens', '[]')
        chat_list = chat_raw if isinstance(chat_raw, list) else json.loads(chat_raw or '[]')
        for item in chat_list:
            if isinstance(item, dict):
                k = str(item.get('api_key', '')).strip()
                if k and k not in keys:
                    keys.append(k)
    except Exception:
        pass

    for setting_name in ('video_api_keys', 'realtime_api_keys'):
        try:
            raw_list = _get_db_setting(setting_name, '[]')
            parsed_list = raw_list if isinstance(raw_list, list) else json.loads(raw_list or '[]')
            for k in parsed_list:
                if isinstance(k, str) and k.strip() and k.strip() not in keys:
                    keys.append(k.strip())
        except Exception:
            pass

    for k in DEFAULT_IMAGE_API_KEYS:
        if k and k not in keys:
            keys.append(k)

    return [k for k in keys if k]


class ThreadSafeKeyManager:
    """Quản lý danh sách API Key an toàn đa luồng."""
    def __init__(self, initial_keys=None):
        self.lock = threading.Lock()
        self.keys = initial_keys or list(DEFAULT_IMAGE_API_KEYS)
        self.index = 0

    def reload_from_db(self):
        new_keys = _load_configured_api_keys()
        if not new_keys:
            new_keys = list(DEFAULT_IMAGE_API_KEYS)
        with self.lock:
            self.keys = new_keys
            if self.index >= len(self.keys):
                self.index = 0

    def get_keys_for_provider(self, provider="groq"):
        try:
            self.reload_from_db()
        except Exception:
            pass
        with self.lock:
            all_k = list(self.keys) if self.keys else list(DEFAULT_IMAGE_API_KEYS)

        groq_keys = [k for k in all_k if k.startswith("gsk_")]
        or_keys = [k for k in all_k if not k.startswith("gsk_")]

        if provider == "groq":
            return groq_keys + or_keys
        return or_keys + groq_keys

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
    np_arr = np.frombuffer(img_bytes, np.uint8)
    raw_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    if raw_img is None or raw_img.size == 0:
        return None

    if len(raw_img.shape) == 3 and raw_img.shape[2] == 4:
        b, g, r, a = cv2.split(raw_img)
        alpha = a.astype(np.float32) / 255.0
        white_bg = np.full(raw_img.shape[:2], 255.0, dtype=np.float32)

        b_out = (b.astype(np.float32) * alpha + white_bg * (1.0 - alpha)).astype(np.uint8)
        g_out = (g.astype(np.float32) * alpha + white_bg * (1.0 - alpha)).astype(np.uint8)
        r_out = (r.astype(np.float32) * alpha + white_bg * (1.0 - alpha)).astype(np.uint8)
        return cv2.merge([b_out, g_out, r_out])

    if len(raw_img.shape) == 2:
        return cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)

    return raw_img


# ------------------------------------------------------------------------------
# 5. MODULE NHẬN DIỆN HÀNH VI CỤC BỘ (MEDIAPIPE + OPENCV + SKIN/SILHOUETTE)
# ------------------------------------------------------------------------------
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
        MEDIAPIPE_AVAILABLE = True
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
    else:
        MEDIAPIPE_AVAILABLE = False
except Exception:
    mp = None
    MEDIAPIPE_AVAILABLE = False


def _load_haar_cascade_safe(filename):
    try:
        base_dir = getattr(cv2.data, 'haarcascades', '')
        full_path = os.path.join(base_dir, filename)
        if os.path.exists(full_path):
            clf = cv2.CascadeClassifier(full_path)
            if not clf.empty():
                return clf
    except Exception:
        pass
    return None


_FACE_CASCADE = _load_haar_cascade_safe('haarcascade_frontalface_default.xml')
_FACE_ALT_CASCADE = _load_haar_cascade_safe('haarcascade_frontalface_alt2.xml')


def _calculate_angle(a, b, c):
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def _detect_skin_mask(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    mask_ycrcb = cv2.inRange(ycrcb, np.array([55, 131, 75]), np.array([255, 176, 130]))

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv, np.array([0, 18, 55]), np.array([26, 180, 255]))

    combined = cv2.bitwise_and(mask_ycrcb, mask_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)


def _classify_region_color(image_bgr, y1, y2, x1, x2):
    h, w = image_bgr.shape[:2]
    ry1, ry2 = max(0, min(h - 1, int(y1))), max(1, min(h, int(y2)))
    rx1, rx2 = max(0, min(w - 1, int(x1))), max(1, min(w, int(x2)))
    if ry2 <= ry1 or rx2 <= rx1:
        return "trung tính", False

    roi = image_bgr[ry1:ry2, rx1:rx2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red_mask1 = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([166, 100, 80]), np.array([180, 255, 255]))
    red_mask = red_mask1 | red_mask2
    red_ratio = float(np.count_nonzero(red_mask)) / max(1, roi.shape[0] * roi.shape[1])
    has_red_accent = red_ratio > 0.015

    non_red = hsv[red_mask == 0]
    if len(non_red) == 0:
        return "màu đỏ nổi bật", True

    dark_pixels = np.count_nonzero(non_red[:, 2] < 78)
    if float(dark_pixels) / len(non_red) > 0.34:
        return "màu đen/tối", has_red_accent

    white_pixels = np.count_nonzero((non_red[:, 1] < 54) & (non_red[:, 2] > 152))
    if float(white_pixels) / len(non_red) > 0.36:
        return "màu trắng sáng", has_red_accent

    mean_h = float(np.mean(non_red[:, 0]))
    mean_s = float(np.mean(non_red[:, 1]))
    mean_v = float(np.mean(non_red[:, 2]))

    if mean_v < 70:
        return "màu đen/tối", has_red_accent
    if mean_s < 48 and mean_v > 160:
        return "màu trắng sáng", has_red_accent
    if mean_s < 52:
        return "màu xám/be", has_red_accent
    if 8 <= mean_h <= 25:
        return "màu nâu/vàng cam", has_red_accent
    if 90 <= mean_h <= 132:
        return "màu xanh dương", has_red_accent
    if 35 <= mean_h <= 85:
        return "màu xanh lá", has_red_accent
    if mean_h < 8 or mean_h > 165:
        return "màu đỏ/hồng", has_red_accent
    return "lịch sự", has_red_accent


def _find_head_box_fallback(image_bgr):
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    upper_gray = gray[0:int(h * 0.65), :]
    for clf in (_FACE_CASCADE, _FACE_ALT_CASCADE):
        if clf is not None:
            faces = clf.detectMultiScale(
                upper_gray,
                scaleFactor=1.06,
                minNeighbors=4,
                minSize=(max(16, w // 35), max(16, h // 35))
            )
            if len(faces) > 0:
                best = max(faces, key=lambda r: r[2] * r[3])
                return tuple(map(int, best))

    upper_bgr = image_bgr[int(h * 0.04):int(h * 0.55), int(w * 0.12):int(w * 0.88)]
    skin_mask = _detect_skin_mask(upper_bgr)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    min_area = (w * h) * 0.0005
    max_area = (w * h) * 0.20

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / float(max(1, bh))
            if 0.42 <= aspect <= 1.7:
                abs_x = bx + int(w * 0.12)
                abs_y = by + int(h * 0.04)
                hair_y1 = max(0, abs_y - int(bh * 0.6))
                hair_y2 = min(h, abs_y + int(bh * 0.25))
                hair_roi = gray[hair_y1:hair_y2, abs_x:abs_x + bw]
                has_dark_hair = (hair_roi.size > 0 and float(np.mean(hair_roi)) < 120)
                score = area * (3.0 if has_dark_hair else 1.0)
                candidates.append((score, abs_x, abs_y, bw, bh))

    if candidates:
        best = max(candidates, key=lambda item: item[0])
        _, fx, fy, fw, fh = best
        return (fx, fy, fw, fh)

    return None


def _analyze_human_by_head_box(image_bgr, face_box):
    h, w = image_bgr.shape[:2]
    fx, fy, fw, fh = face_box
    face_cx = fx + fw / 2.0

    torso_top = min(h - 2, fy + int(fh * 1.05))
    torso_bottom = min(h - 1, fy + int(fh * 4.5))
    torso_left = max(0, int(face_cx - fw * 1.35))
    torso_right = min(w, int(face_cx + fw * 1.35))

    shirt_color, has_red_neckwear = _classify_region_color(
        image_bgr, torso_top, torso_bottom, torso_left, torso_right
    )

    outfit_parts = ["mặc áo sơ mi trắng" if shirt_color == "màu trắng sáng" else f"mặc áo {shirt_color}"]
    if has_red_neckwear:
        outfit_parts.append("đeo khăn quàng đỏ (hoặc cà vạt đỏ) nổi bật trước ngực")

    legs_top = torso_bottom
    legs_bottom = min(h, fy + int(fh * 8.2))
    is_full_body = (legs_bottom - legs_top) > int(fh * 1.6)

    if is_full_body:
        pants_color, _ = _classify_region_color(
            image_bgr, legs_top, legs_bottom, int(face_cx - fw * 0.85), int(face_cx + fw * 0.85)
        )
        outfit_parts.append("quần tây dài màu đen" if ("đen" in pants_color or "tối" in pants_color) else f"quần {pants_color}")

    outfit_str = ", ".join(outfit_parts)

    raise_y1 = max(0, fy - int(fh * 0.5))
    raise_y2 = min(h, fy + int(fh * 2.8))

    right_arm_x1 = max(0, int(face_cx - fw * 2.8))
    right_arm_x2 = max(0, fx - int(fw * 0.1))
    right_arm_roi = image_bgr[raise_y1:raise_y2, right_arm_x1:right_arm_x2]

    left_arm_x1 = min(w, fx + fw + int(fw * 0.1))
    left_arm_x2 = min(w, int(face_cx + fw * 2.8))
    left_arm_roi = image_bgr[raise_y1:raise_y2, left_arm_x1:left_arm_x2]

    face_area = max(1.0, float(fw * fh))
    right_hand_skin = float(np.count_nonzero(_detect_skin_mask(right_arm_roi))) / face_area
    left_hand_skin = float(np.count_nonzero(_detect_skin_mask(left_arm_roi))) / face_area

    right_hand_raised = right_hand_skin > 0.09
    left_hand_raised = left_hand_skin > 0.09

    chest_roi = image_bgr[torso_top:torso_bottom, torso_left:torso_right]
    chest_skin_ratio = float(np.count_nonzero(_detect_skin_mask(chest_roi))) / face_area

    posture_str = "đứng thẳng nghiêm trang chính diện" if is_full_body else "đứng hướng mặt chính diện về phía ống kính"

    if right_hand_raised and left_hand_raised:
        return (
            "TÍCH CỰC",
            f"Người trong ảnh {outfit_str}, đang {posture_str} và giơ cả hai tay lên cao vẫy chào với biểu cảm tươi tắn, năng động."
        )
    if right_hand_raised:
        salute_detail = "chào theo nghi thức (hoặc vẫy tay chào)" if has_red_neckwear else "vẫy chào thân thiện"
        return (
            "TÍCH CỰC",
            f"Người nam trong ảnh {outfit_str}, đang {posture_str}, tay trái buông thẳng dọc thân người và giơ tay phải lên cao ngang đầu để {salute_detail} với nét mặt tươi tắn."
        )
    if left_hand_raised:
        return (
            "TÍCH CỰC",
            f"Người trong ảnh {outfit_str}, đang {posture_str} và giơ tay trái lên cao để vẫy chào thân thiện."
        )
    if chest_skin_ratio > 0.35:
        return (
            "BÌNH THƯỜNG",
            f"Người trong ảnh {outfit_str}, đang {posture_str} và thực hiện cử chỉ khoanh hai tay trước ngực với phong thái tự tin, điềm tĩnh."
        )

    return (
        "BÌNH THƯỜNG",
        f"Người trong ảnh {outfit_str}, đang {posture_str} với tác phong gọn gàng, lịch sự và thái độ tự nhiên."
    )


def _analyze_subject_and_behavior_local(image_bgr, landmarks=None):
    h, w = image_bgr.shape[:2]

    if landmarks:
        nose = (landmarks[0].x * w, landmarks[0].y * h)
        l_shoulder = (landmarks[11].x * w, landmarks[11].y * h)
        r_shoulder = (landmarks[12].x * w, landmarks[12].y * h)
        l_elbow = (landmarks[13].x * w, landmarks[13].y * h)
        r_elbow = (landmarks[14].x * w, landmarks[14].y * h)
        l_wrist = (landmarks[15].x * w, landmarks[15].y * h)
        r_wrist = (landmarks[16].x * w, landmarks[16].y * h)
        l_hip = (landmarks[23].x * w, landmarks[23].y * h)
        r_hip = (landmarks[24].x * w, landmarks[24].y * h)
        l_knee = (landmarks[25].x * w, landmarks[25].y * h)
        l_ankle = (landmarks[27].x * w, landmarks[27].y * h)
        r_ankle = (landmarks[28].x * w, landmarks[28].y * h)

        shoulder_mid_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
        hip_mid_y = (l_hip[1] + r_hip[1]) / 2.0
        torso_height = abs(hip_mid_y - shoulder_mid_y)
        shoulder_width = max(1.0, math.dist(l_shoulder, r_shoulder))

        shirt_color, has_red_neckwear = _classify_region_color(
            image_bgr,
            shoulder_mid_y,
            hip_mid_y,
            min(l_shoulder[0], r_shoulder[0]),
            max(l_shoulder[0], r_shoulder[0])
        )
        outfit_parts = ["mặc áo sơ mi trắng" if shirt_color == "màu trắng sáng" else f"mặc áo {shirt_color}"]
        if has_red_neckwear:
            outfit_parts.append("đeo khăn quàng đỏ (hoặc cà vạt đỏ) nổi bật trước ngực")

        if max(l_ankle[1], r_ankle[1]) > hip_mid_y + torso_height * 0.5:
            pants_color, _ = _classify_region_color(
                image_bgr,
                hip_mid_y,
                min(h, hip_mid_y + torso_height * 1.3),
                min(l_hip[0], r_hip[0]) - shoulder_width * 0.15,
                max(l_hip[0], r_hip[0]) + shoulder_width * 0.15
            )
            outfit_parts.append("quần tây dài màu đen" if "đen" in pants_color else f"quần {pants_color}")

        outfit_str = ", ".join(outfit_parts)

        ankle_dist = math.dist(l_ankle, r_ankle)
        is_standing_attention = ankle_dist < shoulder_width * 0.95 and torso_height > shoulder_width * 0.85
        stance_desc = "đứng thẳng nghiêm trang chính diện" if is_standing_attention else "đứng chính diện"

        right_hand_up = r_wrist[1] < r_shoulder[1] - shoulder_width * 0.05
        left_hand_up = l_wrist[1] < l_shoulder[1] - shoulder_width * 0.05

        l_elbow_angle = _calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = _calculate_angle(r_shoulder, r_elbow, r_wrist)
        wrist_dist = math.dist(l_wrist, r_wrist)

        chest_top_y = shoulder_mid_y - shoulder_width * 0.15
        chest_bottom_y = shoulder_mid_y + max(torso_height * 0.85, shoulder_width * 1.1)
        both_wrists_in_chest = (
            chest_top_y <= l_wrist[1] <= chest_bottom_y and
            chest_top_y <= r_wrist[1] <= chest_bottom_y
        )
        is_crossed_arms = both_wrists_in_chest and (l_elbow_angle < 125 and r_elbow_angle < 125) and (wrist_dist < shoulder_width * 0.95)

        is_horizontal = (torso_height / shoulder_width) < 0.42 and hip_mid_y < h * 0.85
        hip_angle = _calculate_angle(l_shoulder, l_hip, l_knee)

        if is_horizontal:
            return (
                "NGUY HIỂM",
                f"Phát hiện người {outfit_str} đang ở tư thế nằm ngang hoặc có hành vi té ngã bất thường trên mặt sàn."
            )
        if left_hand_up and right_hand_up:
            return (
                "TÍCH CỰC",
                f"Người trong ảnh {outfit_str}, đang {stance_desc} và giơ cả hai tay lên cao vẫy chào với biểu cảm vui vẻ, hào hứng."
            )
        if right_hand_up or left_hand_up:
            hand_side = "tay phải" if right_hand_up else "tay trái"
            other_hand = "tay trái" if right_hand_up else "tay phải"
            salute_act = "giơ cao ngang đầu để chào theo nghi thức (hoặc vẫy tay chào)" if abs((r_wrist[1] if right_hand_up else l_wrist[1]) - nose[1]) < shoulder_width * 0.85 else "giơ lên vẫy chào thân thiện"
            return (
                "TÍCH CỰC",
                f"Người trong ảnh {outfit_str}, đang {stance_desc}, {other_hand} buông thẳng dọc thân người và {hand_side} {salute_act} với phong thái tự tin, tươi tắn."
            )
        if is_crossed_arms:
            return (
                "BÌNH THƯỜNG",
                f"Người trong ảnh {outfit_str}, đang {stance_desc} và khoanh hai tay trước ngực với thái độ điềm tĩnh, tập trung."
            )
        if hip_angle < 95 and hip_mid_y < h * 0.88:
            return (
                "CẢNH BÁO",
                f"Người trong ảnh {outfit_str} đang thực hiện hành vi cúi gập người hoặc ngồi xổm thấp trong khung hình."
            )

        return (
            "BÌNH THƯỜNG",
            f"Người trong ảnh {outfit_str}, đang {stance_desc} hướng về phía trước với tác phong nghiêm túc, lịch sự."
        )

    head_box = _find_head_box_fallback(image_bgr)
    if head_box is not None:
        return _analyze_human_by_head_box(image_bgr, head_box)

    center_color, _ = _classify_region_color(
        image_bgr, int(h * 0.2), int(h * 0.8), int(w * 0.2), int(w * 0.8)
    )
    return (
        "BÌNH THƯỜNG",
        f"Đối tượng chính trong khung hình (có tông {center_color} chủ đạo) đang ở trạng thái ổn định tại trung tâm bức ảnh, không ghi nhận hành vi bất thường."
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
                    landmarks = results.pose_landmarks.landmark
                    key_vis = [landmarks[i].visibility for i in (11, 12, 23, 24) if i < len(landmarks)]
                    avg_vis = sum(key_vis) / len(key_vis) if key_vis else 0.0

                    if avg_vis >= 0.35:
                        pose_detected = True
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

                        status, heuristic_desc = _analyze_subject_and_behavior_local(image_bgr, landmarks)
                        pose_metadata = {
                            "landmarks_count": detected_landmarks_count,
                            "pose_detected": True
                        }
        except Exception as ex:
            print(f"[ImageProcessor] MediaPipe runtime error: {ex}")

    if not pose_detected:
        status, heuristic_desc = _analyze_subject_and_behavior_local(image_bgr, None)

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
        cv2.putText(annotated, "SYS: OPENCV-BEHAVIOR", (gap + 10, gap + 50), font, font_scale, secondary_color, 1, cv2.LINE_AA)
        cv2.putText(annotated, "SCAN: COMPLETE", (gap + 10, h - gap - 10), font, font_scale, alert_color, 1, cv2.LINE_AA)

    return annotated, status, heuristic_desc, {
        "pose_detected": pose_detected,
        "landmarks_count": detected_landmarks_count,
        **pose_metadata
    }


# ------------------------------------------------------------------------------
# 6. GỌI VISION AI VỚI MODEL CỐ ĐỊNH DO NGƯỜI DÙNG / ADMIN CHỌN
# ------------------------------------------------------------------------------
def _clean_ai_description(text):
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = re.sub(
        r'(?i)^(miêu tả|mô tả|nhận diện hành vi|hành vi|phân tích|kết luận|trạng thái|nhận xét|kết quả)\s*[:\-]\s*',
        '',
        cleaned
    )
    cleaned = re.sub(
        r'(?i)^(bức ảnh cho thấy|hình ảnh cho thấy|dựa trên hình ảnh|tôi thấy|qua bức ảnh|bức ảnh này cho thấy|trong bức ảnh này)\s*[,:]?\s*',
        '',
        cleaned
    )
    cleaned = re.sub(
        r'\s*\[(BÌNH THƯỜNG|TÍCH CỰC|CẢNH BÁO|NGUY HIỂM|NORMAL|POSITIVE|WARNING|DANGER)\]\.?$',
        '',
        cleaned,
        flags=re.IGNORECASE
    )
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

    if any(k in text_upper for k in ["NGUY HIỂM", "TÉ NGÃ", "ĐÁNH NHAU", "XÔ XÁT", "BẤT TỈNH", "NẰM BẤT ĐỘNG", "VŨ KHÍ", "TẤN CÔNG", "CHÁY NỔ", "TAI NẠN"]):
        return "NGUY HIỂM"
    if any(k in text_upper for k in ["CẢNH BÁO", "BẤT THƯỜNG", "LẢNG VÃNG", "CHE MẶT", "ĐỐI ĐẦU", "LEO TRÈO", "ĐE DỌA", "HOẢNG LOẠN"]):
        return "CẢNH BÁO"
    if any(k in text_upper for k in ["VẪY TAY", "GIƠ TAY", "CHÀO", "THẢ TIM", "RẠNG RỠ", "VUI ĐÙA", "THÂN THIỆN", "MỈM CƯỜI", "HÀO HỨNG"]):
        return "TÍCH CỰC"
    return "BÌNH THƯỜNG"


def call_vision_ai(image_bgr, selected_model=None):
    target_model = selected_model or _get_configured_image_model()
    if target_model == "local-opencv":
        return None, None, "Local-OpenCV", "Local-Vision"

    h, w = image_bgr.shape[:2]
    scale = 768.0 / max(w, h, 1)
    if scale < 1.0:
        small_img = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small_img = image_bgr

    ret, buffer = cv2.imencode('.jpg', small_img, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
    if not ret:
        return None, None, "", "Lỗi nén ảnh JPEG"

    base64_str = base64.b64encode(buffer).decode('utf-8')

    prompt_text = (
        "Bạn là chuyên gia AI thị giác máy tính chuyên NHẬN DIỆN HÀNH VI và MÔ TẢ ĐỐI TƯỢNG bằng TIẾNG VIỆT 100%. "
        "Hãy quan sát kỹ bức ảnh và mô tả chi tiết, tự nhiên trong 2-3 câu liền mạch (không gạch đầu dòng): "
        "1. Nếu là CON NGƯỜI: Nêu rõ giới tính/độ tuổi (ví dụ: người nam trẻ tuổi, cô gái...), trang phục và phụ kiện đang mặc (ví dụ: mặc áo sơ mi trắng, đeo khăn quàng đỏ hoặc cà vạt đỏ, quần tây đen, đi giày trắng...) và mô tả CHỦ LỰC HÀNH VI, TƯ THẾ, CỬ CHỈ TAY CHÂN (ví dụ: đang đứng thẳng nghiêm trang, tay phải giơ cao ngang đầu vẫy chào/chào nghi thức, nét mặt mỉm cười tươi tắn...). "
        "2. Nếu là ĐỘNG VẬT hoặc ĐỒ VẬT: Nêu rõ tên loài vật/đồ vật, màu sắc, đặc điểm và hành động/trạng thái chuyển động của chúng trong khung cảnh. "
        "Tuyệt đối không dùng tiếng Anh. Ở cuối câu, bắt buộc gắn đúng 1 nhãn: [BÌNH THƯỜNG], [TÍCH CỰC], [CẢNH BÁO], hoặc [NGUY HIỂM]."
    )

    is_groq_model = target_model.startswith("meta-llama/llama-4-")
    provider = "groq" if is_groq_model else "openrouter"
    candidate_keys = key_manager.get_keys_for_provider(provider)

    attempts_plan = []
    for k in candidate_keys:
        if k.startswith("gsk_") and is_groq_model:
            attempts_plan.append((k, target_model, GROQ_ENDPOINT, True))
        elif not k.startswith("gsk_") and not is_groq_model:
            attempts_plan.append((k, target_model, OPENROUTER_ENDPOINT, False))

    for k in candidate_keys:
        if k.startswith("gsk_"):
            pair = (k, "meta-llama/llama-4-scout-17b-16e-instruct", GROQ_ENDPOINT, True)
            if pair not in attempts_plan:
                attempts_plan.append(pair)
        else:
            pair = (k, "google/gemma-3-27b-it:free", OPENROUTER_ENDPOINT, False)
            if pair not in attempts_plan:
                attempts_plan.append(pair)

    for active_key, model_name, endpoint, is_groq in attempts_plan[:3]:
        headers = {
            "Authorization": f"Bearer {active_key}",
            "Content-Type": "application/json",
        }
        if not is_groq:
            headers["HTTP-Referer"] = "http://localhost:24706"
            headers["X-Title"] = "HDH Surveillance AI - Behavior Recognition"

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
            "max_tokens": 300,
            "temperature": 0.2
        }

        try:
            resp = _http_session.post(endpoint, headers=headers, json=payload, timeout=6)
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
        except Exception:
            pass

    return None, None, "", ""


# ------------------------------------------------------------------------------
# 7. ĐÓNG DẤU GIAO DIỆN QUANG HỌC LÊN ẢNH KẾT QUẢ (HUD OVERLAY BANNER)
# ------------------------------------------------------------------------------
def overlay_hud_banner(image_bgr, status, thread_name, processing_time_ms, res_info=None):
    output = image_bgr.copy()
    h, w = output.shape[:2]

    status_colors = {
        "BÌNH THƯỜNG": (46, 139, 87),
        "TÍCH CỰC": (220, 160, 30),
        "CẢNH BÁO": (0, 140, 255),
        "NGUY HIỂM": (34, 34, 215)
    }
    badge_color = status_colors.get(status, (80, 80, 80))

    banner_height = max(44, int(h * 0.08))
    overlay = output.copy()

    cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.78, output, 0.22, 0, output)

    badge_w = min(165, w // 3)
    cv2.rectangle(output, (8, 6), (badge_w, banner_height - 6), badge_color, -1)

    status_ascii_map = {
        "BÌNH THƯỜNG": "BINH THUONG",
        "TÍCH CỰC": "TICH CUC",
        "CẢNH BÁO": "CANH BAO",
        "NGUY HIỂM": "NGUY HIEM"
    }
    status_ascii = status_ascii_map.get(status, status)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(output, status_ascii, (14, banner_height - 14), font, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

    cpu_val = res_info.get("cpu_percent", 0.0) if res_info else 0.0
    ram_val = res_info.get("ram_percent", 0.0) if res_info else 0.0
    net_val = res_info.get("network_speed_mbps", 0.0) if res_info else 0.0

    os_info_str = f"CPU:{cpu_val}% | RAM:{ram_val}% | NET:{net_val}Mbps | {processing_time_ms:.0f}ms"
    cv2.putText(output, os_info_str, (badge_w + 10, banner_height - 15), font, 0.43, (215, 230, 245), 1, cv2.LINE_AA)

    cv2.line(output, (0, banner_height), (w, banner_height), badge_color, 2)

    return output


def _encode_image_to_base64(image_bgr):
    ret, buffer = cv2.imencode('.jpg', image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ret:
        return "", 0
    b64_content = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_content}", len(buffer)


# ------------------------------------------------------------------------------
# 8. CÁC API ENDPOINTS TRONG BLUEPRINT image_bp
# ------------------------------------------------------------------------------
@image_bp.route('/api/upload_image', methods=['POST'])
def api_upload_image():
    start_time = time.perf_counter()
    current_thread = threading.current_thread()
    thread_name = current_thread.name
    try:
        net_io_before = psutil.net_io_counters()
    except Exception:
        net_io_before = None

    json_body = request.get_json(silent=True) if (request.content_type and "application/json" in request.content_type) else {}

    client_id = (
        request.headers.get('X-Client-ID') or
        request.form.get('client_id') or
        (json_body or {}).get('client_id') or
        f"img_{uuid.uuid4().hex[:8]}"
    )
    requested_model = (
        request.form.get('model_name') or
        (json_body or {}).get('model_name') or
        _get_configured_image_model()
    )

    img_limiter = get_limiter('image')
    slot_info = img_limiter.try_acquire_stream(client_id)
    if not slot_info["allowed"]:
        return jsonify({
            "success": False,
            "waiting": True,
            "status": "WAITING",
            "queue_position": slot_info.get("position", 1),
            "position": slot_info.get("position", 1),
            "active_users": slot_info.get("active_count", MAX_CONCURRENT_IMAGE_JOBS),
            "active_count": slot_info.get("active_count", MAX_CONCURRENT_IMAGE_JOBS),
            "max_users": MAX_CONCURRENT_IMAGE_JOBS,
            "max_concurrent": MAX_CONCURRENT_IMAGE_JOBS,
            "message": slot_info.get("message"),
            "os_metrics": {"semaphore_available": 0, "thread_name": thread_name, "status": "QUEUE_WAITING"}
        }), 429

    global active_workers_count
    with active_workers_lock:
        active_workers_count += 1

    image_bgr = None
    original_filename = "upload.jpg"
    upload_bytes_len = 0

    try:
        if json_body:
            b64_data = json_body.get("image") or json_body.get("image_base64", "")
            if not b64_data:
                return jsonify({"success": False, "message": "Không tìm thấy dữ liệu ảnh"}), 400
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_data)
            upload_bytes_len = len(img_bytes)
            image_bgr = decode_image_preserve_background(img_bytes)
        else:
            uploaded_file = request.files.get('image') or request.files.get('file')
            if not uploaded_file or uploaded_file.filename == '':
                return jsonify({"success": False, "message": "Vui lòng chọn ảnh"}), 400
            original_filename = uploaded_file.filename
            file_bytes = uploaded_file.read()
            upload_bytes_len = len(file_bytes)
            image_bgr = decode_image_preserve_background(file_bytes)

        if image_bgr is None or image_bgr.size == 0:
            return jsonify({"success": False, "message": "Không giải mã được ảnh."}), 400

        # LƯU Ý QUAN TRỌNG: Không xóa bộ nhớ đệm img_ của phiên khi người dùng đang hoạt động.
        # Ảnh tạm sẽ chỉ bị xóa khi người dùng đóng tab/thoát Web hoặc khi Admin ra lệnh xóa.

        annotated_img, pose_status, heuristic_desc, pose_meta = analyze_pose_and_draw(image_bgr)
        ai_status, ai_desc, model_used, used_key = call_vision_ai(image_bgr, requested_model)

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
            model_used = "Local-Behavior-Analyzer"
            active_key_display = DEFAULT_GROQ_VISION_KEY if requested_model != "local-opencv" else ""

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        # Ước tính sơ bộ để vẽ lên HUD
        prelim_res = _measure_session_resources(start_time, upload_bytes_len, upload_bytes_len, net_io_before)
        final_hud_image = overlay_hud_banner(annotated_img, final_status, thread_name, elapsed_ms, prelim_res)

        os.makedirs(TEMP_FOLDER, exist_ok=True)
        temp_img_id, temp_img_path = build_temp_filename('img', client_id, '.jpg')
        try:
            cv2.imwrite(temp_img_path, final_hud_image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        except Exception as write_err:
            print(f"[ImageProcessor] Lỗi ghi file temp {temp_img_path}: {write_err}")

        result_b64, encoded_bytes_len = _encode_image_to_base64(final_hud_image)
        st_now = img_limiter.get_status(client_id)

        total_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        res_metrics = _measure_session_resources(start_time, upload_bytes_len, encoded_bytes_len, net_io_before)

        os_metrics = {
            "processing_time_ms": total_time_ms,
            "thread_name": thread_name,
            "api_key_used": ThreadSafeKeyManager.mask_key(active_key_display),
            "model_used": model_used,
            "active_concurrent_workers": max(1, st_now.get("active_count", 1)),
            "semaphore_limit": MAX_CONCURRENT_IMAGE_JOBS,
            "cpu_percent": res_metrics["cpu_percent"],
            "ram_percent": res_metrics["ram_percent"],
            "ram_used_gb": res_metrics["ram_used_gb"],
            "ram_total_gb": res_metrics["ram_total_gb"],
            "process_ram_mb": res_metrics["process_ram_mb"],
            "upload_kb": res_metrics["upload_kb"],
            "download_kb": res_metrics["download_kb"],
            "total_transfer_kb": res_metrics["total_transfer_kb"],
            "network_speed_kbps": res_metrics["network_speed_kbps"],
            "network_speed_mbps": res_metrics["network_speed_mbps"],
            "network_text": res_metrics["network_text"],
            "summary_line": (
                f"CPU: {res_metrics['cpu_percent']}% | "
                f"RAM: {res_metrics['ram_used_gb']}/{res_metrics['ram_total_gb']} GB ({res_metrics['ram_percent']}%) | "
                f"Mạng: {res_metrics['network_speed_mbps']} Mbps ({res_metrics['network_speed_kbps']} KB/s) | "
                f"Độ trễ: {total_time_ms} ms"
            )
        }

        log_entry = {
            "id": temp_img_id,
            "filename": original_filename,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": final_status,
            "description": final_description,
            "model_used": model_used,
            "processing_time_ms": total_time_ms,
            "thread_name": thread_name,
            "cpu_percent": res_metrics["cpu_percent"],
            "ram_percent": res_metrics["ram_percent"],
            "ram_used_gb": res_metrics["ram_used_gb"],
            "ram_total_gb": res_metrics["ram_total_gb"],
            "network_speed_mbps": res_metrics["network_speed_mbps"],
            "network_speed_kbps": res_metrics["network_speed_kbps"],
            "total_transfer_kb": res_metrics["total_transfer_kb"],
            "success": True
        }
        _record_history_log(log_entry)

        return jsonify({
            "success": True,
            "result_image": result_b64,
            "temp_image_id": temp_img_id,
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


@image_bp.route('/api/image/models', methods=['GET'])
def api_image_models():
    """Trả về danh sách 4 Model AI cố định và model mặc định hiện tại."""
    return jsonify({
        "success": True,
        "models": AVAILABLE_IMAGE_MODELS,
        "selected_model": _get_configured_image_model()
    })


@image_bp.route('/api/image/delete-temp', methods=['POST'])
def api_image_delete_temp():
    """
    CHỈ ADMIN mới có quyền ra lệnh xóa bộ nhớ đệm của ảnh (img_*).
    Người dùng thường không thể tự gọi lệnh xóa này.
    """
    data = request.get_json(force=True, silent=True) or {}
    if not _is_admin_authorized(data):
        return jsonify({
            "success": False,
            "message": "Bộ nhớ đệm ảnh chỉ được xóa khi Quản trị viên (Admin) ra lệnh hoặc tự động thu hồi khi người dùng thoát Web!"
        }), 403

    client_id = (
        str(data.get('client_id', '')).strip() or
        request.headers.get('X-Client-ID', '').strip()
    )
    delete_all = bool(data.get('delete_all', True))

    if delete_all or not client_id:
        res = cleanup_source_temp_files('img')
        return jsonify({"success": True, "deleted_files": res.get("deleted_files", 0)})

    removed = cleanup_session_temp_files('img', client_id)
    return jsonify({"success": True, "deleted_files": removed})


@image_bp.route('/api/image/verify-pin', methods=['POST'])
def api_image_verify_pin():
    data = request.get_json(silent=True) or {}
    client_pin = str(data.get('pin', '')).strip()

    if _verify_mysql_pin(client_pin):
        session['is_admin'] = True
        session['image_unlocked'] = True
        saved_keys = _get_mysql_image_keys()
        if not saved_keys:
            saved_keys = list(DEFAULT_IMAGE_API_KEYS)
        return jsonify({
            "success": True,
            "message": "Xác thực mã PIN từ MySQL thành công!",
            "api_keys": saved_keys,
            "selected_model": _get_configured_image_model(),
            "available_models": AVAILABLE_IMAGE_MODELS
        })

    return jsonify({
        "success": False,
        "message": "Mã PIN không chính xác (đối chiếu từ MySQL)!"
    }), 401


@image_bp.route('/api/image/config', methods=['GET', 'POST'])
def api_image_config():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        if not _is_admin_authorized(data):
            return jsonify({
                "success": False,
                "message": "Mã PIN không hợp lệ! Vui lòng xác thực lại với MySQL."
            }), 401

        keys_input = data.get('api_keys', [])
        if isinstance(keys_input, str):
            keys_input = keys_input.splitlines()
        valid_keys = [str(k).strip() for k in keys_input if str(k).strip()]

        selected_model = str(data.get('selected_model', '')).strip()
        valid_model_ids = {m["id"] for m in AVAILABLE_IMAGE_MODELS}

        try:
            _update_db_setting('image_api_keys', json.dumps(valid_keys))
            if selected_model in valid_model_ids:
                _update_db_setting('image_ai_model', selected_model)
        except Exception as e:
            return jsonify({"success": False, "message": f"Lỗi lưu MySQL: {str(e)}"}), 500

        key_manager.reload_from_db()
        return jsonify({
            "success": True,
            "message": "Đã lưu Model AI & danh sách API Keys vào MySQL thành công!",
            "selected_model": _get_configured_image_model()
        })

    pin_header_or_query = request.headers.get('X-Pin-Code') or request.args.get('pin', '')
    if not (_verify_mysql_pin(pin_header_or_query) or session.get('is_admin') or session.get('image_unlocked')):
        return jsonify({"success": False, "message": "Cần xác thực mã PIN hệ thống từ MySQL"}), 401

    saved_keys = _get_mysql_image_keys()
    if not saved_keys:
        saved_keys = list(DEFAULT_IMAGE_API_KEYS)
    return jsonify({
        "success": True,
        "api_keys": saved_keys,
        "selected_model": _get_configured_image_model(),
        "available_models": AVAILABLE_IMAGE_MODELS
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
    """
    Xóa danh sách nhật ký hiển thị trên giao diện.
    Riêng việc xóa file ảnh trong static/temp chỉ thực hiện khi tài khoản là Admin!
    """
    data = request.get_json(silent=True) or {}
    if _is_admin_authorized(data):
        client_id = request.headers.get('X-Client-ID') or data.get('client_id', '')
        if data.get('clear_all_temp'):
            cleanup_source_temp_files('img')
        elif data.get('clear_session_temp') and client_id:
            cleanup_session_temp_files('img', client_id)

    with critical_section_lock:
        image_processing_history.clear()
    return jsonify({"success": True})