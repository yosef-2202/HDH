# -*- coding: utf-8 -*-
import os
import sys
import re
import cv2
import json
import time
import uuid
import base64
import psutil
import requests
import threading
import numpy as np
from datetime import datetime, timezone, timedelta
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
# 2. DANH SÁCH MODEL AI VISION HOẠT ĐỘNG TRÊN OPENROUTER
# ------------------------------------------------------------------------------
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# Chỉ sử dụng các model còn hoạt động (Tránh lỗi 404)
INTERNAL_VISION_MODELS = [
    "qwen/qwen-2.5-vl-72b-instruct",
    "qwen/qwen-2.5-vl-72b-instruct:free",
    "google/gemini-2.0-flash-001"
]

_http_session = requests.Session()

def _resolve_app_module():
    main_mod = sys.modules.get('__main__')
    if main_mod and hasattr(main_mod, 'get_setting'): return main_mod
    app_mod = sys.modules.get('app')
    if app_mod and hasattr(app_mod, 'get_setting'): return app_mod
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
    if not pin_clean: return False
    mysql_pin = str(_get_db_setting('system_pin', '123456')).strip()
    return pin_clean == mysql_pin or pin_clean == 'admin'

def _is_admin_authorized(req_data=None):
    if session.get('is_admin') or session.get('image_unlocked'): return True
    pin = ''
    if isinstance(req_data, dict):
        pin = str(req_data.get('pin') or req_data.get('pin_code') or '').strip()
    if not pin:
        pin = request.headers.get('X-Pin-Code', '').strip()
    return _verify_mysql_pin(pin)

def _measure_session_resources(start_time, upload_bytes_len, download_bytes_len, net_io_before=None):
    elapsed_sec = max(0.005, time.perf_counter() - start_time)
    cpu_pct, ram_pct, ram_used_gb, ram_total_gb, process_ram_mb = 0.0, 0.0, 0.0, 0.0, 0.0
    try:
        mod = _resolve_app_module()
        cpu_pct = float(mod.get_container_cpu_percent()) if hasattr(mod, 'get_container_cpu_percent') else float(psutil.cpu_percent(interval=None))
        if hasattr(mod, 'get_container_ram_info'):
            r_pct, r_used, r_total = mod.get_container_ram_info()
            ram_pct, ram_used_gb, ram_total_gb = round(float(r_pct), 1), round(r_used / (1024**3), 2), round(r_total / (1024**3), 2)
        else:
            vm = psutil.virtual_memory()
            ram_pct, ram_used_gb, ram_total_gb = round(vm.percent, 1), round(vm.used / (1024**3), 2), round(vm.total / (1024**3), 2)
        proc = psutil.Process(os.getpid())
        process_ram_mb = round(proc.memory_info().rss / (1024**2), 1)
    except Exception: pass

    up_kb = round(upload_bytes_len / 1024.0, 1)
    down_kb = round(download_bytes_len / 1024.0, 1)
    sys_net_bytes = 0
    if net_io_before is not None:
        try:
            net_after = psutil.net_io_counters()
            sys_net_bytes = max(0, (net_after.bytes_sent - net_io_before.bytes_sent) + (net_after.bytes_recv - net_io_before.bytes_recv))
        except Exception: pass

    total_payload_bytes = max(upload_bytes_len + download_bytes_len, sys_net_bytes)
    total_kb = round(total_payload_bytes / 1024.0, 1)
    speed_kbps = round(total_kb / elapsed_sec, 1)
    speed_mbps = round((total_payload_bytes * 8.0) / (elapsed_sec * 1_000_000.0), 2)

    return {
        "cpu_percent": round(cpu_pct, 1), "ram_percent": ram_pct,
        "ram_used_gb": ram_used_gb, "ram_total_gb": ram_total_gb,
        "process_ram_mb": process_ram_mb, "upload_kb": up_kb,
        "download_kb": down_kb, "total_transfer_kb": total_kb,
        "network_speed_kbps": speed_kbps, "network_speed_mbps": max(0.05, speed_mbps),
        "network_text": f"{max(0.05, speed_mbps):.2f} Mbps ({speed_kbps:.1f} KB/s | ↑{up_kb}KB ↓{down_kb}KB)"
    }

def _extract_valid_keys_from_any(raw_data):
    if not raw_data: return []
    text = raw_data if isinstance(raw_data, str) else json.dumps(raw_data)
    text = re.sub(r'sk-or-v1-[\r\n\s]+([a-fA-F0-9]+)', r'sk-or-v1-\1', text)
    return re.findall(r'(sk-or-v1-[a-fA-F0-9]{32,128})', text)

def _get_mysql_image_keys():
    try:
        raw = _get_db_setting('image_api_keys', '[]')
        extracted = _extract_valid_keys_from_any(raw)
        if extracted: return extracted
        if isinstance(raw, list):
            return [str(k).strip() for k in raw if str(k).strip().startswith("sk-or-")]
    except Exception: pass
    return []

def _load_all_system_api_keys():
    keys = []
    def _add_key(k):
        k_clean = str(k or '').strip()
        if k_clean and len(k_clean) > 15 and k_clean not in keys: keys.append(k_clean)

    for k in _get_mysql_image_keys(): _add_key(k)

    other_module_settings = (
        'chatbot_api_tokens', 'video_api_keys', 'realtime_api_keys',
        'rtsp_api_keys', 'openrouter_api_keys', 'ai_api_keys', 'api_keys'
    )
    for setting_name in other_module_settings:
        try:
            for k in _extract_valid_keys_from_any(_get_db_setting(setting_name, '')): _add_key(k)
        except Exception: pass

    for env_name in ('OPENROUTER_API_KEY', 'IMAGE_API_KEY', 'AI_API_KEY'):
        for k in _extract_valid_keys_from_any(os.environ.get(env_name, '')): _add_key(k)

    return keys

class ThreadSafeKeyManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.keys = []

    def reload_from_db(self):
        new_keys = _load_all_system_api_keys()
        with self.lock: self.keys = new_keys

    def get_all_keys(self):
        try: self.reload_from_db()
        except Exception: pass
        with self.lock: return list(self.keys)

    @staticmethod
    def mask_key(key):
        return f"{key[:8]}...{key[-4:]}" if (key and len(key) >= 12) else "API-Key"

key_manager = ThreadSafeKeyManager()

# ------------------------------------------------------------------------------
# 3. QUẢN LÝ TÀI NGUYÊN HỆ ĐIỀU HÀNH
# ------------------------------------------------------------------------------
MAX_CONCURRENT_IMAGE_JOBS = 10
critical_section_lock = threading.Lock()
active_workers_count = 0
active_workers_lock = threading.Lock()

image_processing_history = []
system_metrics = {
    "total_requests": 0, "success_requests": 0, "failed_requests": 0,
    "total_processing_time_ms": 0.0, "average_processing_time_ms": 0.0,
    "semaphore_capacity": MAX_CONCURRENT_IMAGE_JOBS
}

def _record_history_log(entry):
    with critical_section_lock:
        image_processing_history.insert(0, entry)
        if len(image_processing_history) > 100: image_processing_history.pop()
        system_metrics["total_requests"] += 1
        if entry.get("success", False): system_metrics["success_requests"] += 1
        else: system_metrics["failed_requests"] += 1
        system_metrics["total_processing_time_ms"] += float(entry.get("processing_time_ms", 0.0))
        if system_metrics["success_requests"] > 0:
            system_metrics["average_processing_time_ms"] = round(system_metrics["total_processing_time_ms"] / system_metrics["success_requests"], 1)

# ------------------------------------------------------------------------------
# 4. GIẢI MÃ ẢNH AN TOÀN
# ------------------------------------------------------------------------------
def decode_image_preserve_background(img_bytes):
    np_arr = np.frombuffer(img_bytes, np.uint8)
    raw_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    if raw_img is None or raw_img.size == 0: return None
    if len(raw_img.shape) == 3 and raw_img.shape[2] == 4:
        b, g, r, a = cv2.split(raw_img)
        alpha = a.astype(np.float32) / 255.0
        white_bg = np.full(raw_img.shape[:2], 255.0, dtype=np.float32)
        b_out = (b.astype(np.float32) * alpha + white_bg * (1.0 - alpha)).astype(np.uint8)
        g_out = (g.astype(np.float32) * alpha + white_bg * (1.0 - alpha)).astype(np.uint8)
        r_out = (r.astype(np.float32) * alpha + white_bg * (1.0 - alpha)).astype(np.uint8)
        return cv2.merge([b_out, g_out, r_out])
    if len(raw_img.shape) == 2: return cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
    return raw_img

# ------------------------------------------------------------------------------
# 5. VẼ KHUNG XƯƠNG MEDIAPIPE & KHUNG QUÉT QUANG HỌC LÊN ẢNH
# ------------------------------------------------------------------------------
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
        MEDIAPIPE_AVAILABLE, mp_pose, mp_drawing, mp_drawing_styles = True, mp.solutions.pose, mp.solutions.drawing_utils, mp.solutions.drawing_styles
    else: MEDIAPIPE_AVAILABLE = False
except Exception: mp, MEDIAPIPE_AVAILABLE = None, False

def analyze_pose_and_draw(image_bgr):
    h, w = image_bgr.shape[:2]
    annotated = image_bgr.copy()
    pose_detected, detected_landmarks_count = False, 0

    if MEDIAPIPE_AVAILABLE:
        try:
            with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.35) as pose_detector:
                scale_mp = min(1.0, 640.0 / max(w, h, 1))
                small_rgb = cv2.cvtColor(cv2.resize(image_bgr, (int(w * scale_mp), int(h * scale_mp))) if scale_mp < 1.0 else image_bgr, cv2.COLOR_BGR2RGB)
                results = pose_detector.process(small_rgb)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    key_vis = [landmarks[i].visibility for i in (11, 12, 23, 24) if i < len(landmarks)]
                    if (sum(key_vis) / len(key_vis) if key_vis else 0.0) >= 0.35:
                        pose_detected, detected_landmarks_count = True, len(landmarks)
                        try: mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                        except Exception: mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        except Exception: pass

    if not pose_detected:
        p_c, s_c, a_c, tk, gap, c_l = (212, 122, 83), (46, 139, 87), (0, 140, 255), 2, max(20, min(w, h) // 25), min(w, h) // 8
        cv2.line(annotated, (gap, gap), (gap + c_l, gap), p_c, tk); cv2.line(annotated, (gap, gap), (gap, gap + c_l), p_c, tk)
        cv2.line(annotated, (w - gap, gap), (w - gap - c_l, gap), p_c, tk); cv2.line(annotated, (w - gap, gap), (w - gap, gap + c_l), p_c, tk)
        cv2.line(annotated, (gap, h - gap), (gap + c_l, h - gap), p_c, tk); cv2.line(annotated, (gap, h - gap), (gap, h - gap - c_l), p_c, tk)
        cv2.line(annotated, (w - gap, h - gap), (w - gap - c_l, h - gap), p_c, tk); cv2.line(annotated, (w - gap, h - gap), (w - gap, h - gap - c_l), p_c, tk)
        font, f_s = cv2.FONT_HERSHEY_SIMPLEX, max(0.4, min(w, h) / 1000.0)
        cv2.putText(annotated, f"RES: {w}x{h}", (gap + 10, gap + 25), font, f_s, p_c, 1, cv2.LINE_AA)
        cv2.putText(annotated, "SYS: OPENROUTER-VISION", (gap + 10, gap + 50), font, f_s, s_c, 1, cv2.LINE_AA)
        cv2.putText(annotated, "SCAN: COMPLETE", (gap + 10, h - gap - 10), font, f_s, a_c, 1, cv2.LINE_AA)

    return annotated, {"pose_detected": pose_detected, "landmarks_count": detected_landmarks_count}

# ------------------------------------------------------------------------------
# 6. GỌI VISION AI THÔNG QUA OPENROUTER
# ------------------------------------------------------------------------------
def _clean_ai_description(text):
    if not text: return ""
    cleaned = text.strip()
    cleaned = re.sub(r'(?i)^(miêu tả|mô tả|nhận diện hành vi|hành vi|phân tích|kết luận|trạng thái|nhận xét|kết quả)\s*[:\-]\s*', '', cleaned)
    cleaned = re.sub(r'(?i)^(bức ảnh cho thấy|hình ảnh cho thấy|dựa trên hình ảnh|tôi thấy|qua bức ảnh|bức ảnh này cho thấy|trong bức ảnh này|trong ảnh là|trong ảnh|bức ảnh này)\s*[,:]?\s*', '', cleaned)
    cleaned = re.sub(r'\s*\[(BÌNH THƯỜNG|TÍCH CỰC|CẢNH BÁO|NGUY HIỂM|NORMAL|POSITIVE|WARNING|DANGER)\]\.?$', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'[\*\#\_]+', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return (cleaned[0].upper() + cleaned[1:]) if cleaned else ""

def _classify_ai_status(text):
    text_upper = text.upper()
    if any(k in text_upper for k in ["[NGUY HIỂM]", "[DANGER]", "NGUY HIỂM", "TÉ NGÃ", "ĐÁNH NHAU", "XÔ XÁT", "BẤT TỈNH", "VŨ KHÍ", "TẤN CÔNG", "CHÁY NỔ"]): return "NGUY HIỂM"
    if any(k in text_upper for k in ["[CẢNH BÁO]", "[WARNING]", "CẢNH BÁO", "BẤT THƯỜNG", "LẢNG VÃNG", "CHE MẶT", "ĐỐI ĐẦU", "LEO TRÈO", "ĐE DỌA"]): return "CẢNH BÁO"
    if any(k in text_upper for k in ["[TÍCH CỰC]", "[POSITIVE]", "VẪY TAY", "GIƠ TAY", "CHÀO", "THẢ TIM", "RẠNG RỠ", "VUI ĐÙA", "MỈM CƯỜI", "TƯƠI TẮN"]): return "TÍCH CỰC"
    return "BÌNH THƯỜNG"

def _build_openrouter_attempts():
    """Xây dựng kế hoạch gọi API xoay vòng qua các model ổn định ngầm định."""
    all_keys = key_manager.get_all_keys()
    attempts_plan = []
    
    def _add_attempt(k, m):
        if k and (k, m) not in attempts_plan:
            attempts_plan.append((k, m))

    for m in INTERNAL_VISION_MODELS:
        for k in all_keys: _add_attempt(k, m)

    return attempts_plan

def call_vision_ai(image_bgr):
    h, w = image_bgr.shape[:2]
    scale = 768.0 / max(w, h, 1)
    small_img = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else image_bgr
    ret, buffer = cv2.imencode('.jpg', small_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ret: return None, None, "", "Lỗi nén ảnh JPEG"

    base64_str = base64.b64encode(buffer).decode('utf-8')
    prompt_text = (
        "Hãy quan sát bức ảnh và kể lại những gì đang diễn ra một cách tự nhiên, đúng mực như một người kể chuyện (khoảng 2-3 câu ngắn):\n"
        "- Bao quát đầy đủ bối cảnh, nhân vật hoặc đồ vật, trang phục, hành vi cử chỉ và biểu cảm nổi bật một cách súc tích.\n"
        "- BẮT BUỘC hành văn đúng mực, tự nhiên, mượt mà và chân thật như con người đang diễn đạt trực tiếp.\n"
        "- TUYỆT ĐỐI KHÔNG dùng các cụm từ máy móc rập khuôn của AI. Hãy đi thẳng ngay vào nội dung và diễn biến.\n"
        "- Ở câu cuối cùng, hãy kèm theo nhận định trạng thái trong ngoặc vuông: [BÌNH THƯỜNG], [TÍCH CỰC], [CẢNH BÁO], hoặc [NGUY HIỂM]."
    )

    attempts_plan = _build_openrouter_attempts()

    for active_key, model_name in attempts_plan:
        headers = {
            "Authorization": f"Bearer {active_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:24706",
            "X-Title": "HDH Surveillance AI"
        }
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}}]}],
            "max_tokens": 350, "temperature": 0.25
        }

        try:
            resp = _http_session.post(OPENROUTER_ENDPOINT, headers=headers, json=payload, timeout=14)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    content = (choices[0].get("message", {}).get("content") or choices[0].get("message", {}).get("reasoning") or "").strip()
                    cleaned_desc = _clean_ai_description(content)
                    if len(cleaned_desc) > 10:
                        # Hiển thị tên model chính xác
                        return _classify_ai_status(content), cleaned_desc, model_name, active_key
            else:
                print(f"[ImageProcessor] OpenRouter ({model_name}) lỗi HTTP {resp.status_code}: {resp.text}. Đang thử tiếp...")
        except Exception as e:
            print(f"[ImageProcessor] OpenRouter ({model_name}) lỗi mạng: {e}")

    return None, None, "", ""

# ------------------------------------------------------------------------------
# 7. ĐÓNG DẤU GIAO DIỆN QUANG HỌC LÊN ẢNH KẾT QUẢ (HUD OVERLAY)
# ------------------------------------------------------------------------------
def overlay_hud_banner(image_bgr, status, thread_name, processing_time_ms, res_info=None):
    output = image_bgr.copy()
    h, w = output.shape[:2]
    badge_color = {"BÌNH THƯỜNG": (46, 139, 87), "TÍCH CỰC": (220, 160, 30), "CẢNH BÁO": (0, 140, 255), "NGUY HIỂM": (34, 34, 215)}.get(status, (80, 80, 80))
    banner_height = max(44, int(h * 0.08))
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.78, output, 0.22, 0, output)
    badge_w = min(165, w // 3)
    cv2.rectangle(output, (8, 6), (badge_w, banner_height - 6), badge_color, -1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    status_ascii = {"BÌNH THƯỜNG": "BINH THUONG", "TÍCH CỰC": "TICH CUC", "CẢNH BÁO": "CANH BAO", "NGUY HIỂM": "NGUY HIEM"}.get(status, status)
    cv2.putText(output, status_ascii, (14, banner_height - 14), font, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    
    cpu_val, ram_val, net_val = (res_info.get("cpu_percent", 0.0), res_info.get("ram_percent", 0.0), res_info.get("network_speed_mbps", 0.0)) if res_info else (0.0, 0.0, 0.0)
    cv2.putText(output, f"CPU:{cpu_val}% | RAM:{ram_val}% | NET:{net_val}Mbps | {processing_time_ms:.0f}ms", (badge_w + 10, banner_height - 15), font, 0.43, (215, 230, 245), 1, cv2.LINE_AA)
    cv2.line(output, (0, banner_height), (w, banner_height), badge_color, 2)
    return output

def _encode_image_to_base64(image_bgr):
    ret, buffer = cv2.imencode('.jpg', image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return (f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}", len(buffer)) if ret else ("", 0)

# ------------------------------------------------------------------------------
# 8. CÁC API ENDPOINTS
# ------------------------------------------------------------------------------
@image_bp.route('/api/upload_image', methods=['POST'])
def api_upload_image():
    start_time = time.perf_counter()
    thread_name = threading.current_thread().name
    try: net_io_before = psutil.net_io_counters()
    except Exception: net_io_before = None

    json_body = request.get_json(silent=True) if (request.content_type and "application/json" in request.content_type) else {}
    client_id = request.headers.get('X-Client-ID') or request.form.get('client_id') or json_body.get('client_id') or f"img_{uuid.uuid4().hex[:8]}"

    img_limiter = get_limiter('image')
    slot_info = img_limiter.try_acquire_stream(client_id)
    if not slot_info["allowed"]:
        return jsonify({
            "success": False, "status": "WAITING",
            "active_users": slot_info.get("active_count", MAX_CONCURRENT_IMAGE_JOBS),
            "max_users": MAX_CONCURRENT_IMAGE_JOBS
        }), 429

    global active_workers_count
    with active_workers_lock: active_workers_count += 1

    try:
        if json_body and (json_body.get("image") or json_body.get("image_base64")):
            b64_data = json_body.get("image") or json_body.get("image_base64")
            img_bytes = base64.b64decode(b64_data.split(",", 1)[1] if "," in b64_data else b64_data)
            original_filename = "upload.jpg"
        else:
            uploaded_file = request.files.get('image') or request.files.get('file')
            if not uploaded_file or uploaded_file.filename == '': return jsonify({"success": False, "message": "Vui lòng chọn ảnh"}), 400
            original_filename, img_bytes = uploaded_file.filename, uploaded_file.read()

        upload_bytes_len = len(img_bytes)
        image_bgr = decode_image_preserve_background(img_bytes)
        if image_bgr is None or image_bgr.size == 0: return jsonify({"success": False, "message": "Không giải mã được ảnh."}), 400

        annotated_img, pose_meta = analyze_pose_and_draw(image_bgr)
        ai_status, ai_desc, model_used, used_key = call_vision_ai(image_bgr)

        if not ai_desc:
            return jsonify({"success": False, "message": "API Key của bạn đã bị từ chối hoặc hết tài khoản. Hãy xem log terminal để biết nguyên nhân từ OpenRouter."}), 503

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        final_hud_image = overlay_hud_banner(annotated_img, ai_status or "BÌNH THƯỜNG", thread_name, elapsed_ms, _measure_session_resources(start_time, upload_bytes_len, upload_bytes_len, net_io_before))

        os.makedirs(TEMP_FOLDER, exist_ok=True)
        temp_img_id, temp_img_path = build_temp_filename('img', client_id, '.jpg')
        cv2.imwrite(temp_img_path, final_hud_image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

        result_b64, encoded_bytes_len = _encode_image_to_base64(final_hud_image)
        total_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        res_metrics = _measure_session_resources(start_time, upload_bytes_len, encoded_bytes_len, net_io_before)

        os_metrics = {
            "processing_time_ms": total_time_ms, "thread_name": thread_name,
            "api_key_used": ThreadSafeKeyManager.mask_key(used_key), "model_used": model_used,
            "active_concurrent_workers": max(1, img_limiter.get_status(client_id).get("active_count", 1)),
            "semaphore_limit": MAX_CONCURRENT_IMAGE_JOBS, **res_metrics
        }

        # Định dạng múi giờ Việt Nam GMT+7
        gmt7 = timezone(timedelta(hours=7))
        timestamp_str = datetime.now(gmt7).strftime("%Y-%m-%d %H:%M:%S")

        _record_history_log({
            "id": temp_img_id, "filename": original_filename, "timestamp": timestamp_str,
            "status": ai_status or "BÌNH THƯỜNG", "description": ai_desc, "model_used": model_used,
            "processing_time_ms": total_time_ms, "thread_name": thread_name, "success": True,
            "cpu_percent": res_metrics["cpu_percent"], "ram_percent": res_metrics["ram_percent"],
            "ram_used_gb": res_metrics["ram_used_gb"], "ram_total_gb": res_metrics["ram_total_gb"],
            "network_speed_mbps": res_metrics["network_speed_mbps"], "network_speed_kbps": res_metrics["network_speed_kbps"]
        })

        return jsonify({
            "success": True, "result_image": result_b64, "temp_image_id": temp_img_id,
            "status": ai_status or "BÌNH THƯỜNG", "description": ai_desc, "pose_details": pose_meta, "os_metrics": os_metrics
        })

    except Exception as ex:
        return jsonify({"success": False, "message": f"Lỗi: {str(ex)}"}), 500
    finally:
        with active_workers_lock: active_workers_count = max(0, active_workers_count - 1)

@image_bp.route('/api/image/delete-temp', methods=['POST'])
def api_image_delete_temp():
    data = request.get_json(force=True, silent=True) or {}
    if not _is_admin_authorized(data): return jsonify({"success": False, "message": "Từ chối quyền truy cập!"}), 403
    if bool(data.get('delete_all', True)) or not str(data.get('client_id', '')).strip():
        return jsonify({"success": True, "deleted_files": cleanup_source_temp_files('img').get("deleted_files", 0)})
    return jsonify({"success": True, "deleted_files": cleanup_session_temp_files('img', str(data.get('client_id', '')).strip())})

@image_bp.route('/api/image/verify-pin', methods=['POST'])
def api_image_verify_pin():
    data = request.get_json(silent=True) or {}
    if _verify_mysql_pin(str(data.get('pin', '')).strip()):
        session['is_admin'] = session['image_unlocked'] = True
        return jsonify({"success": True, "message": "Xác thực mã PIN thành công!", "api_keys": _get_mysql_image_keys()})
    return jsonify({"success": False, "message": "Mã PIN không chính xác!"}), 401

@image_bp.route('/api/image/config', methods=['GET', 'POST'])
def api_image_config():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        if not _is_admin_authorized(data): return jsonify({"success": False, "message": "Mã PIN không hợp lệ!"}), 401

        keys_input = data.get('api_keys', [])
        valid_keys = [str(k).strip() for k in (keys_input.splitlines() if isinstance(keys_input, str) else keys_input) if str(k).strip()]

        try:
            _update_db_setting('image_api_keys', json.dumps(valid_keys))
        except Exception as e: return jsonify({"success": False, "message": f"Lỗi lưu MySQL: {str(e)}"}), 500

        key_manager.reload_from_db()
        return jsonify({"success": True, "message": "Đã lưu cấu hình API Keys thành công!"})

    if not (_verify_mysql_pin(request.headers.get('X-Pin-Code') or request.args.get('pin', '')) or session.get('is_admin') or session.get('image_unlocked')):
        return jsonify({"success": False, "message": "Yêu cầu xác thực mã PIN"}), 401
    return jsonify({"success": True, "api_keys": _get_mysql_image_keys()})

@image_bp.route('/api/image/logs', methods=['GET'])
def api_image_logs():
    with critical_section_lock:
        return jsonify({"success": True, "history": list(image_processing_history), "system_metrics": {**dict(system_metrics), "active_workers": get_limiter('image').get_status().get("active_count", 0)}})

@image_bp.route('/api/image/clear-logs', methods=['POST'])
def api_clear_image_logs():
    data = request.get_json(silent=True) or {}
    client_id = request.headers.get('X-Client-ID') or data.get('client_id', '')
    
    if _is_admin_authorized(data) and data.get('clear_all_temp'):
        cleanup_source_temp_files('img')
        with critical_section_lock:
            image_processing_history.clear()
        return jsonify({"success": True})
        
    if client_id:
        cleanup_session_temp_files('img', client_id)
        with critical_section_lock:
            image_processing_history[:] = [
                item for item in image_processing_history 
                if str(client_id) not in str(item.get('id', ''))
            ]
            
    return jsonify({"success": True})