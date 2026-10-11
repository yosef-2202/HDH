# -*- coding: utf-8 -*-
from flask import Blueprint, render_template, Response, request, jsonify, session
import cv2
import threading
import queue
import time
import base64
import requests
import json
import os
import re
import numpy as np
from datetime import datetime

# Khởi tạo Blueprint
realtime_bp = Blueprint('realtime_bp', __name__)

# ==============================================================================
# CẤU HÌNH HỆ THỐNG & OPENROUTER AI
# ==============================================================================
def _load_api_key():
    env_k = os.getenv("OPENROUTER_API_KEY", "").strip()
    if env_k:
        return env_k
    try:
        if os.path.exists("config_local.json"):
            with open("config_local.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("OPENROUTER_API_KEY", "").strip()
    except Exception:
        pass
    return ""

OPENROUTER_API_KEY = _load_api_key()
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "dots-studio/dots-3-note-preview:free"

FALLBACK_VISION_MODELS = [
    "deepseek/deepseek-v4-flash:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "openrouter/free"
]

AI_FRAME_INTERVAL = 10.0
AI_MIN_REQUEST_INTERVAL = 3.2

# Danh sách log sự kiện AI
system_ai_logs = []
logs_lock = threading.Lock()

def add_system_log(cam_id, cam_name, status_type, behavior_desc, raw_response=""):
    with logs_lock:
        log_entry = {
            "id": len(system_ai_logs) + 1,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "cam_id": cam_id,
            "cam_name": cam_name,
            "status": status_type,
            "description": behavior_desc,
            "raw": raw_response
        }
        system_ai_logs.insert(0, log_entry)
        if len(system_ai_logs) > 100:
            system_ai_logs.pop()


# ==============================================================================
# HÀNG ĐỢI AI (PRODUCER - CONSUMER PATTERN)
# ==============================================================================
ai_task_queue = queue.Queue(maxsize=30)

class AIWorker(threading.Thread):
    def __init__(self, api_key=OPENROUTER_API_KEY, model_name=DEFAULT_MODEL):
        super().__init__(name="Worker-OpenRouter-AI", daemon=True)
        self.api_key = api_key
        self.model_name = model_name
        self.last_api_call_time = 0.0
        self.is_running = True
        self.total_requests = 0

    def run(self):
        while self.is_running:
            try:
                task = ai_task_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                cam_id, cam_name, frame = task
                now = time.time()
                elapsed = now - self.last_api_call_time
                if elapsed < AI_MIN_REQUEST_INTERVAL:
                    time.sleep(AI_MIN_REQUEST_INTERVAL - elapsed)

                self.last_api_call_time = time.time()
                self.total_requests += 1

                status_type, description, raw_text = self._call_openrouter(frame)
                add_system_log(cam_id, cam_name, status_type, description, raw_text)

            except Exception as e:
                print(f"[AI-Worker Error] Lỗi xử lý: {e}")
            finally:
                ai_task_queue.task_done()

    def _call_openrouter(self, frame):
        try:
            h, w = frame.shape[:2]
            scale = 640.0 / max(w, 1)
            if scale < 1.0:
                frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                frame_small = frame

            ret, buffer = cv2.imencode('.jpg', frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ret:
                return "LỖI", "Không thể nén khung hình", ""

            base64_img = base64.b64encode(buffer).decode('utf-8')

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5000",
                "X-Title": "HDH Surveillance AI System"
            }

            prompt_text = (
                "Bạn là AI phân tích biểu cảm và hành vi thời gian thực qua camera máy tính. "
                "Hãy nhìn ảnh và miêu tả thật ngắn gọn trong 1 câu duy nhất (tiếng Việt tự nhiên, chuẩn chính tả):\n"
                "- Nếu nét mặt bình thường: Chỉ ghi là 'Bình thường' kèm hành động.\n"
                "- Nếu có biểu cảm: Ghi rõ 'Cười tươi', 'Mỉm cười', 'Nháy mắt', 'Ngáp', hoặc 'Nhắm mắt'.\n"
                "- Cử chỉ đi kèm: 'chống cằm', 'vẫy tay', 'che mặt', 'nhìn camera'..."
            )

            models_to_try = [self.model_name] + FALLBACK_VISION_MODELS
            last_err = ""

            for model_candidate in models_to_try:
                payload = {
                    "model": model_candidate,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_img}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 80,
                    "temperature": 0.2
                }

                try:
                    response = requests.post(
                        OPENROUTER_ENDPOINT,
                        headers=headers,
                        json=payload,
                        timeout=8
                    )

                    if response.status_code == 200:
                        res_json = response.json()
                        choices = res_json.get("choices", [])
                        if choices:
                            ai_content = choices[0].get("message", {}).get("content", "").strip()
                            ai_content = self._clean_text(ai_content)
                            status_type = self._parse_status_tag(ai_content)
                            return status_type, ai_content, f"[{model_candidate}] {ai_content}"

                    last_err = f"HTTP {response.status_code}: {response.text[:120]}"
                except Exception as ex:
                    last_err = str(ex)

            heuristic_status, heuristic_desc = self._heuristic_fallback(frame)
            return heuristic_status, heuristic_desc, last_err

        except Exception as e:
            return "CẢNH BÁO", f"Ngoại lệ xử lý: {str(e)[:50]}", str(e)

    def _clean_text(self, text):
        if not text:
            return "Bình thường"
        cleaned = text.strip()
        patterns = [
            (r'(?i)nghĩ[a-zà-ỹ]*\s*túc\s*/\s*bình\s*thường', 'Bình thường'),
            (r'(?i)nghiêm\s*túc\s*/\s*bình\s*thường', 'Bình thường'),
            (r'(?i)bình\s*thường\s*/\s*nghĩ[a-zà-ỹ]*\s*túc', 'Bình thường'),
            (r'(?i)bình\s*thường\s*/\s*nghiêm\s*túc', 'Bình thường'),
            (r'(?i)nghĩ[a-zà-ỹ]*\s*túc', 'bình thường'),
            (r'(?i)nghiêm\s*túc', 'bình thường'),
        ]
        for pat, repl in patterns:
            cleaned = re.sub(pat, repl, cleaned)
        cleaned = re.sub(r'(?i)^biểu cảm\s*(&\s*nét mặt)?\s*:\s*', '', cleaned)
        cleaned = re.sub(r'(?i);\s*cử chỉ\s*(&\s*hành động)?\s*:\s*', ', ', cleaned)
        cleaned = re.sub(r'(?i);\s*kết luận\s*:\s*', ', ', cleaned)
        cleaned = re.sub(r'[;\s]+$', '', cleaned).strip()
        cleaned = cleaned.replace('; ', ', ').replace(';', ', ')
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned

    def _parse_status_tag(self, text):
        text_upper = text.upper()
        if any(w in text_upper for w in ["NGUY HIỂM", "DANGER", "TÉ NGÃ", "ĐÁNH NHAU"]):
            return "NGUY HIỂM"
        elif any(w in text_upper for w in ["CẢNH BÁO", "WARNING", "BẤT THƯỜNG", "LẢNG VÃNG", "NGỦ GỤC"]):
            return "CẢNH BÁO"
        elif any(w in text_upper for w in ["CƯỜI", "NHÁY MẮT", "MỈM CƯỜI", "VUI VẺ"]):
            return "TÍCH CỰC"
        return "BÌNH THƯỜNG"

    def _heuristic_fallback(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        if brightness < 30:
            return "CẢNH BÁO", "Ánh sáng quá tối / Khuất tầm nhìn camera"
        elif brightness > 220:
            return "CẢNH BÁO", "Chói sáng bất thường / Camera có thể bị che"
        return "BÌNH THƯỜNG", "Người ngồi trước camera, biểu cảm bình thường"

ai_worker_thread = AIWorker()
ai_worker_thread.start()

# ==============================================================================
# CÁC ROUTE & API
# ==============================================================================

@realtime_bp.route('/realtime/view')
def realtime_dashboard():
    return render_template('realtime.html')

@realtime_bp.route('/api/realtime/upload_web_frame', methods=['POST'])
def api_upload_web_frame():
    data = request.get_json() or {}
    image_b64 = data.get('image', '')
    if not image_b64:
        return jsonify({"success": False})
    try:
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
        img_data = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        cam_id = data.get('cam_id', 'webcam_web')
        cam_name = data.get('name', 'Webcam Trình Duyệt')

        ai_task_queue.put_nowait((cam_id, cam_name, frame))
        return jsonify({"success": True})
    except queue.Full:
        return jsonify({"success": False, "message": "Hàng đợi AI đang bận"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@realtime_bp.route('/api/realtime/logs', methods=['GET'])
def api_get_logs():
    with logs_lock:
        return jsonify({
            "success": True,
            "logs": list(system_ai_logs[:50])
        })

@realtime_bp.route('/api/realtime/clear_logs', methods=['POST'])
def api_clear_logs():
    with logs_lock:
        system_ai_logs.clear()
    return jsonify({"success": True})

@realtime_bp.route('/api/realtime/config', methods=['GET', 'POST'])
def api_config():
    # CHẶN YÊU CẦU NẾU CHƯA NHẬP MÃ PIN
    if not session.get('realtime_unlocked'):
        return jsonify({"success": False, "message": "Cần xác thực mã PIN"}), 401
        
    global ai_worker_thread
    if request.method == 'POST':
        data = request.get_json() or {}
        api_key = data.get('api_key', '').strip()
        model = data.get('model', '').strip()
        if api_key:
            ai_worker_thread.api_key = api_key
        if model:
            ai_worker_thread.model_name = model
            
        # Lock lại sau khi lưu thành công để lần sau phải nhập lại PIN
        session.pop('realtime_unlocked', None)
        return jsonify({"success": True, "message": "Đã lưu cấu hình AI!"})

    return jsonify({
        "success": True,
        "api_key": ai_worker_thread.api_key[:10] + "..." if ai_worker_thread.api_key else "",
        "model": ai_worker_thread.model_name,
        "interval_seconds": AI_FRAME_INTERVAL,
        "min_request_interval": AI_MIN_REQUEST_INTERVAL,
        "total_requests": ai_worker_thread.total_requests
    })