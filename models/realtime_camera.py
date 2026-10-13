# -*- coding: utf-8 -*-
"""
Module: Realtime Camera Surveillance (Phân tích Camera Thời gian thực)
Đồ án môn: Hệ điều hành (Multithreading, Synchronization & Load Balancing)
Mục tiêu tối ưu:
  1. Quản lý Đa phiên (Multi-Session): Hỗ trợ mượt mà >= 10 clients truy cập đồng thời.
     - Sử dụng mô hình Producer-Consumer với Hàng đợi đệm (Bounded Queue).
     - Hồ tiểu trình Worker (Thread Pool) chạy song song, xử lý không bị block chéo.
     - Tách biệt dữ liệu và nhật ký trạng thái theo từng Client ID / Session ID riêng biệt.
  2. Xoay vòng API Keys (Round-Robin Load Balancing):
     - Danh sách mảng 2-3 API Keys tự động luân phiên (Round-Robin).
     - Điều phối phân tải và tự động chuyển đổi (Failover) khi một key chạm Rate Limit.
"""

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
import uuid
import numpy as np
from datetime import datetime

# Khởi tạo Blueprint
realtime_bp = Blueprint('realtime_bp', __name__)

# ==============================================================================
# CẤU HÌNH HỆ THỐNG & DANH SÁCH API KEYS (LOAD BALANCING)
# ==============================================================================

# Danh sách API Keys mặc định (Các key nhạy cảm được nạp tự động từ config_local.json hoặc biến môi trường để bảo mật):
DEFAULT_CONFIGURED_KEYS = [
    # Bạn có thể cấu hình danh sách keys trong file config_local.json hoặc giao diện web
]

def _load_initial_api_keys():
    """Tự động tải danh sách API Keys từ file cấu hình hoặc biến môi trường"""
    keys = list(DEFAULT_CONFIGURED_KEYS)

    # 1. Đọc từ biến môi trường OPENROUTER_API_KEYS (chuỗi phân cách bằng dấu phẩy)
    env_keys_str = os.getenv("OPENROUTER_API_KEYS", "").strip()
    if env_keys_str:
        for k in env_keys_str.split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)

    # 2. Đọc từ biến môi trường đơn OPENROUTER_API_KEY
    env_single = os.getenv("OPENROUTER_API_KEY", "").strip()
    if env_single and env_single not in keys:
        keys.append(env_single)

    # 3. Đọc từ file config_local.json (nếu có)
    try:
        if os.path.exists("config_local.json"):
            with open("config_local.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                json_keys = data.get("OPENROUTER_API_KEYS", [])
                if isinstance(json_keys, list):
                    for k in json_keys:
                        k = str(k).strip()
                        if k and k not in keys:
                            keys.append(k)
                json_single = data.get("OPENROUTER_API_KEY", "").strip()
                if json_single and json_single not in keys:
                    keys.append(json_single)
    except Exception as e:
        print(f"[Cấu hình] Không thể đọc config_local.json: {e}")

    # Loại bỏ các key rỗng
    return [k for k in keys if k]


class KeyManager:
    """
    Bộ điều phối xoay vòng API Keys (Round-Robin Load Balancer)
    Bảo vệ bằng Mutex Lock (threading.Lock) đảm bảo Thread-safe khi nhiều Worker gọi cùng lúc.
    """
    def __init__(self, initial_keys=None):
        self.lock = threading.Lock()
        self.keys = initial_keys or []
        self.index = 0
        self.usage_counts = {k: 0 for k in self.keys}
        self.error_counts = {k: 0 for k in self.keys}

    def get_next_key(self):
        """Lấy API Key tiếp theo theo thuật toán Round-Robin"""
        with self.lock:
            if not self.keys:
                return ""
            key = self.keys[self.index % len(self.keys)]
            self.index = (self.index + 1) % len(self.keys)
            self.usage_counts[key] = self.usage_counts.get(key, 0) + 1
            return key

    def record_error(self, key):
        """Ghi nhận khi một key gặp lỗi (429 Rate Limit hoặc 401)"""
        with self.lock:
            if key in self.error_counts:
                self.error_counts[key] += 1

    def set_keys(self, new_keys):
        """Cập nhật danh sách keys mới (Thread-safe)"""
        with self.lock:
            valid_keys = [k.strip() for k in new_keys if k and str(k).strip()]
            if valid_keys:
                self.keys = valid_keys
                self.index = 0
                self.usage_counts = {k: 0 for k in self.keys}
                self.error_counts = {k: 0 for k in self.keys}

    def get_stats(self):
        """Thống kê trạng thái sử dụng của từng key"""
        with self.lock:
            return {
                "total_keys": len(self.keys),
                "current_index": self.index,
                "keys_preview": [
                    (k[:8] + "..." + k[-4:]) if len(k) > 12 else (k[:6] + "...")
                    for k in self.keys
                ],
                "usage_counts": [self.usage_counts.get(k, 0) for k in self.keys],
                "error_counts": [self.error_counts.get(k, 0) for k in self.keys]
            }


# Khởi tạo đối tượng KeyManager toàn cục
key_manager = KeyManager(_load_initial_api_keys())

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "dots-studio/dots-3-note-preview:free"

FALLBACK_VISION_MODELS = [
    "deepseek/deepseek-v4-flash:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "openrouter/free"
]

AI_FRAME_INTERVAL = 10.0
AI_MIN_REQUEST_INTERVAL = 1.0  # Nhờ có xoay vòng nhiều keys, khoảng cách giữa 2 request có thể rút ngắn xuống 1s

# ==============================================================================
# QUẢN LÝ ĐA PHIÊN (MULTI-SESSION & PER-CLIENT ISOLATION)
# ==============================================================================

# Lưu trạng thái gần nhất của từng phiên người dùng (Client ID -> Session Info)
client_sessions = {}
sessions_lock = threading.Lock()

def update_client_session(client_id, cam_name, status_type, behavior_desc):
    """Cập nhật trạng thái nhận diện riêng cho từng client (Thread-safe)"""
    with sessions_lock:
        client_sessions[client_id] = {
            "client_id": client_id,
            "cam_name": cam_name,
            "status": status_type,
            "description": behavior_desc,
            "updated_at": datetime.now().strftime("%H:%M:%S"),
            "last_active": time.time()
        }
        # Tự động dọn dẹp các phiên không hoạt động quá 30 phút
        now = time.time()
        expired = [cid for cid, s in client_sessions.items() if now - s.get("last_active", 0) > 1800]
        for cid in expired:
            del client_sessions[cid]

# Danh sách log sự kiện AI toàn hệ thống
system_ai_logs = []
logs_lock = threading.Lock()

def add_system_log(client_id, cam_id, cam_name, status_type, behavior_desc, raw_response=""):
    """Ghi nhận log có gắn kèm Client ID để tách biệt người dùng (Thread-safe)"""
    with logs_lock:
        log_entry = {
            "id": len(system_ai_logs) + 1,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "client_id": client_id,
            "cam_id": cam_id,
            "cam_name": cam_name,
            "status": status_type,
            "description": behavior_desc,
            "raw": raw_response
        }
        system_ai_logs.insert(0, log_entry)
        if len(system_ai_logs) > 200:
            system_ai_logs.pop()


# ==============================================================================
# HÀNG ĐỢI ĐA TIỂU TRÌNH (PRODUCER - CONSUMER WITH THREAD POOL)
# ==============================================================================
# Bounded Queue với dung lượng 60 tasks, đủ cho 10+ clients cùng đẩy frame
ai_task_queue = queue.Queue(maxsize=60)

class AIWorker(threading.Thread):
    """
    Consumer Thread trong Thread Pool:
    Chạy song song, cạnh tranh lấy task từ hàng đợi chung để phục vụ nhiều client cùng lúc.
    """
    def __init__(self, worker_id, key_mgr, model_name=DEFAULT_MODEL):
        super().__init__(name=f"Worker-AI-{worker_id}", daemon=True)
        self.worker_id = worker_id
        self.key_manager = key_mgr
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
                client_id, cam_id, cam_name, frame = task
                self.total_requests += 1

                # Lấy API Key xoay vòng theo thuật toán Round-Robin
                current_api_key = self.key_manager.get_next_key()

                # Gửi ảnh lên OpenRouter AI
                status_type, description, raw_text = self._call_openrouter(frame, current_api_key)

                # Cập nhật kết quả vào phiên riêng của client này
                update_client_session(client_id, cam_name, status_type, description)

                # Ghi vào nhật ký hệ thống
                add_system_log(client_id, cam_id, cam_name, status_type, description, raw_text)

            except Exception as e:
                print(f"[{self.name} Error] Lỗi xử lý task: {e}")
            finally:
                ai_task_queue.task_done()

    def _call_openrouter(self, frame, api_key):
        """Gửi khung hình lên AI OpenRouter với cơ chế xoay vòng và tự động thử lại khi lỗi"""
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

            prompt_text = (
                "Bạn là AI phân tích biểu cảm và hành vi thời gian thực qua camera máy tính. "
                "Hãy nhìn ảnh và miêu tả thật ngắn gọn trong 1 câu duy nhất (tiếng Việt tự nhiên, chuẩn chính tả):\n"
                "- Nếu nét mặt bình thường: Chỉ ghi là 'Bình thường' kèm hành động (ví dụ: 'Bình thường, đang nhìn vào màn hình'). Tuyệt đối KHÔNG dùng các từ như 'nghiêm túc' hay 'nghĩ/nghĩnh túc'.\n"
                "- Nếu có biểu cảm: Ghi rõ 'Cười tươi', 'Mỉm cười', 'Nháy mắt', 'Ngáp', hoặc 'Nhắm mắt'.\n"
                "- Cử chỉ đi kèm: 'chống cằm', 'vẫy tay', 'che mặt', 'nhìn camera'...\n"
                "Ví dụ câu mẫu chuẩn: 'Bình thường, đang nhìn vào camera' hoặc 'Mỉm cười, nhìn thẳng' hoặc 'Đang nháy mắt'."
            )

            models_to_try = [self.model_name] + FALLBACK_VISION_MODELS
            active_key = api_key
            last_err = ""

            for model_candidate in models_to_try:
                headers = {
                    "Authorization": f"Bearer {active_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:5000",
                    "X-Title": "HDH Surveillance AI System"
                }

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

                    # Thành công
                    if response.status_code == 200:
                        res_json = response.json()
                        choices = res_json.get("choices", [])
                        if choices:
                            ai_content = choices[0].get("message", {}).get("content", "").strip()
                            ai_content = self._clean_text(ai_content)
                            status_type = self._parse_status_tag(ai_content)
                            return status_type, ai_content, f"[{model_candidate}] {ai_content}"

                    # Nếu gặp lỗi Rate Limit (429) -> Tự động xoay sang Key dự phòng kế tiếp
                    elif response.status_code == 429:
                        self.key_manager.record_error(active_key)
                        backup_key = self.key_manager.get_next_key()
                        if backup_key and backup_key != active_key:
                            active_key = backup_key
                            continue

                    last_err = f"HTTP {response.status_code}: {response.text[:100]}"
                except Exception as ex:
                    last_err = str(ex)

            # Heuristic dự phòng khi ngoại vi bận
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


# ==============================================================================
# KHỞI TẠO HỒ TIỂU TRÌNH WORKER (THREAD POOL - MULTI-WORKERS)
# ==============================================================================
# Khởi chạy 3 Worker Threads chạy song song -> Đáp ứng 10+ client cùng lúc không bị nghẽn
NUM_WORKERS = 3
worker_pool = []

for i in range(NUM_WORKERS):
    w = AIWorker(worker_id=i + 1, key_mgr=key_manager, model_name=DEFAULT_MODEL)
    w.start()
    worker_pool.append(w)


# ==============================================================================
# CÁC ROUTE & API CỦA FLASK BLUEPRINT
# ==============================================================================

@realtime_bp.route('/realtime/view')
def realtime_dashboard():
    # Tự động gán Session ID duy nhất cho mỗi trình duyệt mở trang
    if 'client_id' not in session:
        session['client_id'] = f"client_{uuid.uuid4().hex[:8]}"
    return render_template('realtime.html')

@realtime_bp.route('/api/realtime/upload_web_frame', methods=['POST'])
def api_upload_web_frame():
    """
    Endpoint nhận frame từ Webcam trình duyệt:
    Hỗ trợ 10 người dùng đồng thời, định danh và tách biệt theo client_id.
    """
    data = request.get_json() or {}
    image_b64 = data.get('image', '')
    if not image_b64:
        return jsonify({"success": False, "message": "Thiếu dữ liệu ảnh"}), 400

    # Lấy hoặc tạo Client ID riêng biệt cho từng người dùng
    client_id = data.get('client_id', '').strip()
    if not client_id:
        if 'client_id' not in session:
            session['client_id'] = f"client_{uuid.uuid4().hex[:8]}"
        client_id = session['client_id']

    cam_id = data.get('cam_id', f'cam_{client_id}')
    cam_name = data.get('name', f'Webcam ({client_id[:8]})')

    try:
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
        img_data = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "message": "Không thể giải mã ảnh"}), 400

        # Đẩy vào hàng đợi chung của Producer-Consumer (Non-blocking)
        ai_task_queue.put_nowait((client_id, cam_id, cam_name, frame))
        return jsonify({"success": True, "client_id": client_id})

    except queue.Full:
        return jsonify({"success": False, "message": "Hàng đợi AI đang bận, vui lòng thử lại sau giây lát!"}), 429
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@realtime_bp.route('/api/realtime/logs', methods=['GET'])
def api_get_logs():
    """
    Lấy danh sách log:
    Nếu client truyền ?client_id=..., hệ thống lọc trả về log của riêng client đó
    để 10 người dùng không bị lẫn lộn kết quả phân tích của nhau.
    """
    client_id = request.args.get('client_id', '').strip()
    with logs_lock:
        if client_id and client_id != 'all':
            filtered = [
                l for l in system_ai_logs
                if l.get('client_id') == client_id or not l.get('client_id')
            ]
            return jsonify({"success": True, "logs": filtered[:50]})
        return jsonify({"success": True, "logs": list(system_ai_logs[:50])})

@realtime_bp.route('/api/realtime/sessions', methods=['GET'])
def api_get_sessions():
    """Thống kê các phiên người dùng đang hoạt động đồng thời (Dành cho báo cáo HĐH)"""
    with sessions_lock:
        sessions_list = list(client_sessions.values())

    return jsonify({
        "success": True,
        "active_clients_count": len(sessions_list),
        "sessions": sessions_list,
        "queue_size": ai_task_queue.qsize(),
        "active_workers": len(worker_pool),
        "key_stats": key_manager.get_stats()
    })

@realtime_bp.route('/api/realtime/clear_logs', methods=['POST'])
def api_clear_logs():
    client_id = request.args.get('client_id', '').strip()
    with logs_lock:
        if client_id and client_id != 'all':
            global system_ai_logs
            system_ai_logs = [l for l in system_ai_logs if l.get('client_id') != client_id]
        else:
            system_ai_logs.clear()
    return jsonify({"success": True})

@realtime_bp.route('/api/realtime/config', methods=['GET', 'POST'])
def api_config():
    """
    Cấu hình danh sách API Keys và Model:
    Bảo mật mã PIN qua session.get('realtime_unlocked').
    Hỗ trợ nhập 2-3 keys cách nhau bằng dấu phẩy hoặc xuống dòng để xoay vòng.
    """
    # CHẶN YÊU CẦU NẾU CHƯA NHẬP MÃ PIN
    if not session.get('realtime_unlocked'):
        return jsonify({"success": False, "message": "Cần xác thực mã PIN"}), 401

    global key_manager, worker_pool

    if request.method == 'POST':
        data = request.get_json() or {}
        # Hỗ trợ nhận 1 key hoặc danh sách nhiều keys (chuỗi hoặc mảng)
        keys_input = data.get('api_keys') or data.get('api_key', '')
        model = data.get('model', '').strip()

        if keys_input:
            if isinstance(keys_input, str):
                new_keys = [k.strip() for k in re.split(r'[,;\n]', keys_input) if k.strip()]
            elif isinstance(keys_input, list):
                new_keys = [str(k).strip() for k in keys_input if str(k).strip()]
            else:
                new_keys = []

            if new_keys:
                key_manager.set_keys(new_keys)

        if model:
            for w in worker_pool:
                w.model_name = model

        # Lock lại sau khi lưu thành công để lần sau phải nhập lại PIN
        session.pop('realtime_unlocked', None)

        return jsonify({
            "success": True,
            "message": f"Đã lưu thành công! Đang xoay vòng {len(key_manager.keys)} API Keys qua {len(worker_pool)} Worker Threads."
        })

    key_stats = key_manager.get_stats()
    return jsonify({
        "success": True,
        "total_keys": key_stats["total_keys"],
        "keys_preview": key_stats["keys_preview"],
        "key_stats": key_stats,
        "model": worker_pool[0].model_name if worker_pool else DEFAULT_MODEL,
        "active_workers": len(worker_pool),
        "queue_size": ai_task_queue.qsize(),
        "total_requests": sum(w.total_requests for w in worker_pool)
    })
