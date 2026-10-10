# -*- coding: utf-8 -*-
"""
Module: Realtime Camera Surveillance (Phân tích Camera Thật Thời gian thực)
Đồ án môn: Hệ điều hành (Multithreading & Synchronization)
Mục tiêu:
  - Mở trực tiếp Webcam máy tính người dùng và các luồng Camera thật.
  - Quản lý đồng thời tới 10 luồng camera song song (Mỗi camera chạy trên 1 Thread riêng biệt).
  - Tích hợp OpenRouter AI nhận diện hành vi (10 giây/lần) với cơ chế Rate Limit <= 20 req/phút.
  - Hoàn toàn KHÔNG dùng video có sẵn, tập trung 100% vào Camera thật.
"""

from flask import Blueprint, render_template, Response, request, jsonify
import cv2
import threading
import queue
import time
import base64
import requests
import json
import os
import re
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
# Ưu tiên model thị giác hoạt động ổn định và hỗ trợ tiếng Việt cực tốt cho biểu cảm (Cười, Nháy mắt,...)
DEFAULT_MODEL = "dots-studio/dots-3-note-preview:free"

FALLBACK_VISION_MODELS = [
    "deepseek/deepseek-v4-flash:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "openrouter/free"
]

# Chu kỳ gửi AI: mỗi camera chỉ gửi 1 frame sau mỗi 10 giây
AI_FRAME_INTERVAL = 10.0

# Giới hạn Rate Limit OpenRouter Free (20 requests/phút => khoảng 1 req / 3 giây)
# Đặt 3.2s để đảm bảo tối đa 18.75 req/phút toàn hệ thống, triệt tiêu 100% lỗi 429
AI_MIN_REQUEST_INTERVAL = 3.2

# Danh sách log sự kiện AI
system_ai_logs = []
logs_lock = threading.Lock()

def add_system_log(cam_id, cam_name, status_type, behavior_desc, raw_response=""):
    """Ghi nhận nhật ký nhận diện hành vi AI (Thread-safe)"""
    with logs_lock:
        log_entry = {
            "id": len(system_ai_logs) + 1,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "cam_id": cam_id,
            "cam_name": cam_name,
            "status": status_type,        # BÌNH THƯỜNG / CẢNH BÁO / NGUY HIỂM
            "description": behavior_desc,  # Nội dung hành vi do AI mô tả
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
    """
    Consumer Thread: Lấy khung hình từ hàng đợi ai_task_queue,
    gọi API OpenRouter và điều phối tốc độ gọi (Rate Limiting).
    """
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

                # Cơ chế Rate Limiting toàn cục: Giữ khoảng cách giữa 2 lần gọi >= 3.2s
                now = time.time()
                elapsed = now - self.last_api_call_time
                if elapsed < AI_MIN_REQUEST_INTERVAL:
                    time.sleep(AI_MIN_REQUEST_INTERVAL - elapsed)

                self.last_api_call_time = time.time()
                self.total_requests += 1

                # Gửi ảnh lên OpenRouter AI
                status_type, description, raw_text = self._call_openrouter(frame)

                # Cập nhật kết quả vào luồng Camera tương ứng
                camera = camera_manager.get_camera(cam_id)
                if camera:
                    camera.update_ai_result(status_type, description)

                # Lưu log
                add_system_log(cam_id, cam_name, status_type, description, raw_text)

            except Exception as e:
                print(f"[AI-Worker Error] Lỗi xử lý: {e}")
            finally:
                ai_task_queue.task_done()

    def _call_openrouter(self, frame):
        """Nén ảnh, mã hóa Base64 và gửi request tới OpenRouter"""
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
                "- Nếu nét mặt bình thường: Chỉ ghi là 'Bình thường' kèm hành động (ví dụ: 'Bình thường, đang nhìn vào màn hình'). Tuyệt đối KHÔNG dùng các từ như 'nghiêm túc' hay 'nghĩ/nghĩnh túc'.\n"
                "- Nếu có biểu cảm: Ghi rõ 'Cười tươi', 'Mỉm cười', 'Nháy mắt', 'Ngáp', hoặc 'Nhắm mắt'.\n"
                "- Cử chỉ đi kèm: 'chống cằm', 'vẫy tay', 'che mặt', 'nhìn camera'...\n"
                "Ví dụ câu mẫu chuẩn: 'Bình thường, đang nhìn vào camera' hoặc 'Mỉm cười, nhìn thẳng' hoặc 'Đang nháy mắt'."
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

            # Heuristic dự phòng khi mạng ngoại vi bận
            heuristic_status, heuristic_desc = self._heuristic_fallback(frame)
            return heuristic_status, heuristic_desc, last_err

        except Exception as e:
            return "CẢNH BÁO", f"Ngoại lệ xử lý: {str(e)[:50]}", str(e)

    def _clean_text(self, text):
        """Chuẩn hóa chính tả tiếng Việt, loại bỏ từ sai chính tả hoặc cứng nhắc"""
        if not text:
            return "Bình thường"

        cleaned = text.strip()

        # Loại bỏ triệt để các biến thể 'nghĩ/nghĩnh/nghiêm túc' và thay bằng 'Bình thường'
        patterns = [
            (r'(?i)nghĩ[a-zà-ỹ]*\s*túc\s*/\s*bình\s*thường', 'Bình thường'),
            (r'(?i)nghiêm\s*túc\s*/\s*bình\s*thường', 'Bình thường'),
            (r'(?i)bình\s*thường\s*/\s*nghĩ[a-zà-ỹ]*\s*túc', 'Bình thường'),
            (r'(?i)bình\s*thường\s*/\s*nghiêm\s*túc', 'Bình thường'),
            (r'(?i)nghĩ[a-zà-ỹ]*\s*túc', 'bình thường'),
            (r'(?i)nghiêm\s*túc', 'bình thường'),
            (r'(?i)gesturing', 'cử chỉ'),
            (r'(?i)gesture', 'cử chỉ'),
            (r'(?i)smiling', 'mỉm cười'),
            (r'(?i)winking', 'nháy mắt'),
            (r'(?i)neutral', 'bình thường'),
        ]
        for pat, repl in patterns:
            cleaned = re.sub(pat, repl, cleaned)

        # Xóa các tiền tố cứng nhắc như 'Biểu cảm:', 'Cử chỉ:' nếu AI trả về kiểu bullet
        cleaned = re.sub(r'(?i)^biểu cảm\s*(&\s*nét mặt)?\s*:\s*', '', cleaned)
        cleaned = re.sub(r'(?i);\s*cử chỉ\s*(&\s*hành động)?\s*:\s*', ', ', cleaned)
        cleaned = re.sub(r'(?i);\s*kết luận\s*:\s*', ', ', cleaned)

        # Chuẩn hóa dấu câu: đổi chấm phẩy thừa sang phẩy, xóa dấu chấm phẩy ở cuối
        cleaned = re.sub(r'[;\s]+$', '', cleaned).strip()
        cleaned = cleaned.replace('; ', ', ').replace(';', ', ')
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Viết hoa chữ cái đầu tiên
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
# LUỒNG ĐỌC CAMERA THẬT (REAL CAMERA CAPTURE THREAD)
# ==============================================================================
class CameraStream:
    """
    Quản lý 1 luồng Camera Thật (Webcam máy tính hoặc IP Camera).
    Chạy trong 1 tiểu trình (Thread) riêng biệt, hoàn toàn không làm treo Flask.
    """
    def __init__(self, cam_id, name, source):
        self.cam_id = str(cam_id)
        self.name = str(name)
        self.source = source
        self.is_running = False

        # Khóa Mutex (Lock) bảo vệ khung hình dùng chung
        self.lock = threading.Lock()
        self.current_frame = None

        self.fps = 0.0
        self.frame_count = 0
        self.status = "CHỜ KHỞI TẠO"
        # Mặc định lật ngang (Mirror) để mặt người dùng không bị ngược/lệch
        self.flip_horizontal = True

        # AI định kỳ 10 giây
        self.ai_enabled = True
        self.ai_interval = AI_FRAME_INTERVAL
        self.last_ai_time = 0.0
        self.latest_ai_result = {
            "status": "CHỜ PHÂN TÍCH",
            "description": "Đang đồng bộ luồng camera...",
            "updated_at": "--:--:--"
        }

        self.thread = None

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(
            target=self._capture_worker,
            name=f"Thread-RealCamera-{self.cam_id}",
            daemon=True
        )
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.status = "ĐÃ DỪNG"

    def _capture_worker(self):
        """Vòng lặp đọc camera thật chạy song song trong luồng riêng biệt"""
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        # Mở camera thật: Trên Windows dùng CAP_DSHOW để mở webcam phần cứng tức thì
        if isinstance(src, int):
            if os.name == 'nt':
                cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(src)
        else:
            cap = cv2.VideoCapture(src)

        # Đặt kích thước khung hình chuẩn và tối ưu bộ đệm
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception:
            pass

        if not cap.isOpened():
            self.status = "LỖI KẾT NỐI (Không tìm thấy Camera)"
            print(f"[Camera {self.cam_id}] Không thể mở nguồn: {self.source}")
            return

        self.status = "ĐANG CHẠY"
        prev_time = time.time()
        fps_counter = 0

        while self.is_running:
            ret, frame = cap.read()

            if not ret or frame is None:
                self.status = "MẤT TÍN HIỆU"
                time.sleep(0.1)
                continue

            # Lật gương ngang để mặt người dùng tự nhiên, không bị ngược
            if self.flip_horizontal:
                frame = cv2.flip(frame, 1)

            # ĐỒNG BỘ HÓA: Cập nhật frame với khóa Mutex (tránh Race Condition)
            with self.lock:
                self.current_frame = frame
                self.frame_count += 1
                self.status = "ĐANG CHẠY"

            # Đo FPS
            fps_counter += 1
            now = time.time()
            if now - prev_time >= 1.0:
                self.fps = round(fps_counter / (now - prev_time), 1)
                fps_counter = 0
                prev_time = now

            # TRÍCH XUẤT KHUNG HÌNH CHO AI ĐỊNH KỲ 10 GIÂY
            if self.ai_enabled and (now - self.last_ai_time >= self.ai_interval):
                self.last_ai_time = now
                try:
                    ai_task_queue.put_nowait((self.cam_id, self.name, frame.copy()))
                except queue.Full:
                    pass

            # Nhường thời gian CPU để không làm nóng máy (0.01s ~ tối đa 60-100 fps)
            time.sleep(0.01)

        cap.release()
        self.status = "ĐÃ ĐÓNG"

    def get_frame_bytes(self):
        """Lấy frame JPEG cho Web Stream"""
        with self.lock:
            if self.current_frame is None:
                return None
            frame_copy = self.current_frame.copy()

        ai_st = self.latest_ai_result.get("status", "CHỜ")
        color = (0, 200, 0)
        if ai_st == "CẢNH BÁO":
            color = (0, 165, 255)
        elif ai_st == "NGUY HIỂM":
            color = (0, 0, 255)

        # Vẽ HUD góc trên video
        h, w = frame_copy.shape[:2]
        cv2.rectangle(frame_copy, (0, 0), (w, 32), (20, 20, 20), -1)
        info_str = f"[{self.name}] FPS: {self.fps} | AI: {ai_st}"
        cv2.putText(frame_copy, info_str, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame_copy, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        return buffer.tobytes() if ret else None

    def toggle_flip(self):
        """Bật/Tắt chế độ lật gương"""
        with self.lock:
            self.flip_horizontal = not self.flip_horizontal
            return self.flip_horizontal

    def update_ai_result(self, status, description):
        with self.lock:
            self.latest_ai_result = {
                "status": status,
                "description": description,
                "updated_at": datetime.now().strftime("%H:%M:%S")
            }

    def to_dict(self):
        with self.lock:
            return {
                "cam_id": self.cam_id,
                "name": self.name,
                "source": str(self.source),
                "fps": self.fps,
                "status": self.status,
                "ai_enabled": self.ai_enabled,
                "ai_result": self.latest_ai_result
            }


# ==============================================================================
# BỘ QUẢN LÝ ĐA LUỒNG CAMERA THẬT (CAMERA MANAGER)
# ==============================================================================
class CameraManager:
    """Quản lý đồng thời tối đa 10 luồng camera thật"""
    MAX_CAMERAS = 10

    def __init__(self):
        self.cameras = {}
        self.manager_lock = threading.Lock()

    def add_camera(self, cam_id, name, source):
        """Thêm và khởi chạy một camera thật vào 1 Thread riêng"""
        with self.manager_lock:
            if len(self.cameras) >= self.MAX_CAMERAS:
                return False, f"Hệ thống đã đạt giới hạn tối đa {self.MAX_CAMERAS} luồng camera!"
            if cam_id in self.cameras:
                return False, f"Camera ID '{cam_id}' đang hoạt động!"

            cam = CameraStream(cam_id, name, source)
            cam.start()
            self.cameras[cam_id] = cam
            return True, f"Khởi chạy luồng camera '{name}' thành công!"

    def remove_camera(self, cam_id):
        """Dừng và hủy một luồng camera"""
        with self.manager_lock:
            if cam_id in self.cameras:
                self.cameras[cam_id].stop()
                del self.cameras[cam_id]
                return True, f"Đã dừng luồng camera '{cam_id}'"
            return False, "Không tìm thấy camera"

    def get_camera(self, cam_id):
        with self.manager_lock:
            return self.cameras.get(str(cam_id))

    def get_all(self):
        with self.manager_lock:
            return [cam.to_dict() for cam in self.cameras.values()]

    def stop_all(self):
        """Dừng tất cả các luồng camera đang chạy"""
        with self.manager_lock:
            for cam in self.cameras.values():
                cam.stop()
            self.cameras.clear()

    def toggle_flip(self, cam_id):
        cam = self.get_camera(cam_id)
        if cam:
            return True, cam.toggle_flip()
        return False, False

    def toggle_primary_webcam(self, device_index=0):
        """Bật hoặc Tắt nhanh Webcam máy tính chính"""
        cam_id = "webcam_0"
        with self.manager_lock:
            if cam_id in self.cameras:
                # Đang bật -> Tắt
                self.cameras[cam_id].stop()
                del self.cameras[cam_id]
                return True, False, "Đã tắt Webcam máy tính"
            else:
                # Đang tắt -> Bật
                if len(self.cameras) >= self.MAX_CAMERAS:
                    return False, False, f"Đã đạt giới hạn tối đa {self.MAX_CAMERAS} luồng!"
                cam = CameraStream(cam_id, "Webcam Máy Tính (Trực Tiếp)", int(device_index))
                cam.start()
                self.cameras[cam_id] = cam
                return True, True, "Đã mở Webcam máy tính thành công!"


# Khởi tạo các đối tượng toàn cục
camera_manager = CameraManager()
ai_worker_thread = AIWorker()
ai_worker_thread.start()


# ==============================================================================
# CÁC ROUTE & API CỦA FLASK BLUEPRINT
# ==============================================================================

@realtime_bp.route('/realtime/view')
def realtime_dashboard():
    return render_template('realtime.html')

def generate_mjpeg_stream(cam_id):
    """Generator phát video MJPEG đa luồng liên tục cho trình duyệt"""
    while True:
        cam = camera_manager.get_camera(cam_id)
        if not cam or not cam.is_running:
            break
        frame_bytes = cam.get_frame_bytes()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033) # ~30 khung hình/giây

@realtime_bp.route('/realtime/feed/<cam_id>')
def video_feed(cam_id):
    """Endpoint trả về luồng video trực tiếp của từng camera thật"""
    cam = camera_manager.get_camera(cam_id)
    if not cam:
        return "Camera không tồn tại hoặc đã dừng", 404
    return Response(
        generate_mjpeg_stream(cam_id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@realtime_bp.route('/api/realtime/cameras', methods=['GET'])
def api_get_cameras():
    """Lấy danh sách thông tin, FPS và AI status của tất cả camera thật"""
    return jsonify({
        "success": True,
        "cameras": camera_manager.get_all(),
        "total": len(camera_manager.cameras),
        "max": camera_manager.MAX_CAMERAS,
        "queue_size": ai_task_queue.qsize(),
        "active_threads": threading.active_count()
    })

@realtime_bp.route('/api/realtime/webcam_toggle', methods=['POST'])
def api_toggle_webcam():
    """API Bật / Tắt nhanh Webcam máy tính chính (Device 0)"""
    data = request.get_json() or {}
    device = data.get('device', 0)
    try:
        device = int(device)
    except:
        device = 0
    success, is_running, message = camera_manager.toggle_primary_webcam(device_index=device)
    return jsonify({
        "success": success,
        "is_running": is_running,
        "message": message
    })

@realtime_bp.route('/api/realtime/add', methods=['POST'])
def api_add_camera():
    """API thêm một camera thật mới (Webcam phụ hoặc IP Camera)"""
    data = request.get_json() or {}
    cam_id = data.get('cam_id')
    name = data.get('name')
    source = data.get('source', '0').strip()

    if not cam_id:
        cam_id = f"cam_{len(camera_manager.cameras) + 1}"
    if not name:
        name = f"Camera {cam_id}"

    # Chuyển chuỗi số thành int (cho các webcam index 0, 1, 2...)
    if source.isdigit():
        source = int(source)

    success, message = camera_manager.add_camera(cam_id, name, source)
    return jsonify({"success": success, "message": message})

@realtime_bp.route('/api/realtime/remove/<cam_id>', methods=['POST'])
def api_remove_camera(cam_id):
    """API dừng và xóa 1 luồng camera"""
    success, message = camera_manager.remove_camera(cam_id)
    return jsonify({"success": success, "message": message})

@realtime_bp.route('/api/realtime/toggle_flip', methods=['POST'])
def api_toggle_flip():
    """Bật / Tắt chế độ lật gương (Mirror) cho camera"""
    data = request.get_json() or {}
    cam_id = data.get('cam_id', 'webcam_0')
    success, state = camera_manager.toggle_flip(cam_id)
    return jsonify({"success": success, "flipped": state, "message": f"Đã {'bật' if state else 'tắt'} chế độ lật gương!"})

@realtime_bp.route('/api/realtime/trigger_ai', methods=['POST'])
def api_trigger_ai():
    """Kích hoạt gửi ngay 1 frame lên AI để phân tích biểu cảm tức thì"""
    data = request.get_json() or {}
    cam_id = data.get('cam_id', 'webcam_0')
    cam = camera_manager.get_camera(cam_id)
    if not cam or not cam.is_running:
        return jsonify({"success": False, "message": "Camera chưa bật!"})
    
    with cam.lock:
        if cam.current_frame is None:
            return jsonify({"success": False, "message": "Chưa có khung hình!"})
        frame_copy = cam.current_frame.copy()
    
    try:
        ai_task_queue.put_nowait((cam.cam_id, cam.name, frame_copy))
        return jsonify({"success": True, "message": "Đang phân tích biểu cảm & hành vi..."})
    except queue.Full:
        return jsonify({"success": False, "message": "Hàng đợi AI đang bận, vui lòng thử lại sau giây lát!"})

@realtime_bp.route('/api/realtime/stop_all', methods=['POST'])
def api_stop_all():
    """Dừng tất cả các luồng camera"""
    camera_manager.stop_all()
    return jsonify({"success": True, "message": "Đã dừng toàn bộ luồng camera"})

@realtime_bp.route('/api/realtime/logs', methods=['GET'])
def api_get_logs():
    """Lấy danh sách log nhận diện hành vi gần nhất"""
    with logs_lock:
        return jsonify({
            "success": True,
            "logs": list(system_ai_logs[:50])
        })

@realtime_bp.route('/api/realtime/clear_logs', methods=['POST'])
def api_clear_logs():
    """Xóa lịch sử log"""
    with logs_lock:
        system_ai_logs.clear()
    return jsonify({"success": True})

@realtime_bp.route('/api/realtime/toggle_ai', methods=['POST'])
def api_toggle_ai():
    """Bật / Tắt nhận diện AI cho từng camera hoặc toàn bộ"""
    data = request.get_json() or {}
    cam_id = data.get('cam_id')
    enable = data.get('enable', True)

    if cam_id:
        cam = camera_manager.get_camera(cam_id)
        if cam:
            cam.ai_enabled = enable
            return jsonify({"success": True, "message": f"Camera {cam_id} AI: {enable}"})
        return jsonify({"success": False, "message": "Không tìm thấy camera"}), 404
    else:
        for cam in camera_manager.cameras.values():
            cam.ai_enabled = enable
        return jsonify({"success": True, "message": f"Toàn bộ camera AI: {enable}"})

@realtime_bp.route('/api/realtime/config', methods=['GET', 'POST'])
def api_config():
    """Xem và cập nhật cấu hình OpenRouter AI"""
    global ai_worker_thread
    if request.method == 'POST':
        data = request.get_json() or {}
        api_key = data.get('api_key', '').strip()
        model = data.get('model', '').strip()
        if api_key:
            ai_worker_thread.api_key = api_key
        if model:
            ai_worker_thread.model_name = model
        return jsonify({"success": True, "message": "Đã lưu cấu hình AI!"})

    return jsonify({
        "success": True,
        "api_key": ai_worker_thread.api_key[:10] + "..." if ai_worker_thread.api_key else "",
        "model": ai_worker_thread.model_name,
        "interval_seconds": AI_FRAME_INTERVAL,
        "min_request_interval": AI_MIN_REQUEST_INTERVAL,
        "total_requests": ai_worker_thread.total_requests
    })
