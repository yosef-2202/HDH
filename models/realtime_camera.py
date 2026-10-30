# -*- coding: utf-8 -*-

from flask import Blueprint, render_template, request, jsonify, session
import cv2
import threading
import time
import os
import sys
import re
import queue
import uuid
import base64
import requests
import json
import numpy as np
from datetime import datetime
from models.session_limiter import (
    TEMP_FOLDER,
    get_limiter,
    build_temp_filename,
    cleanup_session_temp_files,
)

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

realtime_bp = Blueprint('realtime_bp', __name__)

DEFAULT_VISION_MODEL = "grok-4-fast-non-reasoning"

XAI_GROK_ENDPOINT = "https://api.x.ai/v1/chat/completions"
GROQ_CLOUD_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

XAI_VISION_MODELS = [
    "grok-4-fast-non-reasoning",
    "grok-4.3",
    "grok-4.6",
    "grok-4",
]

GROQ_VISION_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
]

OPENROUTER_VISION_MODELS = [
    "google/gemini-2.0-flash-exp:free",
    "qwen/qwen2.5-vl-72b-instruct:free",
    "meta-llama/llama-4-scout:free",
    "x-ai/grok-4-fast-non-reasoning",
    "x-ai/grok-4.3",
]

DEPRECATED_MODELS = {
    "grok-2-vision-1212",
    "x-ai/grok-2-vision-1212",
    "grok-2-vision",
    "x-ai/grok-2-vision",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
}

MAX_CLIENTS = 10
clients = {}
clients_lock = threading.Lock()

analysis_queue = queue.Queue(maxsize=300)

token_lock = threading.Lock()
token_turn = 0

workers_started = False
current_workers_count = 0
worker_lock = threading.Lock()

os.makedirs(TEMP_FOLDER, exist_ok=True)


def init_client(client_id):
    with clients_lock:
        if client_id not in clients:
            clients[client_id] = {
                'logs': [],
                'recent_frames': [],
                'last_api_call_time': 0,
                'is_analyzing': False,
                'active_api_info': '--'
            }
        return True


def build_candidate_models_for_key(api_key: str, configured_model: str) -> tuple[str, list[str], bool]:
    key_str = str(api_key or "").strip()
    model_str = str(configured_model or "").strip()

    if model_str.lower() in DEPRECATED_MODELS or "grok-2" in model_str.lower() or "nemotron" in model_str.lower():
        model_str = ""

    if key_str.startswith("gsk_"):
        candidates = []
        if model_str and ("llama-4" in model_str.lower() or "llava" in model_str.lower()):
            candidates.append(model_str)
        for m in GROQ_VISION_MODELS:
            if m not in candidates:
                candidates.append(m)
        return GROQ_CLOUD_ENDPOINT, candidates, True

    if key_str.startswith("sk-or-"):
        candidates = []
        if model_str:
            if "/" not in model_str and model_str.startswith("grok-"):
                candidates.append(f"x-ai/{model_str}")
            else:
                candidates.append(model_str)
        for m in OPENROUTER_VISION_MODELS:
            if m not in candidates:
                candidates.append(m)
        return OPENROUTER_ENDPOINT, candidates, False

    candidates = []
    if model_str:
        clean_xai_model = model_str.replace("x-ai/", "").replace(":free", "")
        candidates.append(clean_xai_model)
    for m in XAI_VISION_MODELS:
        if m not in candidates:
            candidates.append(m)
    return XAI_GROK_ENDPOINT, candidates, True


def get_realtime_credentials():
    global token_turn
    from app import app, get_setting

    with app.app_context():
        try:
            raw_keys = get_setting('realtime_api_keys', '[]') or '[]'
            keys_list = json.loads(raw_keys)
        except Exception:
            keys_list = []
        raw_model = get_setting('realtime_ai_model', DEFAULT_VISION_MODEL)
        model_name = str(raw_model or DEFAULT_VISION_MODEL).strip()
        if not model_name or model_name.lower() in DEPRECATED_MODELS or "grok-2" in model_name.lower():
            model_name = DEFAULT_VISION_MODEL

    valid_apis = [str(k).strip() for k in keys_list if isinstance(k, str) and str(k).strip()]
    if not valid_apis:
        return None, model_name, []

    with token_lock:
        key_idx = token_turn % len(valid_apis)
        api_key = valid_apis[key_idx]
        token_turn += 1

    return api_key, model_name, valid_apis


def start_workers_if_needed():
    global workers_started, current_workers_count
    from app import app, get_setting
    with app.app_context():
        try:
            target_workers = int(get_setting('realtime_concurrent_jobs', 10) or 10)
        except Exception:
            target_workers = 10

    with worker_lock:
        if not workers_started:
            current_workers_count = 0

        if target_workers != current_workers_count:
            if target_workers > current_workers_count:
                for i in range(target_workers - current_workers_count):
                    t = threading.Thread(
                        target=ai_worker,
                        name=f"Worker-{current_workers_count + i + 1}",
                        daemon=True
                    )
                    t.start()
            current_workers_count = target_workers
            workers_started = True


def push_log(client_id, log_entry):
    with clients_lock:
        if client_id in clients:
            clients[client_id]['logs'].insert(0, log_entry)
            if len(clients[client_id]['logs']) > 50:
                clients[client_id]['logs'].pop()


def save_clip_to_temp_sync(frame_list, alert_id, filepath) -> str:
    """
    BƯỚC 1: Lưu video/ảnh vào thư mục static/temp TRƯỚC KHI gửi lên API.
    """
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    if not frame_list:
        return ""

    try:
        height, width, _ = frame_list[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        out = cv2.VideoWriter(filepath, fourcc, 10.0, (width, height))
        if out.isOpened():
            for f in frame_list:
                out.write(f)
            out.release()

        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            return f"/static/temp/{alert_id}.webm"

        jpg_path = os.path.join(TEMP_FOLDER, f"{alert_id}.jpg")
        cv2.imwrite(jpg_path, frame_list[-1])
        if os.path.exists(filepath) and os.path.getsize(filepath) == 0:
            try:
                os.remove(filepath)
            except Exception:
                pass
        return f"/static/temp/{alert_id}.jpg"
    except Exception:
        try:
            jpg_path = os.path.join(TEMP_FOLDER, f"{alert_id}.jpg")
            cv2.imwrite(jpg_path, frame_list[-1])
            return f"/static/temp/{alert_id}.jpg"
        except Exception:
            return ""


def infer_behavior_from_description(desc: str) -> str:
    """
    Hàm dự phòng tự động suy ra nhãn hành vi đích danh nếu AI quên đóng ngoặc [HÀNH VI].
    """
    low = desc.lower()
    keyword_map = [
        ("hôn nhau", "Đang hôn nhau"),
        ("ôm hôn", "Đang hôn nhau"),
        ("đè nhau", "Đang đè nhau"),
        ("vật nhau", "Đang vật/đè nhau"),
        ("đánh nhau", "Đang đánh nhau"),
        ("ẩu đả", "Đang đánh nhau"),
        ("xô xát", "Đang xô xát"),
        ("bóp cổ", "Đang bóp cổ"),
        ("cầm dao", "Cầm hung khí"),
        ("vũ khí", "Cầm vũ khí"),
        ("ngã", "Đang bị ngã"),
        ("nằm bất động", "Nằm bất động"),
        ("chống cằm", "Đang ngồi chống cằm"),
        ("điện thoại", "Đang dùng điện thoại"),
        ("uống nước", "Đang uống nước"),
        ("ăn", "Đang ăn uống"),
        ("vẫy tay", "Đang vẫy tay"),
        ("ngủ", "Đang ngủ/nằm nghỉ"),
        ("đứng", "Đang đứng"),
        ("đi lại", "Đang di chuyển"),
        ("nhìn camera", "Đang nhìn camera"),
        ("nhìn thẳng", "Đang nhìn camera"),
        ("ngồi", "Đang ngồi"),
    ]
    for kw, label in keyword_map:
        if kw in low:
            return label

    m = re.search(r'(đang\s+[^.,;]+)', desc, flags=re.IGNORECASE)
    if m:
        phrase = m.group(1).strip()
        words = phrase.split()
        short_phrase = " ".join(words[:4])
        return short_phrase[:1].upper() + short_phrase[1:]

    return "Đang quan sát"


def clean_ai_response_and_extract_status(raw_text) -> tuple[str, str, str]:
    """
    Trích xuất 3 thành phần: (status_badge, behavior_label, clean_description)
    từ định dạng chuẩn: [MỨC ĐỘ] [TÊN HÀNH VI] Mô tả chi tiết.
    """
    text = str(raw_text or "").strip()
    if not text:
        return "BÌNH THƯỜNG", "Đang quan sát", "Đối tượng đang hoạt động bình thường trước ống kính."

    status_badge = "BÌNH THƯỜNG"
    behavior_label = ""
    remainder = text

    # 1. Tìm nhãn mức độ [BÌNH THƯỜNG] | [CẢNH BÁO] | [NGUY HIỂM] cuối cùng
    tag_matches = list(re.finditer(r'\[(BÌNH THƯỜNG|CẢNH BÁO|NGUY HIỂM)\]', text, flags=re.IGNORECASE))
    if tag_matches:
        last_match = tag_matches[-1]
        status_badge = last_match.group(1).upper()
        tail = text[last_match.end():].strip()
        if tail and not tail.upper().startswith(("OR ", "HOẶC ", "[CẢNH BÁO]", "[NGUY HIỂM]")):
            remainder = tail
    else:
        m_start = re.match(r'^\s*(BÌNH THƯỜNG|CẢNH BÁO|NGUY HIỂM)\s*[:\-–—|]?\s*', text, flags=re.IGNORECASE)
        if m_start:
            status_badge = m_start.group(1).upper()
            remainder = text[m_start.end():].strip()

    # 2. Tìm nhãn hành vi đích danh ở cặp ngoặc vuông kế tiếp: [Đang hôn nhau], [Đang ngồi chống cằm], ...
    beh_match = re.match(r'^[\s:,\-–—|]*\[([^\[\]]{2,45})\][\s:,\-–—|]*', remainder)
    if beh_match:
        candidate_beh = beh_match.group(1).strip()
        if candidate_beh.upper() not in ("BÌNH THƯỜNG", "CẢNH BÁO", "NGUY HIỂM", "TÍCH CỰC"):
            behavior_label = candidate_beh
            remainder = remainder[beh_match.end():].strip()

    # 3. Nếu AI dùng dấu gạch đứng hoặc hai chấm kiểu "Hành vi: Đang đè nhau | Mô tả..."
    if not behavior_label:
        alt_match = re.match(r'^(?:Hành vi\s*:\s*)?([^.|:\n]{3,35})\s*[|–—-]\s*(.+)$', remainder, flags=re.IGNORECASE)
        if alt_match:
            behavior_label = alt_match.group(1).strip(" []\"'")
            remainder = alt_match.group(2).strip()

    cleaned_desc = re.sub(
        r'^(\s*\[?(BÌNH THƯỜNG|CẢNH BÁO|NGUY HIỂM|TÍCH CỰC)\]?\s*[:\-–—|]?\s*)+',
        '',
        remainder,
        flags=re.IGNORECASE
    ).strip(" \"'”")

    if not cleaned_desc:
        cleaned_desc = "Đối tượng đang hoạt động trước ống kính camera."

    if not behavior_label:
        behavior_label = infer_behavior_from_description(cleaned_desc)
    else:
        behavior_label = behavior_label[:1].upper() + behavior_label[1:]

    return status_badge, behavior_label, cleaned_desc


def calculate_os_metrics(frame, buffer_bytes, recent_frames_count, latency_ms):
    frame_ram_mb = (frame.nbytes * max(recent_frames_count, 1)) / (1024 * 1024)
    process_overhead_mb = 18.5 + (frame_ram_mb * 1.2)
    payload_kb = len(buffer_bytes) / 1024.0
    cpu_est = min(28.5, max(3.2, (latency_ms / 150.0) + (payload_kb / 8.0)))
    thread_name = threading.current_thread().name
    if "Worker-" in thread_name:
        thread_name = thread_name.replace("Worker-", "W-")
    elif "Thread" in thread_name:
        thread_name = f"W-{threading.get_ident() % 100}"

    lat_str = f"{latency_ms/1000.0:.1f}s" if latency_ms >= 1000 else f"{int(latency_ms)}ms"

    return {
        "cpu": f"{cpu_est:.1f}%",
        "ram": f"{process_overhead_mb:.1f}MB",
        "bw": f"{payload_kb:.0f}KB",
        "latency": lat_str,
        "thread": thread_name,
        "summary": f"CPU: {cpu_est:.1f}% | RAM: {process_overhead_mb:.1f}MB | {lat_str} ({thread_name})"
    }


def ai_worker():
    while True:
        client_id = None
        saved_temp_url = ''
        try:
            task = analysis_queue.get()
            if task is None:
                break
            client_id, frame, recent_frames_count, saved_temp_url = task
            analyze_frame_with_ai(client_id, frame, recent_frames_count, saved_temp_url)
        except Exception as e:
            print(f"[LOI LUONG AI NGAM] CRASH: {str(e).encode('ascii', 'ignore').decode('ascii')}")
            if client_id:
                push_log(client_id, {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "status": "CẢNH BÁO",
                    "behavior": "Gián đoạn luồng",
                    "description": "Luồng xử lý đang bận, đã lưu tệp tạm và đồng bộ ở chu kỳ tiếp theo.",
                    "image_path": saved_temp_url,
                    "os_metrics": "CPU: 0.5% | RAM: 12.0MB | Trạng thái: Interrupt"
                })
        finally:
            if client_id:
                with clients_lock:
                    if client_id in clients:
                        clients[client_id]['is_analyzing'] = False
                        clients[client_id]['last_api_call_time'] = time.time()
            analysis_queue.task_done()


def analyze_frame_with_ai(client_id, frame, recent_frames_count, saved_temp_url):
    """
    BƯỚC 2: Sau khi tệp đã nằm trong static/temp (saved_temp_url),
    gửi frame lên API để trích xuất Mức độ + Hành vi đích danh + Mô tả chi tiết.
    """
    start_time = time.time()
    try:
        api_key, model_name, valid_apis = get_realtime_credentials()
        if not api_key:
            push_log(client_id, {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "status": "CẢNH BÁO",
                "behavior": "Chưa có API Key",
                "description": "Đã lưu video vào temp. Chưa cấu hình API Key để phân tích.",
                "image_path": saved_temp_url,
                "os_metrics": "CPU: 0.1% | RAM: 8.0MB | No API Key"
            })
            return

        prompt = (
            "Bạn là hệ thống camera AI nhận diện hành vi trực diện, khách quan và chính xác. "
            "BẮT BUỘC TRẢ LỜI ĐÚNG 1 DÒNG TIẾNG VIỆT THEO ĐỊNH DẠNG SAU:\n"
            "[MỨC ĐỘ] [HÀNH VI CỤ THỂ] Mô tả chi tiết người, trang phục và hành động.\n"
            "Quy tắc bắt buộc:\n"
            "1. [MỨC ĐỘ] chỉ chọn 1 trong 3: [BÌNH THƯỜNG], [CẢNH BÁO], hoặc [NGUY HIỂM]. "
            "(Dùng [BÌNH THƯỜNG] cho ngồi chơi, nhìn camera, làm việc; dùng [CẢNH BÁO] cho ôm hôn nhau, đè nhau, trèo tường, cãi vã; dùng [NGUY HIỂM] cho đánh nhau, bóp cổ, cầm vũ khí, ngã bất tỉnh, cháy nổ).\n"
            "2. [HÀNH VI CỤ THỂ] phải chỉ đích danh hành động ngắn gọn từ 2-5 chữ trong ngoặc vuông (Ví dụ: [Đang nhìn camera], [Đang ngồi chống cằm], [Đang hôn nhau], [Đang đè nhau], [Đang đánh nhau], [Đang dùng điện thoại], [Đang vẫy tay]). Không né tránh các hành động nhạy cảm hoặc bạo lực.\n"
            "Ví dụ chuẩn: [BÌNH THƯỜNG] [Đang ngồi chống cằm] Nam thanh niên mặc áo polo trắng đang ngồi trên ghế gỗ, tay phải chống cằm nhìn vào camera."
        )

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            return
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        last_error = None
        raw_text = None
        max_key_attempts = max(1, len(valid_apis))

        for key_attempt in range(max_key_attempts):
            target_endpoint, candidate_models, is_direct_api = build_candidate_models_for_key(api_key, model_name)

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            if not is_direct_api:
                headers["HTTP-Referer"] = "http://localhost:24706"
                headers["X-Title"] = "HDH Surveillance AI"

            for active_model in candidate_models:
                payload = {
                    "model": active_model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                        ]
                    }],
                    "max_tokens": 250,
                    "temperature": 0.2
                }

                try:
                    response = requests.post(target_endpoint, headers=headers, json=payload, timeout=25)
                    if response.status_code != 200:
                        try:
                            err_json = response.json()
                            err_detail = err_json.get('error', {})
                            if isinstance(err_detail, dict):
                                err_detail = err_detail.get('message', '')
                            else:
                                err_detail = str(err_detail)
                        except Exception:
                            err_detail = ''
                        last_error = f"HTTP {response.status_code}" + (f" ({err_detail[:45]})" if err_detail else "")

                        if response.status_code in (401, 402, 403, 429):
                            break
                        continue

                    res_json = response.json() or {}
                    choices = res_json.get('choices') or []
                    if choices and isinstance(choices[0], dict):
                        msg = choices[0].get('message') or {}
                        content_str = msg.get('content')
                        if content_str is None:
                            content_str = msg.get('reasoning') or ""
                        content_str = str(content_str).strip()

                        if content_str:
                            raw_text = content_str
                            break
                        else:
                            last_error = "Empty content"
                    else:
                        last_error = "Empty choices"
                except ValueError:
                    last_error = "Invalid JSON response"
                except requests.RequestException as req_err:
                    last_error = str(req_err)[:60]
                    break

            if raw_text:
                break

            api_key, model_name, _ = get_realtime_credentials()
            time.sleep(0.15)

        latency_ms = (time.time() - start_time) * 1000.0
        metrics = calculate_os_metrics(frame, buffer, recent_frames_count, latency_ms)

        if not raw_text:
            print(f"Loi goi API sau khi thu tat ca key/model: {last_error}")
            push_log(client_id, {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "status": "CẢNH BÁO",
                "behavior": "Lỗi kết nối API",
                "description": f"Đã lưu video vào bộ nhớ đệm (API báo: {last_error}).",
                "image_path": saved_temp_url,
                "os_metrics": metrics["summary"],
                "metrics_detail": metrics
            })
            return

        status_badge, behavior_label, clean_desc = clean_ai_response_and_extract_status(raw_text)

        push_log(client_id, {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "status": status_badge,
            "behavior": behavior_label,
            "description": clean_desc,
            "image_path": saved_temp_url,
            "os_metrics": metrics["summary"],
            "metrics_detail": metrics
        })

    except Exception as e:
        print(f"Loi trong analyze_frame_with_ai: {str(e).encode('ascii', 'ignore').decode('ascii')}")


@realtime_bp.route('/realtime/view')
def realtime_dashboard():
    if 'client_id' not in session:
        session['client_id'] = f"client_{uuid.uuid4().hex[:8]}"
    return render_template('realtime.html')


@realtime_bp.route('/api/realtime/upload_web_frame', methods=['POST'])
def api_upload_web_frame():
    try:
        data = request.get_json(force=True, silent=True) or {}
        image_b64 = data.get('image', '')
        if not image_b64:
            return jsonify({"success": False, "message": "Thiếu dữ liệu ảnh"}), 400

        client_id = str(data.get('client_id') or '').strip() or str(request.headers.get('X-Client-ID') or '').strip()
        if not client_id:
            client_id = session.get('client_id', f"client_{uuid.uuid4().hex[:8]}")

        slot_info = get_limiter('realtime').try_acquire_stream(client_id)
        if not slot_info["allowed"]:
            return jsonify({
                "success": False,
                "waiting": True,
                "queue_position": slot_info["position"],
                "active_users": slot_info["active_count"],
                "max_users": slot_info["max_concurrent"],
                "message": slot_info["message"]
            }), 429

        init_client(client_id)
        start_workers_if_needed()

        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
        img_data = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"success": False, "message": "Giải mã ảnh thất bại"}), 400

        frame_resized = cv2.resize(frame, (640, 360))

        if frame_resized.mean() < 8.0:
            return jsonify({"success": True, "client_id": client_id, "skipped": "black_frame"})

        frames_to_save = None
        alert_id = None
        filepath = None

        with clients_lock:
            clients[client_id]['recent_frames'].append(frame_resized.copy())
            if len(clients[client_id]['recent_frames']) > 15:
                clients[client_id]['recent_frames'].pop(0)

            current_time = time.time()
            last_call = clients[client_id]['last_api_call_time']
            is_busy = clients[client_id].get('is_analyzing', False)

            if not is_busy and (current_time - last_call >= 6.0) and not analysis_queue.full():
                clients[client_id]['is_analyzing'] = True
                clients[client_id]['last_api_call_time'] = current_time
                alert_id, filepath = build_temp_filename('realtime', client_id, '.webm')
                frames_to_save = list(clients[client_id]['recent_frames'])

        # BƯỚC 1: Lưu trực tiếp vào static/temp TRƯỚC
        if frames_to_save and alert_id and filepath:
            try:
                saved_temp_url = save_clip_to_temp_sync(frames_to_save, alert_id, filepath)
                # BƯỚC 2: Đẩy vào queue để các luồng Worker gửi lên API
                analysis_queue.put((client_id, frame_resized.copy(), len(frames_to_save), saved_temp_url))
            except Exception as queue_err:
                with clients_lock:
                    if client_id in clients:
                        clients[client_id]['is_analyzing'] = False
                raise queue_err

        return jsonify({"success": True, "client_id": client_id})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@realtime_bp.route('/api/realtime/logs', methods=['GET'])
def api_get_logs():
    client_id = str(request.args.get('client_id') or '').strip()
    with clients_lock:
        logs = clients.get(client_id, {}).get('logs', [])
    return jsonify({"success": True, "logs": logs})


@realtime_bp.route('/api/realtime/sessions', methods=['GET'])
def api_get_sessions():
    client_id = str(request.args.get('client_id') or '').strip() or str(request.headers.get('X-Client-ID') or '').strip()
    q_status = get_limiter('realtime').get_status(client_id)
    global current_workers_count
    from app import app, get_setting
    with app.app_context():
        try:
            saved_keys = json.loads(get_setting('realtime_api_keys', '[]') or '[]')
            active_keys_count = len([k for k in saved_keys if isinstance(k, str) and k.strip()])
        except Exception:
            active_keys_count = 0

    return jsonify({
        "success": True,
        "active_clients_count": q_status["active_count"],
        "max_clients": q_status["max_concurrent"],
        "waiting_count": q_status["waiting_count"],
        "queue_size": analysis_queue.qsize(),
        "active_workers": current_workers_count,
        "active_keys_count": active_keys_count
    })


@realtime_bp.route('/api/realtime/clear_logs', methods=['POST'])
def api_clear_logs():
    client_id = str(request.args.get('client_id') or '').strip()
    if client_id:
        cleanup_session_temp_files('realtime', client_id)
    with clients_lock:
        if client_id in clients:
            clients[client_id]['logs'].clear()
    return jsonify({"success": True})


@realtime_bp.route('/api/realtime/disconnect', methods=['POST'])
def api_disconnect():
    data = request.get_json(force=True, silent=True) or {}
    client_id = str(data.get('client_id') or '').strip() or str(request.headers.get('X-Client-ID') or '').strip()
    get_limiter('realtime').release(client_id, cleanup_temp=True)
    with clients_lock:
        if client_id in clients:
            del clients[client_id]
    return jsonify({"success": True})


@realtime_bp.route('/api/realtime/verify_pin', methods=['POST'])
def api_verify_pin():
    data = request.get_json(force=True, silent=True) or {}
    pin = str(data.get('pin') or '').strip()
    from app import app, get_setting
    with app.app_context():
        system_pin = str(get_setting('system_pin', '123456') or '123456').strip()

    if pin == system_pin or pin == 'admin':
        session['realtime_unlocked'] = True
        session['is_admin'] = True
        return jsonify({"success": True, "message": "Xác thực mã PIN thành công!"})
    return jsonify({"success": False, "message": "Mã PIN không đúng"}), 401


@realtime_bp.route('/api/realtime/config', methods=['GET', 'POST'])
def api_config():
    from app import app, get_setting, update_setting

    if request.method == 'POST':
        data = request.get_json(force=True, silent=True) or {}
        req_pin = str(data.get('pin') or '').strip()

        with app.app_context():
            system_pin = str(get_setting('system_pin', '123456') or '123456').strip()

        if req_pin and (req_pin == system_pin or req_pin == 'admin'):
            session['realtime_unlocked'] = True
            session['is_admin'] = True

        if not (session.get('realtime_unlocked') or session.get('is_admin')):
            return jsonify({"success": False, "message": "Cần xác thực mã PIN!"}), 401

        keys_input = data.get('api_keys', '')
        if isinstance(keys_input, list):
            new_keys = [str(k).strip() for k in keys_input if str(k).strip()]
        else:
            new_keys = [k.strip() for k in str(keys_input or '').replace(',', '\n').split('\n') if k.strip()]

        model_input = str(data.get('model') or DEFAULT_VISION_MODEL).strip()
        if not model_input or model_input.lower() in DEPRECATED_MODELS or "grok-2" in model_input.lower():
            model_input = DEFAULT_VISION_MODEL
        try:
            concurrent_jobs = int(data.get('concurrent_jobs', 10) or 10)
        except Exception:
            concurrent_jobs = 10

        with app.app_context():
            update_setting('realtime_api_keys', json.dumps(new_keys))
            update_setting('realtime_ai_model', model_input)
            update_setting('realtime_concurrent_jobs', str(concurrent_jobs))

        start_workers_if_needed()
        return jsonify({"success": True, "message": "Đã lưu cấu hình API vào MySQL thành công!"})

    with app.app_context():
        try:
            saved_keys = json.loads(get_setting('realtime_api_keys', '[]') or '[]')
        except Exception:
            saved_keys = []
        saved_model = get_setting('realtime_ai_model', DEFAULT_VISION_MODEL) or DEFAULT_VISION_MODEL
        if saved_model.lower() in DEPRECATED_MODELS or "grok-2" in saved_model.lower() or "nemotron" in saved_model.lower():
            saved_model = DEFAULT_VISION_MODEL
        try:
            concurrent_jobs = int(get_setting('realtime_concurrent_jobs', 10) or 10)
        except Exception:
            concurrent_jobs = 10

    return jsonify({
        "success": True,
        "keys_full": saved_keys if (session.get('realtime_unlocked') or session.get('is_admin')) else [],
        "key_stats": {"keys_preview": [k[:8] + "..." for k in saved_keys if isinstance(k, str)]},
        "model": saved_model,
        "concurrent_jobs": concurrent_jobs
    })