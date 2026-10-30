from flask import Blueprint, render_template, Response, request, jsonify
import cv2
import threading
import time
import sys
import os
import queue
import uuid
import json
import base64
import requests
import psutil
from datetime import datetime
from models.session_limiter import (
    get_limiter,
    build_temp_filename,
    cleanup_session_temp_files,
    is_auto_clean_enabled,
    sanitize_id,
)

rtsp_bp = Blueprint('rtsp_bp', __name__)
current_process = psutil.Process(os.getpid())

# Quản lý 10 người dùng độc lập
MAX_CLIENTS = 10
clients = {} 
clients_lock = threading.Lock()

# Danh sách cảnh báo toàn cục đồng bộ với MySQL để Chatbot học dữ liệu
global_alerts = []

# Hàng đợi dùng chung cho tất cả Client
analysis_queue = queue.Queue(maxsize=300)

# Khóa xoay vòng Token
token_lock = threading.Lock()
token_turn = 0

workers_started = False
current_workers_count = 0
worker_lock = threading.Lock()

os.makedirs('static/temp', exist_ok=True)

def load_recent_alerts_from_db(client_id=None, limit=20):
    try:
        from app import app, RtspAlert
        with app.app_context():
            query = RtspAlert.query
            if client_id is not None:
                cid = sanitize_id(client_id)
                query = query.filter_by(client_id=cid)
            records = query.order_by(RtspAlert.id.desc()).limit(limit).all()
            result = []
            for r in records:
                result.append({
                    "id": r.alert_id,
                    "time": r.time_str or (r.created_at.strftime("%H:%M:%S") if r.created_at else "--:--:--"),
                    "behavior": r.behavior or "Phát hiện đối tượng",
                    "process_time": r.process_time or "--",
                    "os_metrics": r.os_metrics or "",
                    "metrics_detail": None,
                    "level": r.level or "Theo dõi",
                    "badge": r.badge or "info",
                    "desc": r.description or "",
                    "temp_video_url": r.video_url or f"/static/temp/{r.alert_id}.webm"
                })
            return result
    except Exception:
        return []

def save_alert_to_mysql(client_id, rtsp_url, alert_obj, prediction_text):
    try:
        from app import app, db, RtspAlert
        cid = sanitize_id(client_id)
        with app.app_context():
            new_record = RtspAlert(
                alert_id=alert_obj["id"],
                client_id=cid,
                rtsp_url=rtsp_url,
                time_str=alert_obj["time"],
                behavior=alert_obj["behavior"],
                level=alert_obj["level"],
                badge=alert_obj["badge"],
                description=alert_obj["desc"],
                prediction=prediction_text,
                process_time=alert_obj["process_time"],
                os_metrics=alert_obj["os_metrics"],
                video_url=alert_obj["temp_video_url"],
                created_at=datetime.now()
            )
            db.session.add(new_record)
            db.session.commit()
    except Exception as e:
        print(f"[RTSP MySQL Error] Không thể lưu vào DB: {e}")

def init_client(client_id, rtsp_url=''):
    cid = sanitize_id(client_id)
    client_alerts = load_recent_alerts_from_db(client_id=cid, limit=20)
    with clients_lock:
        global global_alerts
        if not global_alerts:
            global_alerts = load_recent_alerts_from_db(client_id=None, limit=50)
        if cid not in clients:
            clients[cid] = {
                'logs': [],
                'alerts': client_alerts,
                'last_api_call_time': 0,
                'camera': None,
                'rtsp_url': rtsp_url,
                'active_api_info': '--',
                'latest_metrics': None
            }
        else:
            if rtsp_url:
                clients[cid]['rtsp_url'] = rtsp_url
        return True

def get_api_credentials():
    global token_turn
    from app import app, get_setting
    with app.app_context():
        try: api_list = json.loads(get_setting('rtsp_api_list', '[]'))
        except: api_list = []
        
    valid_apis = [api for api in api_list if api.get('key', '').strip() != '']
    if not valid_apis: return None, None, None

    with token_lock:
        selected_api = valid_apis[token_turn % len(valid_apis)]
        token_turn += 1
        
    api_key = selected_api['key'].strip()
    model_name = selected_api.get('model', 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free').strip()
    return api_key, model_name, valid_apis

def start_workers_if_needed():
    global workers_started, current_workers_count
    with worker_lock:
        from app import app, get_setting
        with app.app_context():
            try: target_workers = int(get_setting('rtsp_concurrent_jobs', 10))
            except: target_workers = 10

        if not workers_started:
            current_workers_count = 0

        if target_workers > current_workers_count:
            for i in range(target_workers - current_workers_count):
                worker_name = f"W-{current_workers_count + i + 1}"
                threading.Thread(target=ai_worker, name=worker_name, daemon=True).start()
            current_workers_count = target_workers
            workers_started = True

def ai_worker():
    while True:
        task = analysis_queue.get()
        if task is None: break
        client_id, frame, recent_frames, alert_id = task
        analyze_frame_with_ai(client_id, frame, recent_frames, alert_id)
        analysis_queue.task_done()

def record_short_clip(frame_list, alert_id):
    if not frame_list: return
    try:
        height, width, _ = frame_list[0].shape
        filepath = os.path.join('static', 'temp', f"{alert_id}.webm")
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        out = cv2.VideoWriter(filepath, fourcc, 15.0, (width, height))
        for f in frame_list: out.write(f)
        out.release()
    except Exception: pass

def push_log(client_id, message):
    cid = sanitize_id(client_id)
    with clients_lock:
        if cid in clients:
            if len(clients[cid]['logs']) > 50: clients[cid]['logs'].pop(0)
            clients[cid]['logs'].append(message)

def analyze_frame_with_ai(client_id, frame, recent_frames, alert_id):
    cid = sanitize_id(client_id)
    api_key, model_name, valid_apis = get_api_credentials()
    if not api_key: return

    thread_name = threading.current_thread().name
    if not thread_name.startswith("W-"):
        thread_name = f"W-{threading.get_ident() % 10 + 1}"

    api_info = f"{model_name} ({api_key[:4]}...{api_key[-4:]})" if api_key else "--"
    rtsp_url = ""
    with clients_lock:
        if cid in clients:
            clients[cid]['active_api_info'] = api_info
            rtsp_url = clients[cid].get('rtsp_url', '')

    max_retries = max(1, len(valid_apis)) 
    
    for attempt in range(max_retries):
        try:
            timestamp_start = datetime.now().strftime("%H:%M:%S")
            push_log(cid, f"[{timestamp_start}] [{thread_name}] Đang phân tích khung hình...")
            start_time = time.time()
            
            motion_detected = False
            frame_for_ai = frame.copy()
            if len(recent_frames) >= 15:
                gray_past = cv2.GaussianBlur(cv2.cvtColor(recent_frames[0], cv2.COLOR_BGR2GRAY), (21, 21), 0)
                gray_current = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
                thresh = cv2.dilate(cv2.threshold(cv2.absdiff(gray_past, gray_current), 30, 255, cv2.THRESH_BINARY)[1], None, iterations=2)
                H, W = thresh.shape
                cv2.rectangle(thresh, (0, 0), (W, int(H * 0.08)), (0, 0, 0), -1) 
                cv2.rectangle(thresh, (0, int(H * 0.92)), (W, H), (0, 0, 0), -1) 
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for c in contours:
                    if cv2.contourArea(c) > 800:
                        motion_detected = True
                        (x, y, w, h) = cv2.boundingRect(c)
                        cv2.rectangle(frame_for_ai, (x, y), (x+w, y+h), (0, 0, 255), 3)

            motion_directive = "=> RADAR: Đang di chuyển." if motion_detected else "=> RADAR: Đứng yên."
            
            prompt = f"""Bạn là AI phân tích camera an ninh. KHÔNG giao tiếp, KHÔNG giải thích, KHÔNG đóng vai.
{motion_directive}

QUY TẮC BẮT BUỘC:
- Trả lời bằng Tiếng Việt.
- NẾU KHÔNG CÓ ĐỐI TƯỢNG (Người, Xe, Động vật): Chỉ trả lời duy nhất 1 chữ "Không".
- NẾU CÓ ĐỐI TƯỢNG: Phải tuân thủ chính xác định dạng bên dưới, dùng dấu "|" để ngăn cách.

MẪU ĐỊNG DẠNG (Bắt chước y hệt cấu trúc này):
PHẦN 1
Người | Đang đi bộ | Mặc áo xanh, quần đen, đi giữa các kệ hàng
Xe máy | Đang đậu | Xe máy đỏ ở góc trái

PHẦN 2
Mức độ an toàn: Bình thường
Dự đoán sự việc: Khách hàng đang mua sắm."""

            ret, buffer = cv2.imencode('.jpg', frame_for_ai)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            payload_kb = round(len(img_b64) * 0.75 / 1024, 1)
            
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {"model": model_name, "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}]}
            
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=30)
            response.raise_for_status() 
            res_json = response.json()
                
            text = res_json['choices'][0]['message']['content'].strip()
            process_time = round(time.time() - start_time, 2)

            try:
                cpu_val = round(psutil.cpu_percent(interval=None) / max(1, psutil.cpu_count()) + 3.5, 1)
                ram_val = round(current_process.memory_info().rss / (1024 * 1024), 1)
            except Exception:
                cpu_val = 5.4
                ram_val = 42.0

            os_metrics_str = f"CPU: {cpu_val}% | RAM: {ram_val}MB | Trễ: {process_time}s ({thread_name})"
            metrics_detail = {
                "cpu": f"{cpu_val}%",
                "ram": f"{ram_val} MB",
                "bw": f"{payload_kb} KB/req",
                "worker": thread_name
            }

            with clients_lock:
                if cid in clients:
                    clients[cid]['latest_metrics'] = metrics_detail
            
            if text.lower().startswith("không") and len(text) < 15:
                push_log(cid, f"[{datetime.now().strftime('%H:%M:%S')}] [{thread_name}] Không có đối tượng ({os_metrics_str})")
                break

            part1, part2 = text.split("PHẦN 2") if "PHẦN 2" in text else (text, "")
            detected_objects, main_actions = [], []
            
            for line in part1.split('\n'):
                if "|" in line:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 2:
                        obj_type = parts[0].replace("-", "").strip()
                        main_act = parts[1].strip()
                        
                        if motion_detected and any(w in main_act.lower() for w in ["đứng yên", "đang đậu", "tĩnh"]): 
                            main_act = "Đang di chuyển"
                        elif not motion_detected and any(w in main_act.lower() for w in ["chạy", "di chuyển", "đi bộ"]): 
                            main_act = "Đứng yên"
                            
                        desc = parts[2].strip() if len(parts) >= 3 else ""
                        detected_objects.append(f"[{obj_type}: {main_act} - {desc}]")
                        main_actions.append(main_act)
            
            safety_level, badge_color, prediction = "Theo dõi", "info", "Không có dự đoán"
            for line in part2.split('\n'):
                if "Mức độ an toàn:" in line:
                    safety_level = line.split("Mức độ an toàn:")[1].strip().replace("[", "").replace("]", "")
                    if any(w in safety_level.lower() for w in ["nguy hiểm", "khẩn cấp"]): badge_color = "danger"
                    elif any(w in safety_level.lower() for w in ["cảnh báo", "vi phạm", "đáng ngờ"]): badge_color = "warning"
                    else: badge_color = "success"
                elif "Dự đoán sự việc:" in line:
                    prediction = line.split("Dự đoán sự việc:")[1].strip().replace("[", "").replace("]", "")

            if detected_objects:
                unique_actions = list(set(main_actions))
                behavior_title = ", ".join(unique_actions) if unique_actions else "Phát hiện đối tượng"
                action_desc = "<b>Đối tượng:</b><br>" + "<br>".join([f"- {p}" for p in detected_objects]) + f"<br><b>Dự đoán:</b> <i>{prediction}</i>"
                
                threading.Thread(target=record_short_clip, args=(list(recent_frames), alert_id), daemon=True).start()
                push_log(cid, f"[{datetime.now().strftime('%H:%M:%S')}] [{thread_name}] {safety_level.upper()}: {behavior_title} ({os_metrics_str})")

                alert_obj = {
                    "id": alert_id,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "behavior": behavior_title,
                    "process_time": f"{process_time}s",
                    "os_metrics": os_metrics_str,
                    "metrics_detail": metrics_detail,
                    "level": safety_level,
                    "badge": badge_color,
                    "desc": action_desc,
                    "temp_video_url": f"/static/temp/{alert_id}.webm" 
                }

                with clients_lock:
                    global_alerts.insert(0, alert_obj)
                    if len(global_alerts) > 50:
                        global_alerts.pop()
                    if cid in clients:
                        clients[cid]['alerts'].insert(0, alert_obj)
                        if len(clients[cid]['alerts']) > 20:
                            clients[cid]['alerts'].pop()

                save_alert_to_mysql(cid, rtsp_url, alert_obj, prediction)
            break 
            
        except Exception as e:
            time.sleep(1)
            api_key, model_name, valid_apis = get_api_credentials() 

def generate_frames(rtsp_url, client_id):
    cid = sanitize_id(client_id)
    start_workers_if_needed() 
    
    os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "100"
    camera = cv2.VideoCapture(rtsp_url)
    
    with clients_lock:
        if cid in clients:
            clients[cid]['camera'] = camera
            clients[cid]['rtsp_url'] = rtsp_url
    
    recent_frames = [] 
    api_cooldown = 5 

    try:
        while True:
            get_limiter('rtsp').try_acquire_stream(cid)
            if not camera.isOpened(): break
            success, frame = camera.read()
            if not success: break
            
            frame_resized = cv2.resize(frame, (640, 360))
            recent_frames.append(frame_resized.copy())
            if len(recent_frames) > 45: recent_frames.pop(0)
            
            current_time = time.time()
            with clients_lock: last_call = clients.get(cid, {}).get('last_api_call_time', 0)
            
            countdown = max(0, int(api_cooldown - (current_time - last_call)))
            if countdown == 0 and not analysis_queue.full():
                with clients_lock:
                    if cid in clients: clients[cid]['last_api_call_time'] = current_time
                # Đặt tên chuẩn: rtsp_<tên_phiên>_<số_thứ_tự>
                alert_id, _ = build_temp_filename('rtsp', cid, '.webm')
                analysis_queue.put((cid, frame_resized.copy(), list(recent_frames), alert_id))

            cv2.putText(frame_resized, f"Wait: {countdown}s", (520, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if countdown==0 else (0,165,255), 2)
            ret, buffer = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            if ret:
                time.sleep(0.066) 
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        camera.release()
        get_limiter('rtsp').release(cid, cleanup_temp=True)
        with clients_lock:
            if cid in clients and clients[cid]['camera'] == camera:
                clients[cid]['camera'] = None

@rtsp_bp.route('/rtsp')
def rtsp_page(): return render_template('rtsp.html')

@rtsp_bp.route('/video_feed')
def video_feed():
    rtsp_url = request.args.get('url', '')
    raw_cid = request.args.get('client_id', 'guest')
    cid = sanitize_id(raw_cid)
    if not rtsp_url: return "URL trống", 400
    
    slot_info = get_limiter('rtsp').try_acquire_stream(cid)
    if not slot_info["allowed"]:
        return jsonify(slot_info), 429
        
    init_client(cid, rtsp_url)
    push_log(cid, f"[{datetime.now().strftime('%H:%M:%S')}] [SYSTEM] Bắt đầu kết nối Camera...")
    return Response(generate_frames(rtsp_url, cid), mimetype='multipart/x-mixed-replace; boundary=frame')

@rtsp_bp.route('/api/rtsp/data')
def rtsp_data():
    raw_cid = request.args.get('client_id', '').strip()
    cid = sanitize_id(raw_cid) if raw_cid else ''
    q_status = get_limiter('rtsp').get_status(cid)
    
    with clients_lock:
        c_data = clients.get(cid) if cid else None
        if not c_data:
            logs = []
            alerts = load_recent_alerts_from_db(client_id=cid, limit=20) if cid else []
            active_api = '--'
            latest_metrics = None
        else:
            logs = c_data['logs'].copy()
            alerts = c_data['alerts'].copy()
            active_api = c_data.get('active_api_info', '--')
            latest_metrics = c_data.get('latest_metrics')
        
    from app import app, get_setting
    with app.app_context():
        try: concurrent_jobs = int(get_setting('rtsp_concurrent_jobs', 10))
        except: concurrent_jobs = 10
        try:
            api_list = json.loads(get_setting('rtsp_api_list', '[]'))
            valid_keys_count = len([a for a in api_list if a.get('key', '').strip() != ''])
        except:
            valid_keys_count = 0

    stats = {
        "queue_size": analysis_queue.qsize(),
        "workers": concurrent_jobs,
        "active_users": f"{q_status['active_count']}/{MAX_CLIENTS}",
        "waiting_users": q_status['waiting_count'],
        "active_keys": valid_keys_count,
        "latest_metrics": latest_metrics
    }
    return jsonify({"logs": logs, "alerts": alerts, "stats": stats, "active_api": active_api})

@rtsp_bp.route('/api/rtsp/delete_temp/<clip_id>', methods=['POST'])
def delete_temp(clip_id):
    safe_name = os.path.basename(clip_id)
    target = os.path.join('static', 'temp', f"{safe_name}.webm")
    if os.path.exists(target):
        os.remove(target)
    return jsonify({"success": True})

@rtsp_bp.route('/api/rtsp/clear_cache', methods=['POST'])
def clear_cache():
    raw_cid = (request.get_json(silent=True) or {}).get('client_id', '') or request.headers.get('X-Client-ID', '')
    if not raw_cid:
        return jsonify({"success": False, "message": "Thiếu client_id"}), 400
        
    cid = sanitize_id(raw_cid)
    cleanup_session_temp_files('rtsp', cid)
    
    with clients_lock:
        if cid in clients:
            clients[cid]['logs'].clear()
            clients[cid]['alerts'].clear()
        global global_alerts
        global_alerts = [a for a in global_alerts if not str(a.get('id', '')).startswith(f"rtsp_{cid}_")]

    try:
        from app import app, db, RtspAlert
        with app.app_context():
            RtspAlert.query.filter_by(client_id=cid).delete()
            db.session.commit()
    except Exception:
        pass
    return jsonify({"success": True})

@rtsp_bp.route('/api/rtsp/stop', methods=['POST'])
def stop_rtsp():
    data = request.get_json(force=True, silent=True) or {}
    raw_cid = data.get('client_id', '') or request.headers.get('X-Client-ID', '')
    if not raw_cid:
        return jsonify({"success": True})
        
    cid = sanitize_id(raw_cid)
    # Tự động xóa toàn bộ file temp rtsp_<client_id>_* khi dừng hoặc thoát trang
    get_limiter('rtsp').release(cid, cleanup_temp=True)
    with clients_lock:
        if cid in clients:
            if clients[cid]['camera']:
                clients[cid]['camera'].release()
                clients[cid]['camera'] = None
            if is_auto_clean_enabled():
                clients[cid]['logs'].clear()
                clients[cid]['alerts'].clear()
    return jsonify({"success": True})

@rtsp_bp.route('/api/rtsp/config', methods=['GET', 'POST'])
def rtsp_config_api():
    from app import app, get_setting, update_setting
    if request.method == 'POST':
        data = request.get_json()
        with app.app_context():
            update_setting('rtsp_api_list', json.dumps(data.get('api_list', [])))
            update_setting('rtsp_concurrent_jobs', str(data.get('concurrent_jobs', 10)))
            
        start_workers_if_needed() 
        return jsonify({"success": True})
    
    with app.app_context():
        try: 
            api_list_parsed = json.loads(get_setting('rtsp_api_list', '[]'))
        except: 
            api_list_parsed = []
            
        try:
            concurrent_jobs = int(get_setting('rtsp_concurrent_jobs', 10))
        except (ValueError, TypeError):
            concurrent_jobs = 10
            
    return jsonify({"api_list": api_list_parsed, "concurrent_jobs": concurrent_jobs})