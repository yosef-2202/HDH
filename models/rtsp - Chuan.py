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
from datetime import datetime
import google.generativeai as genai
from PIL import Image

rtsp_bp = Blueprint('rtsp_bp', __name__)

camera = None
camera_lock = threading.Lock()

global_logs = []
global_alerts = []
last_api_call_time = 0

analysis_queue = queue.Queue(maxsize=100)
token_lock = threading.Lock()
token_turn = 0

workers_started = False
worker_lock = threading.Lock()

os.makedirs('static/temp', exist_ok=True)

def get_api_credentials():
    global token_turn
    from app import app, get_setting
    with app.app_context():
        try:
            api_list = json.loads(get_setting('rtsp_api_list', '[]'))
        except:
            api_list = []
        
        forced_model_mode = get_setting('rtsp_forced_model_mode', 'auto')
        
    valid_apis = [api for api in api_list if api.get('key', '').strip() != '']
    
    if not valid_apis: return None, None, None, None, None

    with token_lock:
        if forced_model_mode == 'auto':
            selected_api = valid_apis[token_turn % len(valid_apis)]
            token_turn += 1
        else:
            filtered_apis = [api for api in valid_apis if api['provider'] == forced_model_mode]
            if filtered_apis:
                selected_api = filtered_apis[token_turn % len(filtered_apis)]
                token_turn += 1
            else:
                selected_api = valid_apis[token_turn % len(valid_apis)]
                token_turn += 1
        
    api_label = f"Token_{token_turn}"
    api_key = selected_api['key'].strip()
    ai_provider = selected_api['provider']
    
    if ai_provider == 'gemini': 
        model_name = 'gemini-2.5-flash'
    else:
        model_name = selected_api.get('model', '').strip()
        if not model_name: model_name = 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free'
        
    return api_label, api_key, valid_apis, model_name, ai_provider

def start_workers_if_needed():
    global workers_started
    with worker_lock:
        if not workers_started:
            from app import app, get_setting
            with app.app_context():
                try: num_workers = int(get_setting('rtsp_concurrent_jobs', 2)) 
                except: num_workers = 2
            
            for i in range(num_workers):
                threading.Thread(target=ai_worker, daemon=True, name=f"AI-Worker-{i}").start()
            workers_started = True

def ai_worker():
    while True:
        task = analysis_queue.get()
        if task is None: break
        frame, recent_frames, alert_id = task
        analyze_frame_with_ai(frame, recent_frames, alert_id)
        analysis_queue.task_done()

def record_short_clip(frame_list, alert_id):
    if not frame_list: return
    try:
        height, width, _ = frame_list[0].shape
        filepath = f"static/temp/{alert_id}.webm"
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        out = cv2.VideoWriter(filepath, fourcc, 15.0, (width, height))
        for f in frame_list:
            out.write(f)
        out.release()
    except Exception as e:
        print(f"=> LỖI LƯU VIDEO TEMP: {e}", file=sys.stderr)

def analyze_frame_with_ai(frame, recent_frames, alert_id):
    global global_logs, global_alerts
    
    api_label, api_key, valid_apis, model_name, ai_provider = get_api_credentials()
    if not api_key:
        print("=> LỖI: Chưa cấu hình Token API", file=sys.stderr)
        return

    prompt = """Bạn là hệ thống AI giám sát an ninh camera thông minh.
NHIỆM VỤ: Phân tích ảnh, CHỈ PHÁT HIỆN CON NGƯỜI và mô tả hành vi của họ HOÀN TOÀN BẰNG TIẾNG VIỆT.
TUYỆT ĐỐI KHÔNG dùng tiếng Anh. Bỏ qua mô tả quần áo rườm rà, tập trung vào hành động.
Nếu không có con người, trả lời chính xác chữ: 'Không'.
Nếu có người, hãy liệt kê MỖI NGƯỜI TRÊN MỘT DÒNG theo đúng định dạng:
[Hành động chính: Đi lại/Đứng quan sát/Nhìn ngó/Tương tác/Đánh nhau] - [Mô tả chi tiết hành động bằng tiếng Việt] | ymin, xmin, ymax, xmax"""
    
    max_retries = max(1, len(valid_apis)) 
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            text = ""
            
            if ai_provider == 'gemini':
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                response = model.generate_content([prompt, pil_img])
                text = response.text.strip()
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                img_b64 = base64.b64encode(buffer).decode('utf-8')
                image_data_url = f"data:image/jpeg;base64,{img_b64}"
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_data_url}}]}]
                }
                response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=30)
                response.raise_for_status() 
                text = response.json()['choices'][0]['message']['content'].strip()
            
            process_time = round(time.time() - start_time, 2)
            lines = text.split('\n')
            detected_persons = []
            main_actions = []
            
            for line in lines:
                if "|" in line:
                    parts = line.split("|")
                    try:
                        coords = parts[1].strip().replace('[','').replace(']','').split(",")
                        if len(coords) == 4:
                            session_id = str(uuid.uuid4())[:6].upper()
                            raw_action = parts[0].strip()
                            detected_persons.append(f"[Phiên: {session_id}] {raw_action}")
                            
                            main_act = "Phát hiện người"
                            if "[" in raw_action and "]" in raw_action:
                                bracket_content = raw_action.split("]")[0].replace("[", "")
                                if ":" in bracket_content:
                                    main_act = bracket_content.split(":")[1].strip()
                                else:
                                    main_act = bracket_content.strip()
                            elif "-" in raw_action:
                                main_act = raw_action.split("-")[0].strip()
                                
                            main_actions.append(main_act)
                    except: pass
            
            if detected_persons:
                unique_actions = list(set(main_actions))
                action_summary = ", ".join(unique_actions)
                
                behavior_title = f"{action_summary} ({ai_provider})"
                action_desc = "<br>".join([f"- {p}" for p in detected_persons])
                
                threading.Thread(target=record_short_clip, args=(list(recent_frames), alert_id)).start()
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_msg = f"[{timestamp}] [{api_label}] {behavior_title} (Đang chờ: {analysis_queue.qsize()}) - Tốc độ: {process_time}s"
                
                if len(global_logs) > 50: global_logs.pop(0)
                global_logs.append(log_msg)

                global_alerts.insert(0, {
                    "id": alert_id,
                    "time": timestamp,
                    "behavior": behavior_title,
                    "process_time": f"{process_time}s",
                    "level": "Theo dõi",
                    "badge": "info",
                    "desc": action_desc,
                    "api_info": f"{ai_provider}: ...{api_key[-4:]}",
                    "temp_video_url": f"/static/temp/{alert_id}.webm" 
                })
                if len(global_alerts) > 20: global_alerts.pop()
            
            break 
            
        except Exception as e:
            if attempt < max_retries - 1:
                api_label, api_key, valid_apis, model_name, ai_provider = get_api_credentials() 
                time.sleep(1)
            else:
                print(f"=> LỖI TOÀN BỘ TOKEN: {e}", file=sys.stderr)

def generate_frames(rtsp_url):
    global camera, last_api_call_time
    start_workers_if_needed() 
    
    with camera_lock:
        if camera is not None: camera.release()
        camera = cv2.VideoCapture(rtsp_url)
    
    recent_frames = [] 
    api_cooldown = 4 

    while True:
        with camera_lock:
            success, frame = camera.read()
        if not success: break
        
        frame_resized = cv2.resize(frame, (640, 360))
        recent_frames.append(frame_resized.copy())
        if len(recent_frames) > 45: recent_frames.pop(0)
        
        current_time = time.time()
        countdown = max(0, int(api_cooldown - (current_time - last_api_call_time)))
        
        if countdown == 0 and not analysis_queue.full():
            last_api_call_time = current_time
            alert_id = f"temp_vid_{int(time.time())}"
            analysis_queue.put((frame_resized.copy(), list(recent_frames), alert_id))

        q_size = analysis_queue.qsize()
        status_text = f"Queue: {q_size}/100 - Cooldown: {countdown}s" if countdown > 0 else f"Queue: {q_size}/100 - Waiting..."
        cv2.putText(frame_resized, status_text, (350, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0) if q_size > 0 else (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        time.sleep(0.066) 
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- CÁC ROUTE API DÀNH RIÊNG CHO MODULE RTSP ---
@rtsp_bp.route('/rtsp')
def rtsp_page(): return render_template('rtsp.html')

@rtsp_bp.route('/video_feed')
def video_feed():
    rtsp_url = request.args.get('url', '')
    if not rtsp_url: return "Vui lòng cung cấp đường dẫn RTSP", 400
    global global_logs, global_alerts
    global_logs.clear()
    global_alerts.clear()
    global_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] [SYSTEM] Kết nối luồng & khởi động Hàng Chờ...")
    return Response(generate_frames(rtsp_url), mimetype='multipart/x-mixed-replace; boundary=frame')

@rtsp_bp.route('/api/rtsp/data')
def rtsp_data(): return jsonify({"logs": global_logs, "alerts": global_alerts})

@rtsp_bp.route('/api/rtsp/delete_temp/<clip_id>', methods=['POST'])
def delete_temp(clip_id):
    filepath = f"static/temp/{clip_id}.webm"
    if os.path.exists(filepath):
        try: os.remove(filepath)
        except: pass
    return jsonify({"success": True})

@rtsp_bp.route('/api/rtsp/clear_cache', methods=['POST'])
def clear_cache():
    global global_logs, global_alerts
    global_logs.clear()
    global_alerts.clear()
    return jsonify({"success": True})

@rtsp_bp.route('/api/rtsp/set_mode', methods=['POST'])
def set_model_mode():
    from app import app, update_setting
    data = request.get_json()
    mode = data.get('mode', 'auto')
    with app.app_context():
        update_setting('rtsp_forced_model_mode', mode)
    return jsonify({"success": True})

@rtsp_bp.route('/api/rtsp/config', methods=['GET', 'POST'])
def rtsp_config_api():
    from app import app, get_setting, update_setting
    if request.method == 'POST':
        data = request.get_json()
        api_list_data = data.get('api_list', [])
        
        with app.app_context():
            update_setting('rtsp_api_list', json.dumps(api_list_data))
            update_setting('rtsp_concurrent_jobs', str(data.get('concurrent_jobs', 2)))
            update_setting('rtsp_forced_model_mode', data.get('forced_model_mode', 'auto'))
            
        return jsonify({"success": True})
    
    with app.app_context():
        try: api_list_parsed = json.loads(get_setting('rtsp_api_list', '[]'))
        except: api_list_parsed = []
        
        concurrent_jobs = int(get_setting('rtsp_concurrent_jobs', 2))
        forced_model_mode = get_setting('rtsp_forced_model_mode', 'auto')
        
    return jsonify({
        "api_list": api_list_parsed,
        "concurrent_jobs": concurrent_jobs,
        "forced_model_mode": forced_model_mode
    })

@rtsp_bp.route('/api/rtsp/gemini_status')
def api_status_check():
    from app import app, get_setting
    with app.app_context():
        try: api_list = json.loads(get_setting('rtsp_api_list', '[]'))
        except: api_list = []
    
    status = {}
    for i, api in enumerate(api_list, start=1):
        key_name = f"API_{i}"
        token = api.get('key', '').strip()
        provider = api.get('provider', 'gemini')
        if not token: continue
            
        try:
            if provider == 'gemini':
                genai.configure(api_key=token)
                genai.get_model('models/gemini-2.5-flash')
                status[key_name] = {"text": "Hoạt động (Gemini)", "color": "success"}
            else:
                res = requests.get("https://openrouter.ai/api/v1/auth/key", headers={"Authorization": f"Bearer {token}"}, timeout=5)
                if res.status_code == 200:
                    status[key_name] = {"text": "Hoạt động (OpenRouter)", "color": "success"}
                else:
                    status[key_name] = {"text": "Lỗi Token", "color": "danger"}
        except Exception:
            status[key_name] = {"text": "Lỗi Mạng", "color": "danger"}
            
    if not status:
        status["Hệ thống"] = {"text": "Chưa có API Key nào được thiết lập", "color": "secondary"}
        
    return jsonify(status)