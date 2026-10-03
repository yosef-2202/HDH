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
        
    valid_apis = [api for api in api_list if api.get('key', '').strip() != '']
    
    if not valid_apis: return None, None, None, None

    with token_lock:
        api_index = token_turn % len(valid_apis)
        selected_api = valid_apis[api_index]
        token_turn += 1
        
    api_label = f"API_{api_index + 1}"
    api_key = selected_api['key'].strip()
    
    model_name = selected_api.get('model', '').strip()
    if not model_name: 
        model_name = 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free'
        
    return api_label, api_key, valid_apis, model_name

def start_workers_if_needed():
    global workers_started
    with worker_lock:
        if not workers_started:
            from app import app, get_setting
            with app.app_context():
                try: num_workers = int(get_setting('rtsp_concurrent_jobs', 10)) 
                except: num_workers = 10
            
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
    
    api_label, api_key, valid_apis, model_name = get_api_credentials()
    if not api_key:
        return

    prompt = """Bạn là hệ thống AI giám sát an ninh, giao thông và cảnh quan qua camera thông minh.
NHIỆM VỤ: Phân tích ảnh, PHÁT HIỆN CON NGƯỜI, PHƯƠNG TIỆN VÀ ĐỘNG VẬT (chó, bò, v.v.), mô tả mọi hành vi của chúng và DỰ ĐOÁN tình huống. HOÀN TOÀN BẰNG TIẾNG VIỆT.

QUY TẮC CỐT LÕI:
1. TUYỆT ĐỐI KHÔNG dùng tiếng Anh.
2. Bỏ qua mô tả ngoại hình rườm rà. Tập trung tối đa vào tư thế, hành động, quỹ đạo di chuyển.
3. Nếu không có bất kỳ Người, Xe cộ hay Động vật nào trong ảnh, trả lời chính xác chữ: 'Không'.

ĐỊNH DẠNG TRẢ LỜI (Tuân thủ nghiêm ngặt):

PHẦN 1: DANH SÁCH ĐỐI TƯỢNG (Mỗi đối tượng liệt kê trên một dòng)
[Loại đối tượng] | [Hành vi chính] | [Mô tả chi tiết]
VD: [Người] | [Đi bộ] | [Đang đi qua đường]
VD: [Ô tô] | [Đứng yên] | [Đang đậu ở giữa lô]

PHẦN 2: KẾT LUẬN & DỰ ĐOÁN TÌNH HUỐNG
- Mức độ an toàn: [Bình thường / Đáng ngờ / Vi phạm / Cảnh báo / Nguy hiểm khẩn cấp]
- Dự đoán sự việc: [Suy luận tình huống]"""
    
    max_retries = max(1, len(valid_apis)) 
    
    for attempt in range(max_retries):
        try:
            timestamp_start = datetime.now().strftime("%H:%M:%S")
            start_msg = f"[{timestamp_start}] [TIẾN TRÌNH] Đang phân tích bằng {api_label}..."
            if len(global_logs) > 50: global_logs.pop(0)
            global_logs.append(start_msg)

            start_time = time.time()
            
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
            
            try:
                res_json = response.json()
            except ValueError:
                raise ValueError("API không trả về định dạng JSON (có thể server API quá tải).")
                
            if not res_json or 'choices' not in res_json or len(res_json['choices']) == 0:
                raise ValueError("API trả về dữ liệu không hợp lệ (thiếu trường 'choices')")
                
            text = res_json['choices'][0]['message']['content'].strip()
            process_time = round(time.time() - start_time, 2)
            
            if text.lower().startswith("không") and len(text) < 15:
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_msg = f"[{timestamp}] [{api_label}] Không phát hiện đối tượng (Tốc độ: {process_time}s)"
                if len(global_logs) > 50: global_logs.pop(0)
                global_logs.append(log_msg)
                break

            part1 = text
            part2 = ""
            if "PHẦN 2" in text:
                parts_split = text.split("PHẦN 2")
                part1 = parts_split[0]
                part2 = parts_split[1]

            detected_objects = []
            main_actions = []
            
            for line in part1.split('\n'):
                if "|" in line:
                    parts = line.split("|")
                    if len(parts) >= 2:
                        try:
                            session_id = str(uuid.uuid4())[:6].upper()
                            obj_type = parts[0].replace("[", "").replace("]", "").strip()
                            main_act = parts[1].replace("[", "").replace("]", "").strip()
                            
                            desc = ""
                            if len(parts) >= 3:
                                desc = parts[2].replace("[", "").replace("]", "").strip()
                                
                            detected_objects.append(f"[Phiên: {session_id}] [{obj_type}: {main_act} - {desc}]")
                            if main_act:
                                main_actions.append(main_act)
                        except: pass
            
            safety_level = "Theo dõi"
            badge_color = "info"
            prediction = "Chưa có dự đoán"

            for line in part2.split('\n'):
                line_clean = line.strip()
                if "Mức độ an toàn:" in line_clean:
                    safety_level = line_clean.split("Mức độ an toàn:")[1].strip().replace("[", "").replace("]", "")
                    lvl_lower = safety_level.lower()
                    if "nguy hiểm" in lvl_lower or "khẩn cấp" in lvl_lower: badge_color = "danger"
                    elif "cảnh báo" in lvl_lower or "vi phạm" in lvl_lower or "đáng ngờ" in lvl_lower: badge_color = "warning"
                    elif "bình thường" in lvl_lower: badge_color = "success"
                
                elif "Dự đoán sự việc:" in line_clean:
                    prediction = line_clean.split("Dự đoán sự việc:")[1].strip().replace("[", "").replace("]", "")

            if detected_objects:
                unique_actions = list(set(main_actions))
                behavior_title = ", ".join(unique_actions) if unique_actions else "Phát hiện đối tượng"
                
                action_desc = "<b>Đối tượng:</b><br>" + "<br>".join([f"- {p}" for p in detected_objects])
                action_desc += f"<br><br><b>Dự đoán:</b> <i>{prediction}</i>"
                
                threading.Thread(target=record_short_clip, args=(list(recent_frames), alert_id)).start()
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_msg = f"[{timestamp}] [{api_label}] {safety_level.upper()}: {behavior_title} (Tốc độ: {process_time}s)"
                
                if len(global_logs) > 50: global_logs.pop(0)
                global_logs.append(log_msg)

                global_alerts.insert(0, {
                    "id": alert_id,
                    "time": timestamp,
                    "behavior": behavior_title,
                    "process_time": f"{process_time}s",
                    "level": safety_level,
                    "badge": badge_color,
                    "desc": action_desc,
                    "api_info": f"Model AI",
                    "temp_video_url": f"/static/temp/{alert_id}.webm" 
                })
                if len(global_alerts) > 20: global_alerts.pop()
            
            break 
            
        except requests.exceptions.RequestException as e:
            is_429 = hasattr(e, 'response') and e.response is not None and e.response.status_code == 429
            if attempt < max_retries - 1:
                if is_429:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    msg = f"[{timestamp}] [CẢNH BÁO] {api_label} dính Rate Limit, nghỉ 5s rồi đổi..."
                    if len(global_logs) > 50: global_logs.pop(0)
                    global_logs.append(msg)
                    time.sleep(5) 
                else:
                    time.sleep(1)
                api_label, api_key, valid_apis, model_name = get_api_credentials() 
            else:
                pass 
                
        except Exception as e:
            if attempt < max_retries - 1:
                timestamp = datetime.now().strftime("%H:%M:%S")
                msg = f"[{timestamp}] [CẢNH BÁO] {api_label} lỗi phản hồi, chuyển token..."
                if len(global_logs) > 50: global_logs.pop(0)
                global_logs.append(msg)
                
                api_label, api_key, valid_apis, model_name = get_api_credentials() 
                time.sleep(1)
            else:
                pass

def generate_frames(rtsp_url):
    global camera, last_api_call_time
    start_workers_if_needed() 
    
    with camera_lock:
        if camera is not None: 
            camera.release()
            time.sleep(0.5)
        os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "100"
        camera = cv2.VideoCapture(rtsp_url)
    
    recent_frames = [] 
    api_cooldown = 5 

    while True:
        with camera_lock:
            if camera is None or not camera.isOpened():
                break
            success, frame = camera.read()
            
        if not success: 
            break
        
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

        try:
            ret, buffer = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            if not ret: continue
            time.sleep(0.066) 
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception:
            break

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

@rtsp_bp.route('/api/rtsp/stop', methods=['POST'])
def stop_rtsp():
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
    return jsonify({"success": True})

@rtsp_bp.route('/api/rtsp/config', methods=['GET', 'POST'])
def rtsp_config_api():
    from app import app, get_setting, update_setting
    if request.method == 'POST':
        data = request.get_json()
        api_list_data = data.get('api_list', [])
        
        with app.app_context():
            update_setting('rtsp_api_list', json.dumps(api_list_data))
            update_setting('rtsp_concurrent_jobs', str(data.get('concurrent_jobs', 10))) 
            
        return jsonify({"success": True})
    
    with app.app_context():
        try: api_list_parsed = json.loads(get_setting('rtsp_api_list', '[]'))
        except: api_list_parsed = []
        
        concurrent_jobs = int(get_setting('rtsp_concurrent_jobs', 10))
        
    return jsonify({
        "api_list": api_list_parsed,
        "concurrent_jobs": concurrent_jobs
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
        if not token: continue
            
        try:
            res = requests.get("https://openrouter.ai/api/v1/auth/key", headers={"Authorization": f"Bearer {token}"}, timeout=5)
            if res.status_code == 200:
                status[key_name] = {"text": "Hoạt động", "color": "success"}
            else:
                status[key_name] = {"text": "Lỗi Token", "color": "danger"}
        except Exception:
            status[key_name] = {"text": "Lỗi Mạng", "color": "danger"}
            
    if not status:
        status["Hệ thống"] = {"text": "Chưa có API Key", "color": "secondary"}
        
    return jsonify(status)