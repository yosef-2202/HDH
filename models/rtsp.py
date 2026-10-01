from flask import Blueprint, render_template, Response, request, jsonify
import cv2
import threading
import time
import sys
import os
from datetime import datetime
import google.generativeai as genai
from PIL import Image

rtsp_bp = Blueprint('rtsp_bp', __name__)

camera = None
camera_lock = threading.Lock()

global_logs = []
global_alerts = []
gemini_is_analyzing = False
token_turn = 0 
last_api_call_time = 0

os.makedirs('static/temp', exist_ok=True)

def record_short_clip(frame_list, alert_id, bboxes=None):
    if not frame_list: return
    try:
        height, width, _ = frame_list[0].shape
        filepath = f"static/temp/{alert_id}.webm"
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        out = cv2.VideoWriter(filepath, fourcc, 15.0, (width, height))

        # Lưu trữ mảng tọa độ khung cho TỪNG frame
        tracked_boxes_per_frame = [ [] for _ in range(len(frame_list)) ]
        
        if bboxes and len(bboxes) > 0:
            trackers = []
            for bbox in bboxes:
                ymin, xmin, ymax, xmax = bbox
                # Quy đổi về pixel
                y1 = max(0, int(ymin * height / 1000))
                x1 = max(0, int(xmin * width / 1000))
                y2 = min(height, int(ymax * height / 1000))
                x2 = min(width, int(xmax * width / 1000))
                
                w_box = max(1, x2 - x1)
                h_box = max(1, y2 - y1)
                initial_bbox = (x1, y1, w_box, h_box)
                
                # Khởi tạo tracker cho TỪNG người
                try:
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame_list[-1], initial_bbox)
                    trackers.append({'tracker': tracker, 'last_box': initial_bbox})
                except AttributeError:
                    # Fallback nếu OpenCV không có module Tracker
                    trackers.append({'tracker': None, 'last_box': initial_bbox})
            
            # Gán hộp ở frame cuối cùng (mốc do AI phân tích)
            tracked_boxes_per_frame[-1] = [t['last_box'] for t in trackers]

            # Tracking giật lùi về quá khứ cho tất cả các đối tượng
            for i in range(len(frame_list) - 2, -1, -1):
                current_frame_boxes = []
                for t in trackers:
                    if t['tracker'] is not None:
                        success, box = t['tracker'].update(frame_list[i])
                        if success:
                            t['last_box'] = box
                            current_frame_boxes.append(box)
                        else:
                            current_frame_boxes.append(t['last_box']) # Nếu mất dấu, giữ hộp ở vị trí cũ
                    else:
                        current_frame_boxes.append(t['last_box'])
                tracked_boxes_per_frame[i] = current_frame_boxes

        # Ghi frame kèm tất cả các bounding box
        for i, f in enumerate(frame_list):
            frame_copy = f.copy()
            if bboxes and i < len(tracked_boxes_per_frame):
                for box in tracked_boxes_per_frame[i]:
                    tx, ty, tw, th = [int(v) for v in box]
                    cv2.rectangle(frame_copy, (tx, ty), (tx + tw, ty + th), (0, 0, 255), 2)
            out.write(frame_copy)
            
        out.release()
    except Exception as e:
        print(f"=> LỖI LƯU VIDEO TEMP: {e}", file=sys.stderr)

def analyze_frame_with_gemini(frame, recent_frames, alert_id):
    global gemini_is_analyzing, global_logs, global_alerts, token_turn
    try:
        from app import app, get_setting
        with app.app_context():
            token1 = get_setting('token1', '').strip()
            token2 = get_setting('token2', '').strip()
            
        active_tokens = []
        if token1: active_tokens.append(("API 1", token1))
        if token2: active_tokens.append(("API 2", token2))
        
        if not active_tokens:
            print("=> LỖI: Chưa cấu hình Token Gemini", file=sys.stderr)
            return

        api_label, api_key = active_tokens[token_turn % len(active_tokens)]
        token_turn += 1

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3.8-flash')

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        # PROMPT MỚI: Bắt buộc tách từng người, mô tả sâu và xuất toạ độ riêng
        prompt = """Bạn là AI giám sát an ninh. Phân tích ảnh và tìm TỪNG NGƯỜI một cách riêng biệt.
Nếu không có ai, trả lời 'Không'.
Nếu có người, hãy liệt kê MỖI NGƯỜI TRÊN 1 DÒNG theo định dạng:
[Mô tả chi tiết người đó đang làm gì] | ymin, xmin, ymax, xmax
(ymin, xmin, ymax, xmax là toạ độ hộp bao quanh người tỷ lệ 0-1000).
Ví dụ:
Người áo đen đang với tay lấy đồ | 200, 300, 800, 600
Người đội nón đang đẩy xe | 350, 500, 900, 700"""
        
        start_time = time.time()
        response = model.generate_content([prompt, pil_img])
        process_time = round(time.time() - start_time, 2)
        
        text = response.text.strip()
        lines = text.split('\n')
        
        detected_persons = []
        bboxes = []
        
        # Parse từng dòng để bóc tách nhiều người
        for line in lines:
            if "|" in line:
                parts = line.split("|")
                desc = parts[0].strip()
                try:
                    coords = parts[1].strip().replace('[','').replace(']','').split(",")
                    if len(coords) == 4:
                        bbox = tuple(int(x.strip()) for x in coords)
                        detected_persons.append(desc)
                        bboxes.append(bbox)
                except:
                    pass
        
        if detected_persons:
            # Render thông báo nhiều người
            num_people = len(detected_persons)
            behavior_title = f"Phát hiện {num_people} người"
            action_desc = "<br>".join([f"- {person}" for person in detected_persons])
            
            threading.Thread(target=record_short_clip, args=(list(recent_frames), alert_id, bboxes)).start()
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            masked_key = f"...{api_key[-4:]}"
            
            log_msg = f"[{timestamp}] [{api_label}] {behavior_title} (Xử lý: {process_time}s) - Cấp độ: Lưu ý"
            
            if len(global_logs) > 50: global_logs.pop(0)
            global_logs.append(log_msg)

            global_alerts.insert(0, {
                "id": alert_id,
                "time": timestamp,
                "behavior": behavior_title,
                "process_time": f"{process_time}s",
                "level": "Lưu ý",
                "badge": "warning",
                "desc": action_desc,
                "api_info": f"{api_label}: {masked_key}",
                "temp_video_url": f"/static/temp/{alert_id}.webm" 
            })
            if len(global_alerts) > 20: global_alerts.pop()
    except Exception as e:
        print(f"=> LỖI GEMINI API: {e}", file=sys.stderr)
    finally:
        gemini_is_analyzing = False

def generate_frames(rtsp_url):
    global camera, gemini_is_analyzing, global_logs, global_alerts, last_api_call_time
    
    with camera_lock:
        if camera is not None:
            camera.release()
        camera = cv2.VideoCapture(rtsp_url)
    
    recent_frames = [] 
    api_cooldown = 5 # RÚT NGẮN XUỐNG 5 GIÂY (để quét nhiều hơn)

    while True:
        with camera_lock:
            success, frame = camera.read()
            
        if not success:
            break
        else:
            frame_resized = cv2.resize(frame, (640, 360))
            
            recent_frames.append(frame_resized.copy())
            if len(recent_frames) > 45: 
                recent_frames.pop(0)
            
            current_time = time.time()
            time_since_last_call = current_time - last_api_call_time
            countdown = max(0, int(api_cooldown - time_since_last_call))
            
            if countdown == 0 and not gemini_is_analyzing:
                gemini_is_analyzing = True
                last_api_call_time = current_time
                alert_id = f"temp_vid_{int(time.time())}"
                threading.Thread(target=analyze_frame_with_gemini, args=(frame_resized.copy(), list(recent_frames), alert_id)).start()

            status_text = f"AI Cooldown: {countdown}s" if countdown > 0 else "AI: Dang phan tich..."
            cv2.putText(frame_resized, status_text, (400, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if countdown > 0 else (0, 255, 0), 2)
            cv2.putText(frame_resized, "Gemini AI: ON", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            time.sleep(0.066) 
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@rtsp_bp.route('/rtsp')
def rtsp_page():
    return render_template('rtsp.html')

@rtsp_bp.route('/video_feed')
def video_feed():
    rtsp_url = request.args.get('url', '')
    if not rtsp_url:
        return "Vui lòng cung cấp đường dẫn RTSP", 400
    
    global global_logs, global_alerts
    global_logs.clear()
    global_alerts.clear()
    global_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] [SYSTEM] Bắt đầu kết nối luồng & kích hoạt Gemini AI...")
        
    return Response(generate_frames(rtsp_url), mimetype='multipart/x-mixed-replace; boundary=frame')

@rtsp_bp.route('/api/rtsp/data')
def rtsp_data():
    return jsonify({
        "logs": global_logs,
        "alerts": global_alerts
    })
    
@rtsp_bp.route('/api/rtsp/delete_temp/<clip_id>', methods=['POST'])
def delete_temp(clip_id):
    filepath = f"static/temp/{clip_id}.webm"
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except:
            pass
    return jsonify({"success": True})

@rtsp_bp.route('/api/rtsp/clear_cache', methods=['POST'])
def clear_cache():
    global global_logs, global_alerts
    global_logs.clear()
    global_alerts.clear()
    return jsonify({"success": True})

@rtsp_bp.route('/api/rtsp/gemini_status')
def gemini_status():
    from app import app, get_setting
    with app.app_context():
        token1 = get_setting('token1', '').strip()
        token2 = get_setting('token2', '').strip()
    
    status = {}
    for i, token in enumerate([token1, token2], start=1):
        key_name = f"API_Key_{i}"
        if not token:
            status[key_name] = {"text": "Trống", "color": "secondary"}
            continue
        try:
            genai.configure(api_key=token)
            genai.get_model('models/gemini-3.8-flash')
            status[key_name] = {"text": "Hoạt động", "color": "success"}
        except Exception:
            status[key_name] = {"text": "Lỗi / Die", "color": "danger"}
            
    return jsonify(status)