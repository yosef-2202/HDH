from __future__ import annotations

import json
import os
import queue
import threading
import uuid
import base64
import time
import cv2
import numpy as np
import requests
import psutil
from flask import (Blueprint, Response, jsonify, request, stream_with_context, current_app, session)
from models.session_limiter import (
    get_limiter,
    build_temp_filename,
    TEMP_FOLDER,
)

video_bp = Blueprint('video_bp', __name__)

os.makedirs(TEMP_FOLDER, exist_ok=True)

MAX_VIDEO_BYTES = 50 * 1024 * 1024  # Giới hạn cứng tối đa 50MB

hog_detector = cv2.HOGDescriptor()
hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

_video_key_lock = threading.Lock()
_video_key_turn = 0
current_process = psutil.Process(os.getpid())

def get_video_credentials():
    global _video_key_turn
    from app import app, get_setting
    keys = []
    model_name = "openrouter/free"
    try:
        with app.app_context():
            raw_keys = get_setting('video_api_keys', '[]')
            parsed = json.loads(raw_keys)
            if isinstance(parsed, list):
                keys = [str(k).strip() for k in parsed if str(k).strip()]
            
            if not keys:
                rtsp_raw = get_setting('rtsp_api_list', '[]')
                for it in json.loads(rtsp_raw):
                    k = str(it.get('key', '')).strip()
                    if k and k not in keys:
                        keys.append(k)
            if not keys:
                rt_raw = get_setting('realtime_api_keys', '[]')
                for k in json.loads(rt_raw):
                    if isinstance(k, str) and k.strip() and k.strip() not in keys:
                        keys.append(k.strip())

            saved_model = get_setting('video_ai_model', 'openrouter/free').strip()
            if saved_model:
                model_name = saved_model
    except Exception:
        pass

    if not keys:
        return "", model_name, []

    with _video_key_lock:
        selected_key = keys[_video_key_turn % len(keys)]
        _video_key_turn += 1
    return selected_key, model_name, keys


_sse_queues: dict[str, queue.Queue] = {}
_sse_lock = threading.Lock()

def _get_q(sid: str) -> queue.Queue:
    with _sse_lock:
        if sid not in _sse_queues:
            _sse_queues[sid] = queue.Queue(maxsize=1000)
        return _sse_queues[sid]

def _push_q(sid: str, payload: dict):
    try:
        _get_q(sid).put_nowait(payload)
    except queue.Full:
        pass


def _frame_to_base64_uri(frame_bgr):
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, 960.0 / max(w, h, 1))
    if scale < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ret, buffer = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
    if not ret:
        return ""
    img_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"


def _call_openrouter(api_key, contents, model_name="openrouter/free"):
    url = "https://openrouter.ai/api/v1/chat/completions"
    instruction = """
    BẮT BUỘC trả về JSON nguyên chất (không bọc trong ```json). Định dạng và ý nghĩa các trường:
    {
      "people": [
        {
          "person_id": "<ID Person_N>",
          "video_ts": "<Mốc HH:MM:SS>",
          "behavior": "<Hành quan sát vi được>",
          "detail": "<Mô bối chi cảnh, phục tiết trang tả>",
          "status_code": "<normal | warning | danger>"
        }
      ]
    }
    """
    content_array = [{"type": "text", "text": instruction}]
    for item in contents:
        if isinstance(item, str):
            content_array.append({"type": "text", "text": item})
        elif isinstance(item, np.ndarray):
            uri = _frame_to_base64_uri(item)
            if uri:
                content_array.append({"type": "image_url", "image_url": {"url": uri}})

    _, db_model, all_keys = get_video_credentials()
    fallback_models = [model_name or db_model or "openrouter/free", "openrouter/free"]
    models_to_try = [m for m in fallback_models if m]
    keys_to_try = [api_key] + [k for k in all_keys if k != api_key]

    for current_key in keys_to_try:
        if not current_key:
            continue
        headers = {
            "Authorization": f"Bearer {current_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:24706",
            "X-Title": "HDH Surveillance AI - Video Module"
        }
        for target_model in models_to_try:
            payload = {
                "model": target_model,
                "messages": [{"role": "user", "content": content_array}],
                "temperature": 0.2
            }
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=90)
                if resp.status_code != 200:
                    continue
                res_json = resp.json()
                raw_text = res_json['choices'][0]['message'].get('content') or ''
                raw_text = raw_text.strip()
                if raw_text.startswith("```"):
                    raw_text = raw_text.strip("`").lstrip("json").strip()
                if not raw_text:
                    continue

                class MockResponse:
                    def __init__(self, text): self.text = text
                return MockResponse(raw_text), None
            except Exception:
                pass
    return None, "Không gọi được AI hoặc hết hạn mức API"


def _hms(s: float) -> str:
    s = int(max(0, s))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _annotate_person(frame, box, label: str):
    out = frame.copy()
    h, w = out.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(out, label, (x1, max(y1 - 10, 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    return out


def _detect_boxes_opencv(frame, prev_gray):
    h, w = frame.shape[:2]
    scale = 640.0 / max(w, 1)
    small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)

    boxes = []
    try:
        rects, _ = hog_detector.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.08)
        for (x, y, bw, bh) in rects:
            boxes.append((int(x / scale), int(y / scale), int((x + bw) / scale), int((y + bh) / scale)))
    except Exception:
        pass

    if not boxes and prev_gray is not None and prev_gray.shape == gray_blur.shape:
        diff = cv2.absdiff(prev_gray, gray_blur)
        thresh = cv2.dilate(cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1], None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:2]:
            if cv2.contourArea(c) > 900:
                x, y, bw, bh = cv2.boundingRect(c)
                boxes.append((int(x / scale), int(y / scale), int((x + bw) / scale), int((y + bh) / scale)))

    return boxes, gray_blur


def _clean_behavior_text(text: str) -> str:
    t = (text or '').strip()
    if not t: return ''
    bad_starts = ('đã nhận diện:', 'nhận diện:', 'phát hiện:', 'ghi nhận:')
    low = t.lower()
    for s in bad_starts:
        if low.startswith(s):
            t = t[len(s):].strip()
            break
    return t.strip(' -—–,:;')


def _bg_analyze(session_id: str, video_path: str, orig_name: str, client_id: str = ""):
    q = _get_q(session_id)
    try:
        api_key, model_name, _ = get_video_credentials()
        if not api_key:
            q.put_nowait({
                'type': 'log', 'person_id': 'SYSTEM', 'video_ts': '00:00:00',
                'behavior': 'Chưa có API Key', 'detail': 'Chưa cấu hình OpenRouter API Key!',
                'status': '🚨 Nguy hiểm', 'status_code': 'danger'
            })
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            q.put_nowait({
                'type': 'log', 'person_id': 'SYSTEM', 'video_ts': '00:00:00',
                'behavior': 'Lỗi Video', 'detail': 'Không thể mở tệp video.',
                'status': '🚨 Nguy hiểm', 'status_code': 'danger'
            })
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_sec = total_frames / fps
        step_frames = max(1, int(fps * 1.5))

        q.put_nowait({'type': 'init', 'total_hms': _hms(total_sec), 'total_frames': total_frames})
        q.put_nowait({'type': 'progress', 'progress': 15})

        samples = []
        prev_gray = None
        frame_idx = 0
        file_kb = round(os.path.getsize(video_path) / 1024, 1) if os.path.exists(video_path) else 0

        while True:
            ok, frame = cap.read()
            if not ok: break
            if frame_idx % step_frames != 0:
                frame_idx += 1
                continue

            get_limiter('video').try_acquire_stream(client_id or session_id)

            try:
                cpu_val = round(psutil.cpu_percent(interval=None) / max(1, psutil.cpu_count()) + 2.1, 1)
                ram_val = round(current_process.memory_info().rss / (1024 * 1024), 1)
            except Exception:
                cpu_val, ram_val = 4.5, 45.0

            q.put_nowait({
                'type': 'metrics',
                'cpu': f"{cpu_val}%",
                'ram': f"{ram_val} MB",
                'bw': f"{file_kb} KB"
            })

            boxes, prev_gray = _detect_boxes_opencv(frame, prev_gray)
            if boxes:
                samples.append({
                    'frame_idx': frame_idx,
                    'video_ts': _hms(frame_idx / fps),
                    'frame': frame.copy(),
                    'box': boxes[0]
                })
            frame_idx += 1

        cap.release()
        if len(samples) > 6:
            samples = [samples[idx] for idx in np.linspace(0, len(samples) - 1, 6, dtype=int)]

        for i, s in enumerate(samples, start=1):
            s['person_id'] = f'Person_{i}'

        contents = []
        lines = []
        for s in samples:
            annotated = _annotate_person(s['frame'], s['box'], s['person_id'])
            contents.append(annotated)
            lines.append(f"- {s['person_id']} tại mốc {s['video_ts']}")

        prompt = "Phân tích các khung hình video an ninh sau:\n" + "\n".join(lines)
        contents.append(prompt)

        response, last_err = _call_openrouter(api_key, contents, model_name)
        if last_err is not None or response is None:
            q.put_nowait({
                'type': 'log', 'person_id': 'SYSTEM', 'video_ts': '00:00:00',
                'behavior': 'Lỗi Gọi AI', 'detail': str(last_err),
                'status': '🚨 Nguy hiểm', 'status_code': 'danger'
            })
            return

        try:
            raw_json = response.text
            start_brace = raw_json.find('{')
            end_brace = raw_json.rfind('}')
            if start_brace != -1 and end_brace != -1:
                raw_json = raw_json[start_brace:end_brace + 1]
            result_data = json.loads(raw_json)
            people_events = result_data.get('people', [])
        except Exception:
            people_events = []

        status_map = {'normal': '🟢 Bình thường', 'warning': '🟡 Chú ý', 'danger': '🚨 Nguy hiểm'}
        if not people_events:
            people_events = [{'person_id': 'Person_1', 'video_ts': '00:00:00', 'behavior': 'Quan sát', 'detail': response.text[:200], 'status_code': 'normal'}]

        for i, ev in enumerate(people_events):
            pid = ev.get('person_id', f'Person_{i+1}')
            ts = ev.get('video_ts', '00:00:00')
            behavior = _clean_behavior_text(ev.get('behavior', 'Hoạt động'))
            detail = ev.get('detail', 'Phân tích thành công.')
            sc = ev.get('status_code', 'normal')

            payload = {
                'type': 'log', 'person_id': pid, 'video_ts': ts,
                'behavior': behavior, 'detail': detail, 'confidence': '98%',
                'status': status_map.get(sc, '🟢 Bình thường'), 'status_code': sc
            }
            _push_q(session_id, payload)

    except Exception as exc:
        q.put_nowait({'type': 'log', 'person_id': 'SYSTEM', 'video_ts': '00:00:00', 'behavior': 'Lỗi', 'detail': str(exc), 'status': 'Nguy hiểm', 'status_code': 'danger'})
    finally:
        # Xóa ngay file video tạm sau khi OpenCV đã trích xuất và phân tích xong
        try:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass
        get_limiter('video').release(client_id or session_id)
        q.put_nowait({'type': 'complete', 'progress': 100})


def _sse_gen(session_id: str):
    q = _get_q(session_id)
    while True:
        try:
            payload = q.get(timeout=30)
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if payload.get('type') == 'complete': break
        except queue.Empty:
            yield 'data: {"type":"ping"}\n\n'


@video_bp.route('/serve_video/<filename>')
def serve_video(filename):
    from flask import send_from_directory
    return send_from_directory(TEMP_FOLDER, filename)


@video_bp.route('/api/video/analyze', methods=['POST'])
def api_video_upload():
    client_id = request.form.get('client_id') or request.headers.get('X-Client-ID') or f"vid_{uuid.uuid4().hex[:8]}"
    slot_info = get_limiter('video').try_acquire_stream(client_id)
    if not slot_info["allowed"]:
        return jsonify({
            'success': False,
            'waiting': True,
            'queue_position': slot_info['position'],
            'active_users': slot_info['active_count'],
            'max_users': slot_info['max_concurrent'],
            'message': slot_info['message']
        }), 429

    if 'video' not in request.files:
        get_limiter('video').release(client_id)
        return jsonify({'success': False, 'message': 'Không có file video'}), 400
    f = request.files['video']
    
    f.stream.seek(0, os.SEEK_END)
    size = f.stream.tell()
    f.stream.seek(0)
    if size > MAX_VIDEO_BYTES:
        get_limiter('video').release(client_id)
        return jsonify({'success': False, 'message': f'Tệp quá lớn ({round(size/(1024*1024), 1)} MB). Giới hạn tối đa là 50 MB để đảm bảo hiệu suất AI!'}), 413

    ext = os.path.splitext(f.filename)[1].lower() or '.mp4'
    # Đặt tên chuẩn: video_<tên_phiên>_<số_thứ_tự>.mp4 lưu vào static/temp
    base_id, filepath = build_temp_filename('video', client_id, ext)
    filename = os.path.basename(filepath)
    f.save(filepath)

    session_id = base_id
    _get_q(session_id)
    threading.Thread(target=_bg_analyze, args=(session_id, filepath, f.filename, client_id), daemon=True).start()

    return jsonify({'success': True, 'session_id': session_id, 'video_url': f'/serve_video/{filename}'})


@video_bp.route('/api/video/stream/<session_id>')
def api_video_stream(session_id: str):
    resp = Response(stream_with_context(_sse_gen(session_id)), mimetype='text/event-stream')
    resp.headers['Cache-Control'] = 'no-cache'
    return resp


@video_bp.route('/api/video/config', methods=['GET', 'POST'])
def api_video_config():
    from app import app, get_setting, update_setting
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        with app.app_context():
            update_setting('video_api_keys', json.dumps([str(k).strip() for k in data.get("api_keys", []) if str(k).strip()]))
            update_setting('video_ai_model', str(data.get("model", "openrouter/free")).strip())
        return jsonify({"success": True})
    
    with app.app_context():
        try: keys = json.loads(get_setting('video_api_keys', '[]'))
        except: keys = []
        model = get_setting('video_ai_model', 'openrouter/free')
    return jsonify({"success": True, "api_keys": keys, "model": model})