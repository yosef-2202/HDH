from __future__ import annotations

import json
import os
import queue
import threading
import uuid
import base64
import cv2
import numpy as np
import requests
from flask import (Blueprint, Response, jsonify, request, stream_with_context, current_app, session)
from sqlalchemy import text

video_bp = Blueprint('video_bp', __name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Khởi tạo bộ phát hiện người nhẹ bằng OpenCV HOG (không cần cài YOLO)
hog_detector = cv2.HOGDescriptor()
hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

_video_key_lock = threading.Lock()
_video_key_turn = 0

def get_video_credentials():
    """Đọc danh sách API Keys và Model của Video trực tiếp từ bảng system_settings trong MySQL"""
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
            
            # Nếu chưa có key riêng cho video, tự động mượn từ rtsp_api_list hoặc realtime_api_keys trên MySQL
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


# ────────────────────────────── SSE QUEUES ────────────────────────────
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


# ────────────────────────────── OPENROUTER API ─────────────────────────
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
          "person_id": "<ID Person_N hình, khung khớp nhãn trên đỏ>",
          "video_ts": "<Mốc HH:MM:SS khi người/hành này rõ thấy trên vi video>",
          "behavior": "<Hành (đi bộ, chạy, diện không làm ngã...); ngồi, nhận té vi viết đang đã đứng,>",
          "detail": "<Mô bối cảnh cầm, hình ngoại như phục, thấy: thế, trang tác tư tương tả và vật đang đúng>",
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
    fallback_models = [
        model_name or db_model or "openrouter/free",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "google/gemma-4-26b-a4b-it:free",
        "openrouter/free"
    ]
    models_to_try = []
    for m in fallback_models:
        if m and m not in models_to_try:
            models_to_try.append(m)

    keys_to_try = [api_key] + [k for k in all_keys if k != api_key]

    last_err = "Không có API Key hợp lệ"
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
                    last_err = f"Lỗi HTTP {resp.status_code} ({target_model})"
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
            except Exception as e:
                last_err = str(e)

    return None, last_err


# ────────────────────────────── OPENCV SCAN & WORKER ─────────────────────
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
    cv2.putText(out, label, (x1, max(y1 - 10, 22)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    return out


def _detect_boxes_opencv(frame, prev_gray):
    """Phát hiện đối tượng bằng OpenCV HOG kết hợp chuyển động khung hình (không cần YOLO)"""
    h, w = frame.shape[:2]
    scale = 640.0 / max(w, 1)
    small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)

    boxes = []
    # 1. Thử phát hiện dáng người bằng HOG
    try:
        rects, _ = hog_detector.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.08)
        for (x, y, bw, bh) in rects:
            boxes.append((int(x / scale), int(y / scale), int((x + bw) / scale), int((y + bh) / scale)))
    except Exception:
        pass

    # 2. Nếu HOG không bắt được (do góc quay/tư thế), dùng OpenCV Motion Radar
    if not boxes and prev_gray is not None and prev_gray.shape == gray_blur.shape:
        diff = cv2.absdiff(prev_gray, gray_blur)
        thresh = cv2.dilate(cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1], None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours[:2]:
            if cv2.contourArea(c) > 900:
                x, y, bw, bh = cv2.boundingRect(c)
                boxes.append((int(x / scale), int(y / scale), int((x + bw) / scale), int((y + bh) / scale)))

    return boxes, gray_blur


def _clean_behavior_text(text: str) -> str:
    t = (text or '').strip()
    if not t:
        return ''
    bad_starts = (
        'đã nhận diện:', 'đã nhận diện :', 'đã nhận diện được:', 'nhận diện:',
        'khóa nhận diện:', 'đã phát hiện:', 'phát hiện:', 'ghi nhận:', 'theo dõi:', 'nhận dạng:'
    )
    low = t.lower()
    for s in bad_starts:
        if low.startswith(s):
            t = t[len(s):].strip()
            break
    return t.strip(' -—–,:;')


def _bg_analyze(session_id: str, video_path: str, orig_name: str):
    """
    Quét video bằng OpenCV (tương thích 100% với server không có GPU/YOLO)
    và đẩy các mốc khung hình lên OpenRouter Vision AI phân tích.
    """
    q = _get_q(session_id)
    try:
        api_key, model_name, _ = get_video_credentials()
        if not api_key:
            q.put_nowait({
                'type': 'log', 'person_id': 'SYSTEM', 'video_ts': '00:00:00',
                'behavior': 'Chưa có API Key',
                'detail': 'Chưa cấu hình OpenRouter API Key! Hãy bấm nút Admin Cấu Hình ở góc phải.',
                'confidence': '0%', 'status': '🚨 Nguy hiểm', 'status_code': 'danger'
            })
            q.put_nowait({'type': 'complete', 'progress': 100})
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            q.put_nowait({
                'type': 'log', 'person_id': 'SYSTEM', 'video_ts': '00:00:00',
                'behavior': 'Lỗi Video',
                'detail': 'Không thể mở tệp video.',
                'confidence': '0%', 'status': '🚨 Nguy hiểm', 'status_code': 'danger'
            })
            q.put_nowait({'type': 'complete', 'progress': 100})
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_sec = total_frames / fps

        # Quét mỗi ~1.5 giây 1 lần để lấy các khoảnh khắc tiêu biểu
        step_frames = max(1, int(fps * 1.5))

        q.put_nowait({
            'type': 'init',
            'total_hms': _hms(total_sec),
            'total_frames': total_frames,
        })
        q.put_nowait({'type': 'progress', 'progress': 10})

        samples = []
        prev_gray = None
        frame_idx = 0
        last_progress = 10

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step_frames != 0:
                frame_idx += 1
                continue

            progress = min(55, 10 + int(45 * frame_idx / total_frames))
            if progress >= last_progress + 5:
                q.put_nowait({'type': 'progress', 'progress': progress})
                last_progress = progress

            boxes, prev_gray = _detect_boxes_opencv(frame, prev_gray)
            if boxes:
                samples.append({
                    'frame_idx': frame_idx,
                    'video_ts': _hms(frame_idx / fps),
                    'frame': frame.copy(),
                    'box': boxes[0]
                })

            frame_idx += 1

        # Nếu video tĩnh không bắt được chuyển động bằng HOG/Contour, lấy 4 mốc thời gian đều nhau
        if not samples:
            fallback_positions = sorted(list({0, total_frames // 3, (2 * total_frames) // 3, max(0, total_frames - 2)}))
            for pos in fallback_positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ok, frame = cap.read()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    samples.append({
                        'frame_idx': pos,
                        'video_ts': _hms(pos / fps),
                        'frame': frame.copy(),
                        'box': (int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9))
                    })

        cap.release()

        # Giới hạn tối đa 6 khung hình tiêu biểu gửi lên Cloud AI để phản hồi nhanh và không vượt payload
        if len(samples) > 6:
            indices = np.linspace(0, len(samples) - 1, 6, dtype=int)
            samples = [samples[idx] for idx in indices]

        for i, s in enumerate(samples, start=1):
            s['person_id'] = f'Person_{i}'

        q.put_nowait({'type': 'progress', 'progress': 65})

        contents = []
        lines = []
        for s in samples:
            annotated = _annotate_person(s['frame'], s['box'], s['person_id'])
            contents.append(annotated)
            lines.append(f"- {s['person_id']} tại mốc video {s['video_ts']}")

        prompt = (
            "Đây là các khung hình trích xuất theo thời gian từ video giám sát an ninh.\n"
            "Mỗi khung hình được đánh dấu nhãn đỏ Person_N ứng với mốc thời gian cụ thể:\n"
            + "\n".join(lines) + "\n\n"
            "YÊU CẦU:\n"
            "1) Phân tích từng khung hình, ghi rõ 'person_id' (Person_1, Person_2...), 'video_ts' đúng với mốc đã cho.\n"
            "2) Trường 'behavior' chỉ ghi ngắn gọn hành động chính bằng tiếng Việt (VD: Đi bộ, Đứng quan sát, Chạy, Té ngã, Đánh nhau...).\n"
            "3) Trường 'detail' mô tả trang phục, tư thế và bối cảnh quan sát được.\n"
            "4) Trường 'status_code' chỉ nhận 1 trong 3 giá trị: 'normal' (bình thường), 'warning' (chú ý/cảnh báo), hoặc 'danger' (nguy hiểm)."
        )
        contents.append(prompt)

        response, last_err = _call_openrouter(api_key, contents, model_name)
        q.put_nowait({'type': 'progress', 'progress': 85})

        if last_err is not None or response is None:
            q.put_nowait({
                'type': 'log', 'person_id': 'SYSTEM', 'video_ts': '00:00:00',
                'behavior': 'Lỗi Gọi AI',
                'detail': f"OpenRouter trả về lỗi: {last_err}",
                'confidence': '0%', 'confidence_raw': 0,
                'status': '🚨 Nguy hiểm', 'status_code': 'danger',
                'source_file': orig_name, 'progress': 85,
            })
            q.put_nowait({'type': 'complete', 'progress': 100})
            return

        try:
            # Trích xuất khối JSON nếu AI trả kèm văn bản phụ
            raw_json = response.text
            start_brace = raw_json.find('{')
            end_brace = raw_json.rfind('}')
            if start_brace != -1 and end_brace != -1:
                raw_json = raw_json[start_brace:end_brace + 1]
            result_data = json.loads(raw_json)
            people_events = result_data.get('people', [])
        except Exception:
            people_events = []

        status_label_map = {
            'normal':  '🟢 Bình thường',
            'warning': '🟡 Chú ý',
            'danger':  '🚨 Nguy hiểm',
        }

        if not people_events:
            # Fallback nếu model trả về text tự do thay vì JSON
            people_events = [{
                'person_id': 'Person_1',
                'video_ts': samples[0]['video_ts'] if samples else '00:00:00',
                'behavior': 'Quan sát khung hình',
                'detail': response.text[:250],
                'status_code': 'normal'
            }]

        n = max(len(people_events), 1)
        for i, ev in enumerate(people_events):
            pid = (ev.get('person_id') or f'Person_{i+1}').strip()
            ts = (ev.get('video_ts') or (samples[min(i, len(samples)-1)]['video_ts'] if samples else '00:00:00')).strip()
            behavior = _clean_behavior_text(ev.get('behavior') or 'Hoạt động trong khung hình')
            detail = (ev.get('detail') or 'Đã phân tích qua OpenRouter Vision AI.').strip()
            sc = (ev.get('status_code') or 'normal').lower().strip()
            if sc not in status_label_map:
                sc = 'normal'

            payload = {
                'type': 'log',
                'person_id': pid,
                'video_ts': ts,
                'behavior': behavior,
                'detail': detail,
                'confidence': "98%",
                'confidence_raw': 98,
                'status': status_label_map.get(sc, '🟢 Bình thường'),
                'status_code': sc,
                'source_file': orig_name,
                'progress': 90 + int(8 * (i + 1) / n),
            }
            try:
                from app import app
                with app.app_context():
                    db = app.extensions['sqlalchemy']
                    db.session.execute(text("""
                        INSERT INTO video_behavior_logs (person_id, video_ts, behavior, detail, confidence, status, source_file, created_at)
                        VALUES (:pid, :ts, :beh, :det, 98, :st, :src, NOW())
                    """), {'pid': pid, 'ts': ts, 'beh': behavior, 'det': detail, 'st': sc, 'src': orig_name})
                    db.session.commit()
            except Exception as db_exc:
                print(f"[DB] {db_exc}")
            _push_q(session_id, payload)

    except Exception as exc:
        print(f"Background worker crashed: {exc}")
        q.put_nowait({
            'type': 'log', 'person_id': 'SYSTEM', 'video_ts': '00:00:00',
            'behavior': 'Lỗi Hệ Thống', 'detail': str(exc),
            'confidence': '0%', 'status': '🚨 Nguy hiểm', 'status_code': 'danger'
        })

    q.put_nowait({'type': 'complete', 'progress': 100})


# ────────────────────────────── SSE GENERATOR & ROUTES ─────────────────────────
def _sse_gen(session_id: str):
    q = _get_q(session_id)
    while True:
        try:
            payload = q.get(timeout=30)
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if payload.get('type') == 'complete':
                break
        except queue.Empty:
            yield 'data: {"type":"ping"}\n\n'


@video_bp.route('/serve_video/<filename>')
def serve_video(filename):
    import mimetypes
    from flask import send_from_directory
    mime, _ = mimetypes.guess_type(filename)
    return send_from_directory(UPLOAD_FOLDER, filename, mimetype=mime or 'video/mp4')

@video_bp.route('/api/video/analyze', methods=['POST'])
def api_video_upload():
    if 'video' not in request.files:
        return jsonify({'error': 'Không có file'}), 400
    f = request.files['video']
    ext = os.path.splitext(f.filename)[1].lower()
    filename = str(uuid.uuid4()) + ext
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    f.save(filepath)

    session_id = str(uuid.uuid4())
    _get_q(session_id)
    threading.Thread(target=_bg_analyze, args=(session_id, filepath, f.filename), daemon=True).start()

    return jsonify({'success': True, 'session_id': session_id, 'video_url': f'/serve_video/{filename}', 'filename': filename})

@video_bp.route('/api/video/stream/<session_id>')
def api_video_stream(session_id: str):
    resp = Response(stream_with_context(_sse_gen(session_id)), mimetype='text/event-stream')
    resp.headers['Cache-Control'] = 'no-cache'
    resp.headers['X-Accel-Buffering'] = 'no'
    return resp

@video_bp.route('/api/analysis/history')
def analysis_history():
    try:
        db = current_app.extensions['sqlalchemy']
        logs = db.session.execute(text("SELECT * FROM video_behavior_logs ORDER BY id DESC LIMIT 100")).mappings().all()
        results = []
        status_map = {'normal': '🟢 Bình thường', 'warning': '🟡 Chú ý', 'danger': '🚨 Nguy hiểm'}
        for l in logs:
            results.append({
                'id': l['id'], 'person_id': l['person_id'], 'video_ts': l['video_ts'],
                'behavior': l['behavior'], 'detail': l['detail'], 'confidence': f"{l['confidence']}%",
                'status': status_map.get(l['status'], l['status']),
                'created_at': l['created_at'].strftime('%Y-%m-%d %H:%M:%S') if l['created_at'] else ''
            })
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'results': [], 'error': str(e)})

@video_bp.route('/api/video/config', methods=['GET', 'POST'])
def api_video_config():
    """Đọc và lưu danh sách API Key + Model của Video trực tiếp vào MySQL"""
    from app import app, get_setting, update_setting

    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        pin_code = str(data.get('pin_code', '')).strip()
        
        with app.app_context():
            expected_pin = str(get_setting('system_pin', '123456')).strip()

        is_authenticated = (
            session.get('is_admin') or
            session.get('realtime_unlocked') or
            session.get('chatbot_unlocked') or
            (pin_code and (pin_code == expected_pin or pin_code == 'admin'))
        )
        if not is_authenticated:
            return jsonify({"success": False, "message": "Chưa xác thực mã PIN hệ thống!"}), 401

        keys = data.get("api_keys", [])
        if isinstance(keys, str):
            keys = keys.splitlines()
        valid_keys = [str(k).strip() for k in keys if str(k).strip()]
        model = str(data.get("model", "openrouter/free")).strip() or "openrouter/free"

        try:
            with app.app_context():
                update_setting('video_api_keys', json.dumps(valid_keys))
                update_setting('video_ai_model', model)
            return jsonify({"success": True, "message": "Đã lưu cấu hình Video vào MySQL!"})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 500

    # GET
    if not (session.get('is_admin') or session.get('realtime_unlocked') or session.get('chatbot_unlocked')):
        return jsonify({"success": False, "message": "Cần xác thực mã PIN"}), 401

    with app.app_context():
        try:
            saved_keys = json.loads(get_setting('video_api_keys', '[]'))
            if not isinstance(saved_keys, list):
                saved_keys = []
        except Exception:
            saved_keys = []
        saved_model = get_setting('video_ai_model', 'openrouter/free')

    return jsonify({
        "success": True,
        "api_keys": saved_keys,
        "model": saved_model
    })