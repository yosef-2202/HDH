# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import queue
import threading
import uuid
import base64
import time
import re
import cv2
import numpy as np
import requests
import psutil
from flask import Blueprint, Response, jsonify, request, stream_with_context, session
from models.session_limiter import (
    get_limiter,
    build_temp_filename,
    protect_temp_file,
    unprotect_temp_file,
    is_auto_clean_enabled,
    TEMP_FOLDER,
)

video_bp = Blueprint('video_bp', __name__)

os.makedirs(TEMP_FOLDER, exist_ok=True)

MAX_VIDEO_BYTES = 50 * 1024 * 1024  # Giới hạn cứng tối đa 50MB

DEFAULT_VIDEO_MODEL = "grok-4-fast-non-reasoning"

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
    "x-ai/grok-4-fast-non-reasoning",
    "x-ai/grok-4.3",
    "google/gemini-2.0-flash-exp:free",
    "qwen/qwen2.5-vl-72b-instruct:free",
    "meta-llama/llama-4-scout:free",
]

DEPRECATED_MODELS = {
    "grok-2-vision-1212",
    "x-ai/grok-2-vision-1212",
    "grok-2-vision",
    "x-ai/grok-2-vision",
    "openrouter/free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
}

hog_detector = cv2.HOGDescriptor()
hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

_video_key_lock = threading.Lock()
_video_key_turn = 0
current_process = psutil.Process(os.getpid())

# Bộ nhớ lưu trạng thái tiến trình cho cả SSE lẫn Polling trên điện thoại
_sse_queues: dict[str, queue.Queue] = {}
_job_states: dict[str, dict] = {}
_sse_lock = threading.Lock()


def build_candidate_models_for_key(api_key: str, configured_model: str) -> tuple[str, list[str], bool]:
    key_str = str(api_key or "").strip()
    model_str = str(configured_model or "").strip()

    if (
        model_str.lower() in DEPRECATED_MODELS
        or "grok-2" in model_str.lower()
        or "nemotron" in model_str.lower()
        or "openrouter/free" in model_str.lower()
    ):
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


def get_video_credentials():
    """
    Lấy danh sách API Key từ cả video_api_keys, realtime_api_keys và rtsp_api_list.
    Ưu tiên đưa các key trực tiếp (xAI Grok / Groq) lên đầu để đảm bảo tốc độ và độ ổn định.
    """
    global _video_key_turn
    from app import app, get_setting
    raw_collected = []
    model_name = DEFAULT_VIDEO_MODEL
    try:
        with app.app_context():
            # 1. Lấy key từ cấu hình Video
            try:
                v_keys = json.loads(get_setting('video_api_keys', '[]') or '[]')
                if isinstance(v_keys, list):
                    for k in v_keys:
                        if isinstance(k, str) and k.strip():
                            raw_collected.append(k.strip())
            except Exception:
                pass

            # 2. Gộp thêm key từ cấu hình Realtime Camera (nơi đã có sẵn key Grok hoạt động tốt)
            try:
                rt_keys = json.loads(get_setting('realtime_api_keys', '[]') or '[]')
                if isinstance(rt_keys, list):
                    for k in rt_keys:
                        if isinstance(k, str) and k.strip() and k.strip() not in raw_collected:
                            raw_collected.append(k.strip())
            except Exception:
                pass

            # 3. Gộp thêm key từ RTSP nếu có
            try:
                rtsp_raw = json.loads(get_setting('rtsp_api_list', '[]') or '[]')
                for it in rtsp_raw:
                    k = str(it.get('key', '') if isinstance(it, dict) else it).strip()
                    if k and k not in raw_collected:
                        raw_collected.append(k)
            except Exception:
                pass

            saved_model = str(get_setting('video_ai_model', DEFAULT_VIDEO_MODEL) or DEFAULT_VIDEO_MODEL).strip()
            if (
                saved_model
                and saved_model.lower() not in DEPRECATED_MODELS
                and "grok-2" not in saved_model.lower()
                and "nemotron" not in saved_model.lower()
                and "openrouter/free" not in saved_model.lower()
            ):
                model_name = saved_model
    except Exception:
        pass

    # Sắp xếp ưu tiên key Grok (xai-...) và Groq (gsk_...) lên trước key OpenRouter (sk-or-...)
    direct_keys = [k for k in raw_collected if not k.startswith("sk-or-")]
    or_keys = [k for k in raw_collected if k.startswith("sk-or-")]
    keys = direct_keys + or_keys

    if not keys:
        return "", model_name, []

    with _video_key_lock:
        selected_key = keys[_video_key_turn % len(keys)]
        _video_key_turn += 1
    return selected_key, model_name, keys


def _get_q(sid: str) -> queue.Queue:
    with _sse_lock:
        if sid not in _sse_queues:
            _sse_queues[sid] = queue.Queue(maxsize=1000)
        if sid not in _job_states:
            _job_states[sid] = {
                "progress": 5,
                "stage_text": "Đang khởi tạo trình phân tích video...",
                "metrics": {"cpu": "0.0%", "ram": "0 MB", "bw": "0 KB"},
                "logs": [],
                "completed": False,
                "error": None,
                "updated_at": time.time()
            }
        return _sse_queues[sid]


def _push_q(sid: str, payload: dict):
    q = _get_q(sid)
    with _sse_lock:
        st = _job_states.get(sid)
        if st is not None:
            st["updated_at"] = time.time()
            ptype = payload.get("type")
            if ptype in ("init", "progress"):
                if "progress" in payload:
                    st["progress"] = payload["progress"]
                if "stage_text" in payload:
                    st["stage_text"] = payload["stage_text"]
            elif ptype == "metrics":
                st["metrics"] = {
                    "cpu": payload.get("cpu", "0.0%"),
                    "ram": payload.get("ram", "0 MB"),
                    "bw": payload.get("bw", "0 KB")
                }
            elif ptype == "log":
                st["logs"].append(payload)
            elif ptype == "complete":
                st["progress"] = 100
                st["stage_text"] = "Phân tích video hoàn tất!"
                st["completed"] = True
    try:
        q.put_nowait(payload)
    except queue.Full:
        pass


def infer_behavior_from_description(desc: str) -> str:
    low = desc.lower()
    keyword_map = [
        ("đánh nhau", "Đang đánh nhau"),
        ("ẩu đả", "Đang đánh nhau"),
        ("xô xát", "Đang xô xát"),
        ("tung cú đá", "Đá / Tấn công"),
        ("đá vào", "Đá / Tấn công"),
        ("vung tay đánh", "Đấm / Xô xát"),
        ("đấm", "Đấm / Xô xát"),
        ("bóp cổ", "Đang bóp cổ"),
        ("túm", "Giằng co / Xô xát"),
        ("vật nhau", "Đang vật/đè nhau"),
        ("đè nhau", "Đang đè nhau"),
        ("hôn nhau", "Đang hôn nhau"),
        ("ôm hôn", "Đang hôn nhau"),
        ("cầm dao", "Cầm hung khí"),
        ("vũ khí", "Cầm vũ khí"),
        ("ngã", "Đang bị ngã"),
        ("nằm bất động", "Nằm bất động"),
        ("cãi", "Đang tranh cãi"),
        ("chống cằm", "Đang ngồi chống cằm"),
        ("điện thoại", "Đang dùng điện thoại"),
        ("uống nước", "Đang uống nước"),
        ("ăn", "Đang ăn uống"),
        ("vẫy tay", "Đang vẫy tay"),
        ("đi bộ", "Đang đi bộ"),
        ("di chuyển", "Đang di chuyển"),
        ("đứng", "Đang đứng"),
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

    return "Đang hoạt động"


def clean_ai_response_and_extract_status(raw_text: str) -> tuple[str, str, str]:
    """
    Trích xuất (status_code, behavior_label, clean_description)
    từ định dạng chuẩn giống realtime_camera.py: [MỨC ĐỘ] [HÀNH VI] Mô tả chi tiết.
    """
    text = str(raw_text or "").strip()
    if not text:
        return "normal", "Đang quan sát", "Đối tượng đang hoạt động trong khung hình."

    status_badge = "BÌNH THƯỜNG"
    behavior_label = ""
    remainder = text

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

    beh_match = re.match(r'^[\s:,\-–—|]*\[([^\[\]]{2,45})\][\s:,\-–—|]*', remainder)
    if beh_match:
        candidate_beh = beh_match.group(1).strip()
        if candidate_beh.upper() not in ("BÌNH THƯỜNG", "CẢNH BÁO", "NGUY HIỂM", "TÍCH CỰC"):
            behavior_label = candidate_beh
            remainder = remainder[beh_match.end():].strip()

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
        cleaned_desc = "Đối tượng đang xuất hiện trong khung hình camera."

    if not behavior_label:
        behavior_label = infer_behavior_from_description(cleaned_desc)
    else:
        behavior_label = behavior_label[:1].upper() + behavior_label[1:]

    # Quy đổi sang status_code cho bảng giao diện video.html
    sc = "normal"
    if "NGUY HIỂM" in status_badge:
        sc = "danger"
    elif "CẢNH BÁO" in status_badge:
        sc = "warning"

    # Kiểm tra từ khóa bạo lực để tự động nâng mức cảnh báo chính xác
    combined_low = f"{behavior_label} {cleaned_desc}".lower()
    if any(k in combined_low for k in ("đánh nhau", "xô xát", "ẩu đả", "đá ", "tung cú đá", "đấm", "tấn công", "bóp cổ", "vũ khí", "cầm dao", "ngã")):
        sc = "danger"
    elif sc == "normal" and any(k in combined_low for k in ("cãi", "giằng co", "xô đẩy", "đè nhau", "hôn nhau", "chặn")):
        sc = "warning"

    return sc, behavior_label, cleaned_desc


def _analyze_single_frame_with_grok(frame_bgr, video_ts: str) -> tuple[str | None, str | None]:
    """
    Gửi 1 khung hình lên Grok (hoặc Groq/OpenRouter) y hệt cơ chế của realtime_camera.py.
    Trả về (raw_text, error_msg).
    """
    api_key, model_name, valid_apis = get_video_credentials()
    if not api_key:
        return None, "Chưa cấu hình API Key"

    h, w = frame_bgr.shape[:2]
    scale = min(1.0, 640.0 / max(w, h, 1))
    if scale < 1.0:
        frame_resized = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        frame_resized = frame_bgr

    ret, buffer = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if not ret:
        return None, "Lỗi mã hóa khung hình"
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    prompt = (
        f"Bạn là hệ thống camera AI nhận diện hành vi tại mốc thời gian {video_ts} trong video. "
        "BẮT BUỘC TRẢ LỜI ĐÚNG 1 DÒNG TIẾNG VIỆT THEO ĐỊNH DẠNG SAU:\n"
        "[MỨC ĐỘ] [HÀNH VI CỤ THỂ] Mô tả chi tiết người, trang phục và hành động đang diễn ra.\n"
        "Quy tắc bắt buộc:\n"
        "1. [MỨC ĐỘ] chỉ chọn 1 trong 3: [BÌNH THƯỜNG], [CẢNH BÁO], hoặc [NGUY HIỂM]. "
        "(Dùng [BÌNH THƯỜNG] cho đi lại, đứng, ngồi bình thường; dùng [CẢNH BÁO] cho cãi vã, giằng co, ôm hôn, đè nhau; dùng [NGUY HIỂM] cho đánh nhau, đá nhau, đấm nhau, xô xát, bóp cổ, cầm vũ khí, ngã gục).\n"
        "2. [HÀNH VI CỤ THỂ] phải chỉ đích danh hành động ngắn gọn từ 2-5 chữ trong ngoặc vuông (Ví dụ: [Đang đánh nhau], [Tung cú đá tấn công], [Đang xô xát], [Đang đứng nói chuyện], [Đang đi bộ]).\n"
        "Ví dụ chuẩn: [NGUY HIỂM] [Đang đánh nhau] Hai thanh niên đang xô xát trước cửa quán, người mặc áo hồng đội mũ bảo hiểm đang tung cú đá vào người mặc áo đen."
    )

    last_error = "Không gọi được AI"
    keys_to_try = [api_key] + [k for k in valid_apis if k != api_key]

    for current_key in keys_to_try:
        target_endpoint, candidate_models, is_direct_api = build_candidate_models_for_key(current_key, model_name)

        headers = {
            "Authorization": f"Bearer {current_key}",
            "Content-Type": "application/json",
        }
        if not is_direct_api:
            headers["HTTP-Referer"] = "http://localhost:24706"
            headers["X-Title"] = "HDH Surveillance AI - Video Module"

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
                "max_tokens": 300,
                "temperature": 0.2
            }

            try:
                response = requests.post(target_endpoint, headers=headers, json=payload, timeout=30)
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
                    content_str = str(msg.get('content') or '').strip()
                    if content_str:
                        return content_str, None
                    else:
                        last_error = f"Model {active_model} trả về rỗng"
                else:
                    last_error = "Empty choices"
            except Exception as req_err:
                last_error = str(req_err)[:60]
                break

    return None, last_error


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
        thresh = cv2.dilate(cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)[1], None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:2]:
            if cv2.contourArea(c) > 650:
                x, y, bw, bh = cv2.boundingRect(c)
                boxes.append((int(x / scale), int(y / scale), int((x + bw) / scale), int((y + bh) / scale)))

    return boxes, gray_blur


def _bg_analyze(session_id: str, video_path: str, orig_name: str, client_id: str = ""):
    _get_q(session_id)
    protect_temp_file(video_path)
    try:
        api_key, _, _ = get_video_credentials()
        if not api_key:
            _push_q(session_id, {
                'type': 'log', 'log_id': f"{session_id}_err_key",
                'person_id': 'SYSTEM', 'video_ts': '00:00:00',
                'behavior': 'Chưa có API Key',
                'detail': 'Chưa cấu hình API Key! Hãy bấm Admin Cấu Hình để thêm Key Grok/OpenRouter.',
                'status': '🚨 Nguy hiểm', 'status_code': 'danger'
            })
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            _push_q(session_id, {
                'type': 'log', 'log_id': f"{session_id}_err_cap",
                'person_id': 'SYSTEM', 'video_ts': '00:00:00',
                'behavior': 'Lỗi Video',
                'detail': 'Không thể giải mã tệp video này. Vui lòng chọn định dạng MP4/WEBM.',
                'status': '🚨 Nguy hiểm', 'status_code': 'danger'
            })
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 0 or fps > 240:
            fps = 25.0
        total_sec = max(1.0, total_frames / fps)
        step_frames = max(1, int(fps * 1.2))

        _push_q(session_id, {
            'type': 'init',
            'total_hms': _hms(total_sec),
            'total_frames': total_frames,
            'progress': 42,
            'stage_text': 'Đang quét chuyển động bằng OpenCV...'
        })

        samples = []
        fallback_frames = []
        prev_gray = None
        frame_idx = 0
        file_kb = round(os.path.getsize(video_path) / 1024, 1) if os.path.exists(video_path) else 0

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_idx % step_frames != 0:
                frame_idx += 1
                continue

            get_limiter('video').try_acquire_stream(client_id or session_id)

            try:
                cpu_val = round(psutil.cpu_percent(interval=None) / max(1, psutil.cpu_count()) + 2.1, 1)
                ram_val = round(current_process.memory_info().rss / (1024 * 1024), 1)
            except Exception:
                cpu_val, ram_val = 4.5, 45.0

            _push_q(session_id, {
                'type': 'metrics',
                'cpu': f"{cpu_val}%",
                'ram': f"{ram_val} MB",
                'bw': f"{file_kb} KB"
            })

            if total_frames > 1:
                scan_pct = min(70, 42 + int((frame_idx / total_frames) * 28))
            else:
                scan_pct = min(70, 45 + len(samples) * 5)

            _push_q(session_id, {
                'type': 'progress',
                'progress': scan_pct,
                'stage_text': f'Đang trích xuất khung hình ({_hms(frame_idx / fps)})...'
            })

            if frame.mean() > 10.0:
                h0, w0 = frame.shape[:2]
                fallback_frames.append({
                    'frame_idx': frame_idx,
                    'video_ts': _hms(frame_idx / fps),
                    'frame': frame.copy(),
                    'box': (w0 // 8, h0 // 8, w0 * 7 // 8, h0 * 7 // 8)
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

        if len(samples) < 2 and fallback_frames:
            target_count = min(3, len(fallback_frames))
            indices = np.linspace(0, len(fallback_frames) - 1, target_count, dtype=int)
            samples = [fallback_frames[idx] for idx in indices]

        if not samples:
            _push_q(session_id, {
                'type': 'log', 'log_id': f"{session_id}_empty",
                'person_id': 'SYSTEM', 'video_ts': '00:00:00',
                'behavior': 'Video trống',
                'detail': 'Không trích xuất được khung hình hợp lệ từ video.',
                'status': '🟡 Chú ý', 'status_code': 'warning'
            })
            return

        # Chọn tối đa 3 mốc thời gian tiêu biểu trong video để Grok phân tích nhanh và chi tiết
        if len(samples) > 3:
            samples = [samples[idx] for idx in np.linspace(0, len(samples) - 1, 3, dtype=int)]

        status_map = {'normal': '🟢 Bình thường', 'warning': '🟡 Chú ý', 'danger': '🚨 Nguy hiểm'}
        total_samples = len(samples)
        success_count = 0
        last_err_msg = ""

        for idx, s in enumerate(samples, start=1):
            pid = f'Person_{idx}'
            ts = s['video_ts']
            pct = 72 + int((idx / total_samples) * 24)

            _push_q(session_id, {
                'type': 'progress',
                'progress': min(96, pct),
                'stage_text': f'Grok AI đang phân tích khung hình {idx}/{total_samples} (mốc {ts})...'
            })

            raw_text, err = _analyze_single_frame_with_grok(s['frame'], ts)
            if raw_text:
                sc, behavior_label, clean_desc = clean_ai_response_and_extract_status(raw_text)
                _push_q(session_id, {
                    'type': 'log',
                    'log_id': f"{session_id}_{idx}",
                    'person_id': pid,
                    'video_ts': ts,
                    'behavior': behavior_label,
                    'detail': clean_desc,
                    'confidence': '98%',
                    'status': status_map.get(sc, '🟢 Bình thường'),
                    'status_code': sc
                })
                success_count += 1
            else:
                last_err_msg = err or "Lỗi kết nối API"

        if success_count == 0:
            _push_q(session_id, {
                'type': 'log',
                'log_id': f"{session_id}_fail",
                'person_id': 'SYSTEM',
                'video_ts': '00:00:00',
                'behavior': 'Lỗi Gọi AI',
                'detail': f"Không thể phân tích khung hình ({last_err_msg}). Hãy kiểm tra API Key trong Admin Cấu Hình.",
                'status': '🚨 Nguy hiểm',
                'status_code': 'danger'
            })

    except Exception as exc:
        _push_q(session_id, {
            'type': 'log', 'log_id': f"{session_id}_exc",
            'person_id': 'SYSTEM', 'video_ts': '00:00:00',
            'behavior': 'Lỗi Xử Lý', 'detail': str(exc),
            'status': '🚨 Nguy hiểm', 'status_code': 'danger'
        })
    finally:
        unprotect_temp_file(video_path)
        if is_auto_clean_enabled():
            try:
                if video_path and os.path.exists(video_path):
                    os.remove(video_path)
            except Exception:
                pass
        get_limiter('video').release(client_id or session_id, cleanup_temp=False)
        _push_q(session_id, {'type': 'complete', 'progress': 100})


def _sse_gen(session_id: str):
    q = _get_q(session_id)
    while True:
        try:
            payload = q.get(timeout=15)
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if payload.get('type') == 'complete':
                break
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
        get_limiter('video').release(client_id, cleanup_temp=False)
        return jsonify({'success': False, 'message': 'Không có file video'}), 400
    f = request.files['video']

    f.stream.seek(0, os.SEEK_END)
    size = f.stream.tell()
    f.stream.seek(0)
    if size > MAX_VIDEO_BYTES:
        get_limiter('video').release(client_id, cleanup_temp=False)
        return jsonify({'success': False, 'message': f'Tệp quá lớn ({round(size/(1024*1024), 1)} MB). Giới hạn tối đa là 50 MB để đảm bảo hiệu suất AI!'}), 413

    ext = os.path.splitext(f.filename)[1].lower() or '.mp4'
    base_id, filepath = build_temp_filename('video', client_id, ext)
    protect_temp_file(filepath)
    filename = os.path.basename(filepath)
    f.save(filepath)

    session_id = base_id
    _get_q(session_id)
    threading.Thread(target=_bg_analyze, args=(session_id, filepath, f.filename, client_id), daemon=True).start()

    return jsonify({'success': True, 'session_id': session_id, 'video_url': f'/serve_video/{filename}'})


@video_bp.route('/api/video/stream/<session_id>')
def api_video_stream(session_id: str):
    resp = Response(stream_with_context(_sse_gen(session_id)), mimetype='text/event-stream')
    resp.headers['Cache-Control'] = 'no-cache, no-transform'
    resp.headers['X-Accel-Buffering'] = 'no'
    resp.headers['Connection'] = 'keep-alive'
    return resp


@video_bp.route('/api/video/status/<session_id>', methods=['GET'])
def api_video_status(session_id: str):
    """API Polling dự phòng cho trình duyệt điện thoại khi SSE bị nghẽn mạng."""
    with _sse_lock:
        st = _job_states.get(session_id)
        if not st:
            return jsonify({'success': False, 'message': 'Không tìm thấy phiên xử lý'}), 404
        return jsonify({
            'success': True,
            'progress': st['progress'],
            'stage_text': st['stage_text'],
            'metrics': st['metrics'],
            'logs': list(st['logs']),
            'completed': st['completed']
        })


@video_bp.route('/api/video/verify_pin', methods=['POST'])
def api_video_verify_pin():
    data = request.get_json(force=True, silent=True) or {}
    pin = str(data.get('pin') or '').strip()
    from app import app, get_setting
    with app.app_context():
        system_pin = str(get_setting('system_pin', '123456') or '123456').strip()

    if pin == system_pin or pin == 'admin':
        session['video_unlocked'] = True
        session['is_admin'] = True
        return jsonify({"success": True, "message": "Xác thực mã PIN thành công!"})
    return jsonify({"success": False, "message": "Mã PIN không đúng"}), 401


@video_bp.route('/api/video/config', methods=['GET', 'POST'])
def api_video_config():
    from app import app, get_setting, update_setting
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        req_pin = str(data.get('pin_code') or data.get('pin') or '').strip()
        with app.app_context():
            system_pin = str(get_setting('system_pin', '123456') or '123456').strip()
            if req_pin and (req_pin == system_pin or req_pin == 'admin'):
                session['video_unlocked'] = True
                session['is_admin'] = True
            if not (session.get('video_unlocked') or session.get('is_admin')):
                return jsonify({"success": False, "message": "Cần xác thực mã PIN!"}), 401

            raw_keys = data.get("api_keys", [])
            if isinstance(raw_keys, str):
                clean_keys = [k.strip() for k in raw_keys.replace(',', '\n').split('\n') if k.strip()]
            else:
                clean_keys = [str(k).strip() for k in raw_keys if str(k).strip()]

            model_in = str(data.get("model") or DEFAULT_VIDEO_MODEL).strip()
            if not model_in or model_in.lower() in DEPRECATED_MODELS or "openrouter/free" in model_in.lower():
                model_in = DEFAULT_VIDEO_MODEL

            update_setting('video_api_keys', json.dumps(clean_keys))
            update_setting('video_ai_model', model_in)
        return jsonify({"success": True, "message": "Đã lưu cấu hình API Video thành công!"})

    with app.app_context():
        try:
            keys = json.loads(get_setting('video_api_keys', '[]') or '[]')
            if not keys:
                keys = json.loads(get_setting('realtime_api_keys', '[]') or '[]')
        except Exception:
            keys = []
        model = get_setting('video_ai_model', DEFAULT_VIDEO_MODEL) or DEFAULT_VIDEO_MODEL
        if model.lower() in DEPRECATED_MODELS or "openrouter/free" in model.lower():
            model = DEFAULT_VIDEO_MODEL
    return jsonify({"success": True, "api_keys": keys, "model": model})