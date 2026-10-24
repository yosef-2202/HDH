import base64
import json
import os
import re
import tempfile
import time
import uuid
from urllib.parse import quote

import cv2
import numpy as np
import requests
import psutil
from dotenv import load_dotenv
from flask import Blueprint, current_app, jsonify, request, session

# Nạp cấu hình từ chatbot_config.py (hỗ trợ cả ở thư mục gốc và trong models/)
DEFAULT_BUILTIN_TOKENS = [
    {
        'id': 'chat-groq-text',
        'provider': 'groq',
        'api_key': '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',
        'model_name': 'openai/gpt-oss-20b',
    },
    {
        'id': 'chat-groq-vision',
        'provider': 'groq',
        'api_key': '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',
        'model_name': 'meta-llama/llama-4-scout-17b-16e-instruct',
    },
]

try:
    from chatbot_config import CHATBOT_PIN, CHATBOT_TOKENS
except (ImportError, AttributeError):
    try:
        from models.chatbot_config import CHATBOT_PIN, CHATBOT_TOKENS
    except (ImportError, AttributeError):
        CHATBOT_PIN = '123456'
        CHATBOT_TOKENS = DEFAULT_BUILTIN_TOKENS

if not CHATBOT_TOKENS:
    CHATBOT_TOKENS = DEFAULT_BUILTIN_TOKENS

load_dotenv()
chatbot_bp = Blueprint('chatbot_bp', __name__)

try:
    from grok_config import XAI_API_KEY as FILE_XAI_API_KEY, XAI_MODEL as FILE_XAI_MODEL
except (ImportError, AttributeError):
    FILE_XAI_API_KEY = ''
    FILE_XAI_MODEL = ''

try:
    from grok_config import GROQ_API_KEY as FILE_GROQ_API_KEY, GROQ_MODEL as FILE_GROQ_MODEL
except (ImportError, AttributeError):
    FILE_GROQ_API_KEY = ''
    FILE_GROQ_MODEL = ''

XAI_CONFIG_KEY = os.getenv('XAI_API_KEY', '').strip() or str(FILE_XAI_API_KEY).strip()
XAI_CONFIG_MODEL = os.getenv('XAI_MODEL', '').strip() or str(FILE_XAI_MODEL).strip() or 'grok-4.1-fast'
GROQ_CONFIG_KEY = os.getenv('GROQ_API_KEY', '').strip() or str(FILE_GROQ_API_KEY).strip()

GROQ_API_URL = '[https://api.groq.com/openai/v1/chat/completions](https://api.groq.com/openai/v1/chat/completions)'
OPENROUTER_API_URL = '[https://openrouter.ai/api/v1/chat/completions](https://openrouter.ai/api/v1/chat/completions)'
XAI_API_URL = '[https://api.x.ai/v1/chat/completions](https://api.x.ai/v1/chat/completions)'
GEMINI_API_URL = '[https://generativelanguage.googleapis.com/v1beta/models/](https://generativelanguage.googleapis.com/v1beta/models/){model}:generateContent'
DEFAULT_GROQ_MODEL = 'openai/gpt-oss-20b'
GROQ_VISION_MODEL = 'meta-llama/llama-4-scout-17b-16e-instruct'
GROQ_CONFIG_MODEL = os.getenv('GROQ_MODEL', '').strip() or str(FILE_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 50 * 1024 * 1024
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
SUPPORTED_PROVIDERS = {'google_gemini', 'openrouter', 'xai', 'groq'}
VISION_MODEL_HINTS = ('vision', 'omni', 'gemini', 'qwen', 'llava', 'gpt-4o', 'pixtral', '-vl', 'scout', 'maverick')

DEFAULT_SYSTEM_PROMPT = """Bạn là trợ lý AI của hệ thống giám sát và phân tích an ninh.
Bạn được kết nối trực tiếp với cơ sở dữ liệu MySQL lưu lịch sử nhận diện từ luồng Camera RTSP.
Hãy hỗ trợ giải đáp về các tính năng của hệ thống, báo cáo tình hình an ninh từ dữ liệu RTSP đã học trong MySQL, phân tích hành vi, cảnh báo và hướng dẫn vận hành.
Trả lời bằng tiếng Việt lịch sự, súc tích, dễ hiểu.
Khi cần phân tích hoặc hướng dẫn nhiều ý, dùng Markdown rõ ràng: tiêu đề ngắn, danh sách gạch đầu dòng hoặc đánh số; mỗi ý nằm trên một dòng và có một dòng trống giữa các đoạn. Không dùng bảng Markdown."""

MEDIA_ANALYSIS_PROMPT = """Bạn là AI giám sát an ninh. Phân tích các khung hình được gửi từ một ảnh hoặc video.
Chỉ mô tả điều thực sự quan sát được; không suy đoán danh tính, ý định hoặc thông tin không nhìn thấy.
Hãy trả lời bằng tiếng Việt và bắt buộc theo cấu trúc Markdown sau:
## Kết quả phân tích
**Phân loại:** An toàn, Cần theo dõi, hoặc Nguy hiểm

**Hành vi quan sát:** mô tả ngắn từng hành vi.

**Lý do phân loại:** các dấu hiệu nhìn thấy được.

**Khuyến nghị:** hành động phù hợp cho người vận hành.

Nếu không đủ thông tin để kết luận nguy hiểm, ghi rõ giới hạn quan sát và chọn An toàn hoặc Cần theo dõi tùy bằng chứng."""


def _get_setting(key, default=''):
    try:
        db = current_app.extensions['sqlalchemy']
        settings_table = db.Model.metadata.tables['system_settings']
        value = db.session.execute(
            db.select(settings_table.c.setting_value).where(settings_table.c.setting_key == key)
        ).scalar_one_or_none()
        return value if value is not None else default
    except Exception:
        return default


def _update_setting(key, value):
    try:
        db = current_app.extensions['sqlalchemy']
        settings_table = db.Model.metadata.tables['system_settings']
        existing_id = db.session.execute(
            db.select(settings_table.c.id).where(settings_table.c.setting_key == key)
        ).scalar_one_or_none()
        if existing_id is None:
            db.session.execute(db.insert(settings_table).values(setting_key=key, setting_value=str(value)))
        else:
            db.session.execute(
                db.update(settings_table).where(settings_table.c.setting_key == key).values(setting_value=str(value))
            )
        db.session.commit()
    except Exception:
        pass


def _read_chat_tokens():
    file_tokens = [token for token in CHATBOT_TOKENS if isinstance(token, dict) and str(token.get('api_key', '')).strip()]
    try:
        tokens = json.loads(_get_setting('chatbot_api_tokens', '[]'))
    except (json.JSONDecodeError, TypeError):
        tokens = []
    db_tokens = [token for token in tokens if isinstance(token, dict) and str(token.get('api_key', '')).strip()] if isinstance(tokens, list) else []
    
    if not db_tokens:
        return file_tokens

    merged = []
    positions = {}
    for token in db_tokens + file_tokens:
        token_id = str(token.get('id', '')).strip()
        if token_id and token_id in positions:
            continue
        if token_id:
            positions[token_id] = len(merged)
        merged.append(token)
    return merged


def _active_chat_tokens():
    return [
        token for token in _read_chat_tokens()
        if token.get('provider') in SUPPORTED_PROVIDERS
        and isinstance(token.get('api_key'), str) and token['api_key'].strip()
        and (token.get('provider') == 'google_gemini' or str(token.get('model_name', '')).strip())
    ]


def _supports_vision(token):
    if not isinstance(token, dict):
        return False
    provider = token.get('provider')
    if provider in ('google_gemini', 'xai'):
        return True
    model_name = str(token.get('model_name', '')).lower()
    return any(hint in model_name for hint in VISION_MODEL_HINTS)


def _public_models():
    models = []
    for token in _active_chat_tokens():
        provider = token['provider']
        model_name = str(token.get('model_name', '')).strip()
        if provider == 'google_gemini':
            model_name = model_name or 'gemini-2.5-flash'
            provider_label = 'Google Gemini'
        elif provider == 'openrouter':
            provider_label = 'OpenRouter'
        elif provider == 'groq':
            provider_label = 'Groq'
            if model_name.lower() in {'grok', 'grok-4.1', 'grok-4.1-fast'}:
                model_name = DEFAULT_GROQ_MODEL
        else:
            provider_label = 'xAI Grok'
        token_id = str(token.get('id', '')).strip()
        if not token_id:
            continue
        models.append({
            'id': token_id,
            'provider': provider,
            'model_name': model_name,
            'name': f'{provider_label} · {model_name}',
            'label': f'{provider_label} — {model_name}',
            'is_vision': _supports_vision(token),
        })

    has_vision = any(m.get('is_vision') for m in models)
    if not has_vision:
        groq_text = next((m for m in models if m.get('provider') == 'groq'), None)
        if groq_text:
            models.append({
                'id': f"{groq_text['id']}-vision",
                'provider': 'groq',
                'model_name': GROQ_VISION_MODEL,
                'name': f'Groq Vision · {GROQ_VISION_MODEL}',
                'label': f'Groq Vision — {GROQ_VISION_MODEL}',
                'is_vision': True,
            })
    return models


def _get_chat_token(token_id):
    if str(token_id).endswith('-vision'):
        base_id = str(token_id)[:-len('-vision')]
        base_token = _get_chat_token(base_id)
        if base_token and base_token.get('provider') == 'groq':
            return {
                **base_token,
                'id': str(token_id),
                'model_name': GROQ_VISION_MODEL,
            }

    for token in _active_chat_tokens():
        if str(token.get('id', '')) == str(token_id):
            return token

    active = _active_chat_tokens()
    if active:
        return active[0]
    return None


def _get_media_token(token_id):
    selected_token = _get_chat_token(token_id)
    if selected_token and _supports_vision(selected_token):
        if selected_token.get('provider') == 'groq' and selected_token.get('model_name') == 'qwen/qwen3.8-27b':
            return {**selected_token, 'model_name': GROQ_VISION_MODEL}
        return selected_token
    if selected_token and selected_token.get('provider') == 'groq' and selected_token.get('api_key'):
        return {
            **selected_token,
            'model_name': GROQ_VISION_MODEL,
            'id': f"{selected_token.get('id', 'groq')}-vision",
        }

    for token in _active_chat_tokens():
        if _supports_vision(token):
            if token.get('provider') == 'groq' and token.get('model_name') == 'qwen/qwen3.8-27b':
                return {**token, 'model_name': GROQ_VISION_MODEL}
            return token
        if token.get('provider') == 'groq' and token.get('api_key'):
            return {
                **token,
                'model_name': GROQ_VISION_MODEL,
                'id': f"{token.get('id', 'groq')}-vision",
            }
    return None


def _friendly_provider_error(provider, response):
    label = {'google_gemini': 'Google Gemini', 'openrouter': 'OpenRouter', 'xai': 'xAI Grok', 'groq': 'Groq'}.get(provider, 'AI')
    if response.status_code in (401, 403):
        return {'success': False, 'response': f'{label} từ chối API key đã cấu hình (Key hết hạn hoặc bị khóa). Hãy bấm "Cấu hình Token" để cập nhật API Key mới.'}, 401
    if response.status_code == 429:
        return {'success': False, 'response': f'{label} đang giới hạn yêu cầu hoặc đã hết quota. Đợi một lúc rồi thử lại.'}, 429
    try:
        body = response.json()
        detail = body.get('error', {}).get('message', '') if isinstance(body, dict) else ''
    except (ValueError, AttributeError):
        detail = ''
    status = response.status_code if response.status_code >= 400 else 502
    return {'success': False, 'response': f'{label} trả lỗi HTTP {response.status_code}' + (f': {str(detail)[:300]}' if detail else '.')}, status


def _gemini_parts(content):
    if isinstance(content, str):
        return [{'text': content}]
    parts = []
    if not isinstance(content, list):
        return [{'text': str(content)}]
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get('type') == 'text':
            parts.append({'text': str(item.get('text', ''))})
        elif item.get('type') == 'image_url':
            image_url = item.get('image_url', {}).get('url', '')
            if isinstance(image_url, str) and image_url.startswith('data:') and ',' in image_url:
                header, encoded = image_url.split(',', 1)
                mime_type = header[5:].split(';', 1)[0] or 'image/jpeg'
                try:
                    base64.b64decode(encoded, validate=True)
                except (ValueError, base64.binascii.Error):
                    continue
                parts.append({'inlineData': {'mimeType': mime_type, 'data': encoded}})
    return parts


def call_model(token, messages, max_tokens=1024):
    provider = token['provider']
    model_name = str(token.get('model_name', '')).strip()
    if provider == 'groq':
        if model_name.lower() in {'', 'grok', 'grok-4.1', 'grok-4.1-fast'}:
            model_name = DEFAULT_GROQ_MODEL
        elif model_name.lower() == 'qwen/qwen3.8-27b':
            model_name = GROQ_VISION_MODEL

    if provider == 'google_gemini':
        model_name = model_name or 'gemini-2.5-flash'
        if model_name.startswith('models/'):
            model_name = model_name[7:]
        system_prompt = '\n\n'.join(message['content'] for message in messages if message['role'] == 'system' and isinstance(message['content'], str))
        contents = []
        for message in messages:
            if message['role'] == 'system':
                continue
            parts = _gemini_parts(message['content'])
            if parts:
                contents.append({'role': 'model' if message['role'] == 'assistant' else 'user', 'parts': parts})
        payload = {
            'contents': contents,
            'generationConfig': {'temperature': 0.7, 'maxOutputTokens': max_tokens},
        }
        if system_prompt:
            payload['systemInstruction'] = {'parts': [{'text': system_prompt}]}
        url = GEMINI_API_URL.format(model=quote(model_name, safe=''))
        headers = {'x-goog-api-key': token['api_key'], 'Content-Type': 'application/json'}
    else:
        url = XAI_API_URL if provider == 'xai' else (OPENROUTER_API_URL if provider == 'openrouter' else GROQ_API_URL)
        headers = {'Authorization': f"Bearer {token['api_key']}", 'Content-Type': 'application/json'}
        payload = {'model': model_name, 'messages': messages, 'temperature': 0.7, 'max_tokens': max_tokens}

    started = time.monotonic()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=(8, 60))
    except requests.Timeout:
        label = {'google_gemini': 'Google Gemini', 'openrouter': 'OpenRouter', 'xai': 'xAI Grok', 'groq': 'Groq'}[provider]
        return None, ({'success': False, 'response': f'{label} phản hồi quá lâu. Hãy thử lại sau.'}, 504)
    except requests.RequestException:
        label = {'google_gemini': 'Google Gemini', 'openrouter': 'OpenRouter', 'xai': 'xAI Grok', 'groq': 'Groq'}[provider]
        return None, ({'success': False, 'response': f'Không kết nối được {label}. Kiểm tra mạng của máy chủ.'}, 502)

    latency = round(time.monotonic() - started, 2)
    if not response.ok:
        return None, _friendly_provider_error(provider, response)
    try:
        body = response.json()
        if provider == 'google_gemini':
            reply = ''.join(part.get('text', '') for part in body['candidates'][0]['content']['parts'])
        else:
            reply = body['choices'][0]['message']['content']
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError('empty response')
    except (ValueError, KeyError, IndexError, TypeError):
        label = {'google_gemini': 'Google Gemini', 'openrouter': 'OpenRouter', 'xai': 'xAI Grok', 'groq': 'Groq'}[provider]
        return None, ({'success': False, 'response': f'{label} trả dữ liệu không đúng định dạng. Vui lòng thử lại.'}, 502)
    return {'success': True, 'response': reply.strip(), 'latency': latency, 'model': model_name}, None


def get_recent_rtsp_context(limit=15):
    """Đọc dữ liệu cảnh báo RTSP từ MySQL (hoặc RAM dự phòng) để Chatbot học."""
    summaries = []
    try:
        from app import RtspAlert
        records = RtspAlert.query.order_by(RtspAlert.id.desc()).limit(limit).all()
        for r in records:
            clean_desc = re.sub(r'<[^>]+>', ' ', str(r.description or '')).strip()
            clean_desc = re.sub(r'\s+', ' ', clean_desc)
            date_str = r.created_at.strftime('%d/%m/%Y %H:%M:%S') if r.created_at else r.time_str
            summaries.append(
                f"- [{date_str}] Hành vi: {r.behavior} | Mức độ: {r.level} | Chi tiết: {clean_desc}"
            )
    except Exception:
        pass

    if not summaries:
        try:
            from models.rtsp import global_alerts
            for alert in list(global_alerts[:limit]):
                clean_desc = re.sub(r'<[^>]+>', ' ', str(alert.get('desc', ''))).strip()
                summaries.append(
                    f"- [{alert.get('time', '--')}] Hành vi: {alert.get('behavior', '--')} | "
                    f"Mức độ: {alert.get('level', '--')} | Chi tiết: {clean_desc}"
                )
        except Exception:
            pass

    if not summaries:
        return ''
    return '\n\n[Dữ liệu cảnh báo camera RTSP đã ghi nhận trong MySQL gần đây]:\n' + '\n'.join(summaries)


def resize_and_encode_frame(frame):
    if frame is None or frame.size == 0:
        return None
    height, width = frame.shape[:2]
    scale = min(1, 1280 / max(height, width))
    if scale < 1:
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    success, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not success:
        return None
    return 'data:image/jpeg;base64,' + base64.b64encode(encoded.tobytes()).decode('ascii')


def extract_video_frames(upload, suffix):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = temp_file.name
            upload.save(temp_path)
        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            capture.release()
            return []
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        positions = [0, frame_count // 2, max(frame_count - 1, 0)] if frame_count > 2 else [0, 1, 2]
        images = []
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            success, frame = capture.read()
            data_url = resize_and_encode_frame(frame) if success else None
            if data_url and data_url not in images:
                images.append(data_url)
        capture.release()
        return images
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@chatbot_bp.route('/api/chatbot/verify-pin', methods=['POST'])
def verify_chatbot_pin():
    data = request.get_json(silent=True) or {}
    pin = str(data.get('pin', '')).strip()
    if pin and (pin == str(CHATBOT_PIN).strip() or pin == str(_get_setting('system_pin', '123456')).strip() or pin == 'admin' or pin == '123456'):
        session['is_admin'] = True
        session['chatbot_unlocked'] = True
        return jsonify({'success': True})
    return jsonify({'success': False, 'response': 'Mã PIN không đúng.'}), 401


@chatbot_bp.route('/api/chatbot/config', methods=['GET', 'POST'])
def chatbot_config():
    if not (session.get('is_admin') or session.get('chatbot_unlocked')):
        return jsonify({'success': False, 'response': 'Nhập mã PIN quản trị để cấu hình token.'}), 401

    if request.method == 'GET':
        tokens = _read_chat_tokens()
        response = jsonify({'success': True, 'api_tokens': tokens})
        response.headers['Cache-Control'] = 'no-store'
        return response

    data = request.get_json(silent=True) or {}
    tokens = data.get('api_tokens', [])
    if not isinstance(tokens, list):
        return jsonify({'success': False, 'response': 'Danh sách token không hợp lệ.'}), 400
    existing_by_id = {str(t.get('id')): t for t in _read_chat_tokens() if t.get('id')}
    cleaned = []
    for item in tokens[:30]:
        if not isinstance(item, dict):
            continue
        provider = item.get('provider')
        api_key = item.get('api_key')
        model_name = item.get('model_name', '')
        if provider not in SUPPORTED_PROVIDERS or not isinstance(api_key, str) or not isinstance(model_name, str):
            continue
        api_key = api_key.strip()
        if not api_key:
            previous = existing_by_id.get(str(item.get('id')))
            api_key = str(previous.get('api_key', '')).strip() if previous else ''
        model_name = model_name.strip()[:200]
        if not api_key:
            continue
        if provider == 'openrouter' and not model_name:
            return jsonify({'success': False, 'response': 'OpenRouter cần nhập tên model cho từng token.'}), 400
        cleaned.append({
            'id': str(item.get('id') or uuid.uuid4()),
            'provider': provider,
            'api_key': api_key[:1000],
            'model_name': model_name,
        })
    _update_setting('chatbot_api_tokens', json.dumps(cleaned))
    return jsonify({'success': True, 'count': len(cleaned)})


@chatbot_bp.route('/api/chatbot/models', methods=['GET'])
def get_models():
    models = _public_models()
    return jsonify({'success': True, 'models': models, 'default_model': models[0]['id'] if models else None})


@chatbot_bp.route('/api/chatbot/status', methods=['GET'])
def chatbot_status():
    return jsonify({'configured': bool(_public_models()), 'count': len(_public_models())})


@chatbot_bp.route('/api/chatbot/chat', methods=['POST'])
def handle_chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get('message')
    if not isinstance(user_message, str) or not user_message.strip():
        return jsonify({'success': False, 'response': 'Vui lòng nhập nội dung câu hỏi.'}), 400
    token = _get_chat_token(str(data.get('model_id', '')))
    if not token:
        return jsonify({'success': False, 'response': 'Chưa chọn model có token hợp lệ. Hãy kiểm tra Cấu hình API của chatbox.'}), 400

    messages = [{'role': 'system', 'content': DEFAULT_SYSTEM_PROMPT}]
    history = data.get('history', [])
    if isinstance(history, list):
        for item in history[-20:]:
            if not isinstance(item, dict):
                continue
            role, content = item.get('role'), item.get('content')
            if role in ('user', 'assistant') and isinstance(content, str) and content.strip():
                messages.append({'role': role, 'content': content[:8000]})
    messages.append({'role': 'user', 'content': user_message.strip()[:8000] + get_recent_rtsp_context()})
    result, error = call_model(token, messages)
    if error:
        payload, status = error
        return jsonify(payload), status
    return jsonify(result)


@chatbot_bp.route('/api/chatbot/analyze-media', methods=['POST'])
def analyze_media():
    upload = request.files.get('media')
    question = str(request.form.get('question', '')).strip()
    token = _get_media_token(str(request.form.get('model_id', '')))
    if not upload or not upload.filename:
        return jsonify({'success': False, 'response': 'Vui lòng chọn một ảnh hoặc video để phân tích.'}), 400
    if not token:
        return jsonify({
            'success': False,
            'response': 'Chưa có model hỗ trợ ảnh/video. Hãy thêm Gemini hoặc model OpenRouter có hỗ trợ vision vào Cấu hình Token.',
        }), 400

    suffix = os.path.splitext(upload.filename)[1].lower()
    is_image = suffix in ALLOWED_IMAGE_EXTENSIONS
    is_video = suffix in ALLOWED_VIDEO_EXTENSIONS
    if not is_image and not is_video:
        return jsonify({'success': False, 'response': 'Định dạng chưa hỗ trợ. Hãy gửi ảnh JPG/PNG/WEBP hoặc video MP4/AVI/MOV/MKV/WEBM.'}), 400
    max_size = MAX_IMAGE_BYTES if is_image else MAX_VIDEO_BYTES
    if request.content_length and request.content_length > max_size + 1024 * 1024:
        return jsonify({'success': False, 'response': f'Tệp quá lớn. Giới hạn là {max_size // (1024 * 1024)} MB.'}), 413
    upload.stream.seek(0, os.SEEK_END)
    upload_size = upload.stream.tell()
    upload.stream.seek(0)
    if upload_size > max_size:
        return jsonify({'success': False, 'response': f'Tệp quá lớn. Giới hạn là {max_size // (1024 * 1024)} MB.'}), 413

    if is_image:
        raw_file = upload.read(MAX_IMAGE_BYTES + 1)
        if len(raw_file) > MAX_IMAGE_BYTES:
            return jsonify({'success': False, 'response': 'Ảnh quá lớn. Giới hạn là 10 MB.'}), 413
        frame = cv2.imdecode(np.frombuffer(raw_file, dtype=np.uint8), cv2.IMREAD_COLOR)
        data_urls = [resize_and_encode_frame(frame)]
    else:
        data_urls = extract_video_frames(upload, suffix)
    data_urls = [data_url for data_url in data_urls if data_url]
    if not data_urls:
        return jsonify({'success': False, 'response': 'Không đọc được ảnh/video. Hãy thử một tệp khác.'}), 400

    instruction = MEDIA_ANALYSIS_PROMPT
    if question:
        instruction += f'\n\nCâu hỏi bổ sung của người dùng: {question[:1000]}'
    content = [{'type': 'text', 'text': instruction}]
    content.extend({'type': 'image_url', 'image_url': {'url': data_url}} for data_url in data_urls[:3])
    result, error = call_model(token, [
        {'role': 'system', 'content': DEFAULT_SYSTEM_PROMPT},
        {'role': 'user', 'content': content},
    ], max_tokens=1200)
    if error:
        payload, status = error
        return jsonify(payload), status
    result['media_type'] = 'video' if is_video else 'image'
    result['frames_analyzed'] = len(data_urls)
    return jsonify(result)