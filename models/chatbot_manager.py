import base64
import json
import os
import tempfile
import time
import uuid
from urllib.parse import quote

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from flask import Blueprint, current_app, jsonify, request, session

load_dotenv()
chatbot_bp = Blueprint('chatbot_bp', __name__)

# Có thể cấu hình Grok bằng biến môi trường hoặc file riêng `grok_config.py`.
# File riêng không được commit lên GitHub; dùng cho máy chủ chỉ có SFTP.
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

GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions'
OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/chat/completions'
XAI_API_URL = 'https://api.x.ai/v1/chat/completions'
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
DEFAULT_GROQ_MODEL = 'openai/gpt-oss-20b'
GROQ_VISION_MODEL = 'qwen/qwen3.8-27b'
GROQ_CONFIG_MODEL = os.getenv('GROQ_MODEL', '').strip() or str(FILE_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 50 * 1024 * 1024
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
SUPPORTED_PROVIDERS = {'google_gemini', 'openrouter', 'xai', 'groq'}
VISION_MODEL_HINTS = ('vision', 'omni', 'gemini', 'qwen', 'llava', 'gpt-4o', 'pixtral', '-vl')

DEFAULT_SYSTEM_PROMPT = """Bạn là trợ lý AI của hệ thống giám sát và phân tích an ninh.
Hãy hỗ trợ giải đáp về các tính năng của hệ thống, RTSP/video, camera, nhận diện tư thế,
cảnh báo và cách vận hành. Trả lời bằng tiếng Việt lịch sự, súc tích, dễ hiểu.
Không khẳng định hệ thống đã thực hiện chức năng nếu thông tin hiện có không chứng minh điều đó.
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
    db = current_app.extensions['sqlalchemy']
    settings_table = db.Model.metadata.tables['system_settings']
    value = db.session.execute(
        db.select(settings_table.c.setting_value).where(settings_table.c.setting_key == key)
    ).scalar_one_or_none()
    return value if value is not None else default


def _update_setting(key, value):
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


def _read_chat_tokens():
    try:
        tokens = json.loads(_get_setting('chatbot_api_tokens', '[]'))
    except (json.JSONDecodeError, TypeError):
        return []
    return [token for token in tokens if isinstance(token, dict)] if isinstance(tokens, list) else []


def _active_chat_tokens():
    return [
        token for token in _read_chat_tokens()
        if token.get('provider') in SUPPORTED_PROVIDERS
        and isinstance(token.get('api_key'), str) and token['api_key'].strip()
        and (token.get('provider') == 'google_gemini' or str(token.get('model_name', '')).strip())
    ]


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
        # Hiện thêm model Vision của Groq dùng chung API key gsk_...
        if provider == 'groq' and not _supports_vision(token):
            models.append({
                'id': f'{token_id}-vision',
                'provider': 'groq',
                'model_name': GROQ_VISION_MODEL,
                'name': f'Groq Vision · {GROQ_VISION_MODEL}',
                'label': f'Groq Vision — {GROQ_VISION_MODEL}',
                'is_vision': True,
            })

    # Nếu chưa có token riêng cho chatbox, nạp tự động từ api_tokens của hệ thống (Settings/RTSP)
    if not models:
        try:
            sys_tokens = json.loads(_get_setting('api_tokens', '[]'))
            for idx, token in enumerate(sys_tokens, start=1):
                if not isinstance(token, dict):
                    continue
                api_key = token.get('api_key', '').strip()
                if not api_key:
                    continue
                provider = token.get('provider', 'google_gemini')
                model_name = token.get('model_name', '').strip()
                if provider == 'google_gemini':
                    model_name = model_name or 'gemini-2.5-flash'
                    provider_label = 'Google Gemini (Hệ thống)'
                else:
                    provider_label = 'OpenRouter (Hệ thống)'
                token_id = f'sys-token-{idx}'
                is_vis = provider == 'google_gemini' or any(hint in model_name.lower() for hint in VISION_MODEL_HINTS)
                models.append({
                    'id': token_id,
                    'provider': provider,
                    'model_name': model_name,
                    'name': f'{provider_label} · {model_name}',
                    'label': f'{provider_label} — {model_name}',
                    'is_vision': is_vis,
                })
        except Exception:
            pass

    # Cho phép máy chủ chỉ dùng file cấu hình riêng, không cần mở modal token.
    if XAI_CONFIG_KEY and not any(item.get('id') == 'env-xai' for item in models):
        models.append({
            'id': 'env-xai', 'provider': 'xai', 'model_name': XAI_CONFIG_MODEL,
            'name': f'xAI Grok · {XAI_CONFIG_MODEL}',
            'label': f'xAI Grok — {XAI_CONFIG_MODEL}', 'is_vision': True,
        })

    # Fallback Groq dùng được với key gsk_... trong file riêng hoặc biến môi trường.
    if GROQ_CONFIG_KEY and not any(item.get('id') == 'file-groq' for item in models):
        models.extend([
            {'id': 'file-groq', 'provider': 'groq', 'model_name': GROQ_CONFIG_MODEL,
             'name': f'Groq · {GROQ_CONFIG_MODEL}', 'label': f'Groq — {GROQ_CONFIG_MODEL}', 'is_vision': False},
            {'id': 'legacy-groq-20b', 'provider': 'groq', 'model_name': DEFAULT_GROQ_MODEL,
             'name': 'Groq · GPT-OSS 20B', 'label': 'Groq — GPT-OSS 20B', 'is_vision': False},
            {'id': 'legacy-groq-120b', 'provider': 'groq', 'model_name': 'openai/gpt-oss-120b',
             'name': 'Groq · GPT-OSS 120B', 'label': 'Groq — GPT-OSS 120B', 'is_vision': False},
            {'id': 'legacy-groq-vision', 'provider': 'groq', 'model_name': GROQ_VISION_MODEL,
             'name': 'Groq · Qwen 3.8 27B (ảnh/video)', 'label': 'Groq — Qwen 3.8 27B', 'is_vision': True},
        ])
    return models


def _get_chat_token(token_id):
    # Model Vision Groq dùng lại token gốc, chỉ thay model name.
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

    # Settings/RTSP tokens are also exposed as safe model choices when no
    # dedicated chatbox list exists. Resolve those choices on the server.
    if str(token_id).startswith('sys-token-'):
        try:
            index = int(str(token_id).rsplit('-', 1)[-1]) - 1
            system_tokens = json.loads(_get_setting('api_tokens', '[]'))
            token = system_tokens[index]
            if isinstance(token, dict) and token.get('api_key', '').strip():
                provider = token.get('provider', 'google_gemini')
                if provider in SUPPORTED_PROVIDERS:
                    return {
                        'id': str(token_id),
                        'provider': provider,
                        'api_key': token['api_key'].strip(),
                        'model_name': str(token.get('model_name', '')).strip(),
                    }
        except (ValueError, IndexError, TypeError, json.JSONDecodeError):
            pass

    # Tìm trong danh sách token hệ thống (sys-token-X)
    if str(token_id).startswith('sys-token-'):
        try:
            sys_tokens = json.loads(_get_setting('api_tokens', '[]'))
            valid_tokens = [t for t in sys_tokens if isinstance(t, dict) and t.get('api_key', '').strip()]
            idx = int(token_id.split('-')[-1]) - 1
            if 0 <= idx < len(valid_tokens):
                t = valid_tokens[idx]
                return {
                    'id': token_id,
                    'provider': t.get('provider', 'google_gemini'),
                    'api_key': t['api_key'].strip(),
                    'model_name': t.get('model_name', '').strip()
                }
        except Exception:
            pass

    if token_id in ('legacy-groq-20b', 'legacy-groq-120b', 'legacy-groq-vision'):
        api_key = GROQ_CONFIG_KEY
        if api_key and not _active_chat_tokens():
            model_by_id = {
                'legacy-groq-20b': DEFAULT_GROQ_MODEL,
                'legacy-groq-120b': 'openai/gpt-oss-120b',
                'legacy-groq-vision': GROQ_VISION_MODEL,
            }
            return {'id': token_id, 'provider': 'groq', 'api_key': api_key, 'model_name': model_by_id[token_id]}

    if token_id == 'file-groq' and GROQ_CONFIG_KEY:
        return {
            'id': 'file-groq', 'provider': 'groq',
            'api_key': GROQ_CONFIG_KEY, 'model_name': GROQ_CONFIG_MODEL,
        }

    if token_id == 'env-xai' and XAI_CONFIG_KEY:
        return {
            'id': 'env-xai', 'provider': 'xai',
            'api_key': XAI_CONFIG_KEY, 'model_name': XAI_CONFIG_MODEL,
        }
    return None


def _supports_vision(token):
    """Return whether the configured provider/model can receive image frames."""
    if not isinstance(token, dict):
        return False
    provider = token.get('provider')
    if provider in ('google_gemini', 'xai'):
        return True
    if provider == 'groq':
        return token.get('model_name') == GROQ_VISION_MODEL
    model_name = str(token.get('model_name', '')).lower()
    return any(hint in model_name for hint in VISION_MODEL_HINTS)


def _get_media_token(token_id):
    """Tự động chuyển đổi hoặc lựa chọn API hỗ trợ Vision để phân tích hình ảnh và video."""
    # 1. Nếu token người dùng đang chọn đã hỗ trợ Vision -> ưu tiên dùng
    selected_token = _get_chat_token(token_id)
    if selected_token and _supports_vision(selected_token):
        return selected_token
    # Groq dùng cùng API key cho model chữ và model Vision.
    if selected_token and selected_token.get('provider') == 'groq' and selected_token.get('api_key'):
        return {
            **selected_token,
            'model_name': GROQ_VISION_MODEL,
            'id': f"{selected_token.get('id', 'groq')}-vision",
        }

    # 2. Tự động chuyển đổi sang token có Vision trong chatbox (xAI Grok hoặc Gemini)
    active_tokens = _active_chat_tokens()
    for token in active_tokens:
        if token.get('provider') == 'xai':
            return token
    for token in active_tokens:
        if token.get('provider') == 'google_gemini':
            return token
    for token in active_tokens:
        if token.get('provider') == 'groq' and token.get('api_key'):
            return {
                **token,
                'model_name': GROQ_VISION_MODEL,
                'id': f"{token.get('id', 'groq')}-vision",
            }
    for token in active_tokens:
        if _supports_vision(token):
            return token

    # 3. Tự động kiểm tra và dùng token Vision từ cấu hình hệ thống (Settings)
    try:
        sys_tokens = json.loads(_get_setting('api_tokens', '[]'))
        for t in sys_tokens:
            if isinstance(t, dict) and t.get('api_key', '').strip():
                provider = t.get('provider', 'google_gemini')
                model_name = t.get('model_name', '').strip()
                if provider == 'google_gemini':
                    return {
                        'id': 'sys-media-gemini',
                        'provider': 'google_gemini',
                        'api_key': t['api_key'].strip(),
                        'model_name': model_name or 'gemini-2.5-flash'
                    }
                elif any(h in model_name.lower() for h in VISION_MODEL_HINTS):
                    return {
                        'id': 'sys-media-openrouter',
                        'provider': 'openrouter',
                        'api_key': t['api_key'].strip(),
                        'model_name': model_name
                    }
    except Exception:
        pass

    # 4. Fallback cuối: Groq Qwen Vision (nếu có key trong env)
    return _get_chat_token('legacy-groq-vision')


def _friendly_provider_error(provider, response):
    label = {'google_gemini': 'Google Gemini', 'openrouter': 'OpenRouter', 'xai': 'xAI Grok', 'groq': 'Groq'}.get(provider, 'AI')
    if response.status_code in (401, 403):
        return {'success': False, 'response': f'{label} từ chối API key đã cấu hình. Hãy kiểm tra token trong Cấu hình API của chatbox.'}, 401
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
    # Một số cấu hình cũ đã lưu tên xAI "grok" dưới provider Groq.
    # Tự chuyển sang model Groq hợp lệ để không trả HTTP 404.
    if provider == 'groq' and model_name.lower() in {'', 'grok', 'grok-4.1', 'grok-4.1-fast'}:
        model_name = DEFAULT_GROQ_MODEL
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


def get_recent_rtsp_context():
    """Expose only recent alert summaries; never camera URLs or API keys."""
    try:
        from models.rtsp import global_alerts
        alerts = list(global_alerts[:5])
    except Exception:
        return ''
    if not alerts:
        return ''
    summaries = []
    for alert in alerts:
        description = str(alert.get('desc', '')).replace('<br>', '; ').replace('<', '').replace('>', '')
        summaries.append(
            f"- {alert.get('time', 'Không rõ thời gian')}: {alert.get('behavior', 'Không rõ hành vi')}; "
            f"mức {alert.get('level', 'Chưa phân loại')}; chi tiết: {description}"
        )
    return '\n\nDữ liệu cảnh báo camera realtime gần đây:\n' + '\n'.join(summaries)


def resize_and_encode_frame(frame):
    """Create a compact JPEG data URL for supported vision providers."""
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
    """Read three representative frames without retaining the uploaded video on disk."""
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


@chatbot_bp.route('/api/chatbot/config', methods=['GET', 'POST'])
def chatbot_config():
    if not session.get('is_admin'):
        return jsonify({'success': False, 'response': 'Nhập mã PIN quản trị để cấu hình token.'}), 401

    if request.method == 'GET':
        try:
            tokens = json.loads(_get_setting('chatbot_api_tokens', '[]'))
        except (json.JSONDecodeError, TypeError):
            tokens = []
        if not isinstance(tokens, list):
            tokens = []
        response = jsonify({'success': True, 'api_tokens': tokens})
        response.headers['Cache-Control'] = 'no-store'
        return response

    data = request.get_json(silent=True) or {}
    tokens = data.get('api_tokens', [])
    if not isinstance(tokens, list):
        return jsonify({'success': False, 'response': 'Danh sách token không hợp lệ.'}), 400
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
