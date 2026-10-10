from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from urllib.parse import quote_plus
from werkzeug.middleware.proxy_fix import ProxyFix
import json
import psutil
import os

# Import module RTSP & Realtime Camera
from models.rtsp import rtsp_bp
from models.realtime_camera import realtime_bp

app = Flask(__name__)
app.secret_key = 'hdh_secret_key_2026'

# Khai báo ProxyFix giúp ứng dụng nhận diện chuẩn xác IP thực
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# --- CẤU HÌNH MYSQL ---
db_user = 'u50283_95Wuon3aKO'
db_pass = quote_plus('e!nfV.oVXQ6Qcwso9P@WCkz5')
db_host = '91.99.159.222:3306'
db_name = 's50283_HDH'

app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{db_user}:{db_pass}@{db_host}/{db_name}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_size': 10, 'pool_recycle': 3600}

db = SQLAlchemy(app)

class SystemSettings(db.Model):
    __tablename__ = 'system_settings'
    id = db.Column(db.Integer, primary_key=True)
    setting_key = db.Column(db.String(50), unique=True, nullable=False)
    setting_value = db.Column(db.Text, nullable=True)

with app.app_context():
    db.create_all()
    default_settings = {
        'system_pin': '123456',
        'rtsp_api_list': '[]',
        'rtsp_concurrent_jobs': '2',
        'enable_video': 'true',
        'enable_rtsp': 'true',
        'enable_chatbot': 'true',
        'enable_image': 'true',
        'enable_realtime': 'true',
        'chatbot_api_tokens': '[]'  # Thêm field cấu hình Token Chatbot
    }
    for key, value in default_settings.items():
        if not SystemSettings.query.filter_by(setting_key=key).first():
            db.session.add(SystemSettings(setting_key=key, setting_value=value))
    db.session.commit()

def get_setting(key, default=''):
    record = SystemSettings.query.filter_by(setting_key=key).first()
    return record.setting_value if record else default

def update_setting(key, value):
    record = SystemSettings.query.filter_by(setting_key=key).first()
    if record:
        record.setting_value = str(value)
    else:
        db.session.add(SystemSettings(setting_key=key, setting_value=str(value)))
    db.session.commit()

@app.context_processor
def inject_config():
    return dict(config={
        'enable_video': get_setting('enable_video') == 'true',
        'enable_rtsp': get_setting('enable_rtsp') == 'true',
        'enable_chatbot': get_setting('enable_chatbot') == 'true',
        'enable_image': get_setting('enable_image') == 'true',
        'enable_realtime': get_setting('enable_realtime') == 'true',
    })

app.register_blueprint(rtsp_bp)
app.register_blueprint(realtime_bp)

# --- HÀM LẤY RAM CỦA CONTAINER ---
def get_container_ram_info():
    try:
        if os.path.exists('/sys/fs/cgroup/memory.max'):
            with open('/sys/fs/cgroup/memory.current', 'r') as f:
                usage = int(f.read().strip())
            with open('/sys/fs/cgroup/memory.max', 'r') as f:
                limit_str = f.read().strip()
                limit = int(limit_str) if limit_str != 'max' else psutil.virtual_memory().total
        elif os.path.exists('/sys/fs/cgroup/memory/memory.limit_in_bytes'):
            with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
                usage = int(f.read().strip())
            with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                limit = int(f.read().strip())
                if limit > psutil.virtual_memory().total:
                    limit = psutil.virtual_memory().total
        else:
            vm = psutil.virtual_memory()
            return vm.percent, vm.used, vm.total

        percent = (usage / limit) * 100 if limit > 0 else 0
        return percent, usage, limit
    except Exception:
        vm = psutil.virtual_memory()
        return vm.percent, vm.used, vm.total

# --- API XÁC THỰC MÃ PIN HỆ THỐNG CŨ (Giữ nguyên tương thích ngược) ---
@app.route('/verify-pin', methods=['POST'])
def verify_pin():
    data = request.get_json()
    if data and data.get('pin') == get_setting('system_pin'):
        session['is_admin'] = True
        return jsonify({"success": True, "redirect": url_for('settings_page')})
    return jsonify({"success": False, "message": "Mã PIN không đúng."}), 401

@app.route('/')
@app.route('/index')
def index(): return render_template('index.html')

@app.route('/video')
def video(): return render_template('video.html')

@app.route('/rtsp')
def rtsp(): return render_template('rtsp.html')

@app.route('/image')
def image(): return render_template('image.html')

@app.route('/chatbot')
def chatbot(): return render_template('chatbot.html')

@app.route('/realtime')
def realtime(): return render_template('realtime.html')

@app.route('/settings')
def settings_page():
    if not session.get('is_admin'):
        return redirect(url_for('index'))
    return render_template('settings.html')

@app.route('/api/stats')
def api_stats():
    cpu = psutil.cpu_percent(interval=0.1)
    ram_percent, ram_used_bytes, ram_total_bytes = get_container_ram_info()
    return jsonify({
        "active_jobs": 0, 
        "concurrent_jobs": int(get_setting('rtsp_concurrent_jobs', 2)), 
        "queue_size": 0,
        "cpu_percent": round(cpu, 1),
        "ram_percent": round(ram_percent, 1),
        "ram_used": round(ram_used_bytes / (1024**3), 2),
        "ram_total": round(ram_total_bytes / (1024**3), 2)
    })

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    if request.method == 'POST':
        data = request.get_json()
        update_setting('enable_video', 'true' if data.get('enable_video') else 'false')
        update_setting('enable_rtsp', 'true' if data.get('enable_rtsp') else 'false')
        update_setting('enable_chatbot', 'true' if data.get('enable_chatbot') else 'false')
        update_setting('enable_image', 'true' if data.get('enable_image') else 'false')
        update_setting('enable_realtime', 'true' if data.get('enable_realtime') else 'false')
        return jsonify({"success": True})
    
    return jsonify({
        "enable_video": get_setting('enable_video') == 'true',
        "enable_rtsp": get_setting('enable_rtsp') == 'true',
        "enable_chatbot": get_setting('enable_chatbot') == 'true',
        "enable_image": get_setting('enable_image') == 'true',
        "enable_realtime": get_setting('enable_realtime') == 'true',
    })

# --- API QUẢN LÝ CHATBOT (Mới & Độc lập) ---

# API verify riêng cho Chatbot
@app.route('/api/chatbot/verify-pin', methods=['POST'])
def chatbot_verify_pin():
    data = request.get_json()
    if data and data.get('pin') == get_setting('system_pin'):
        session['chatbot_unlocked'] = True
        return jsonify({"success": True})
    return jsonify({"success": False, "message": "Mã PIN không đúng."}), 401

@app.route('/api/chatbot/config', methods=['GET', 'POST'])
def api_chatbot_config():
    # Chỉ kiểm tra session riêng của Chatbot
    if not session.get('chatbot_unlocked'):
        return jsonify({"success": False, "response": "Cần xác thực mã PIN"}), 401
    
    if request.method == 'POST':
        data = request.get_json()
        tokens = data.get('api_tokens', [])
        update_setting('chatbot_api_tokens', json.dumps(tokens))
        
        # Xóa trạng thái mở khóa sau khi lưu thành công để bắt buộc nhập lại PIN lần sau
        session.pop('chatbot_unlocked', None) 
        return jsonify({"success": True})
        
    tokens_str = get_setting('chatbot_api_tokens', '[]')
    try:
        tokens = json.loads(tokens_str)
    except:
        tokens = []
        
    # Xóa trạng thái mở khóa sau khi lấy dữ liệu cấu hình để bắt buộc nhập lại PIN lần sau
    session.pop('chatbot_unlocked', None)
    return jsonify({"success": True, "api_tokens": tokens})

@app.route('/api/chatbot/models', methods=['GET'])
def api_chatbot_models():
    tokens_str = get_setting('chatbot_api_tokens', '[]')
    try:
        tokens = json.loads(tokens_str)
    except:
        tokens = []
    
    models = []
    for token in tokens:
        if token.get('api_key'):
            model_id = token.get('id') or f"{token.get('provider')}_{token.get('model_name')}"
            is_vision = True if token.get('provider') in ['google_gemini', 'xai'] or 'vision' in token.get('model_name', '').lower() else False
            models.append({
                "id": model_id,
                "name": token.get('model_name') or token.get('provider'),
                "provider": token.get('provider'),
                "is_vision": is_vision
            })
    return jsonify({"success": True, "models": models})

@app.route('/api/chatbot/chat', methods=['POST'])
def api_chatbot_chat():
    data = request.get_json()
    message = data.get('message', '')
    return jsonify({
        "success": True,
        "response": f"Tôi đã nhận được: '{message}'. Xin hãy tích hợp backend xử lý AI.",
        "latency": "0.1"
    })

@app.route('/api/chatbot/analyze-media', methods=['POST'])
def api_chatbot_analyze_media():
    question = request.form.get('question', '')
    media_file = request.files.get('media')
    return jsonify({
        "success": True,
        "response": f"Đã tải lên tệp `{media_file.filename if media_file else 'Không rõ'}` và nhận câu hỏi: '{question}'",
        "latency": "0.5"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=24706, threaded=True)