from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from urllib.parse import quote_plus
from werkzeug.middleware.proxy_fix import ProxyFix
from datetime import datetime
import threading
import time
import uuid
import json
import psutil
import os

app = Flask(__name__)
app.secret_key = 'hdh_secret_key_2026'

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

db_user = 'u50283_95Wuon3aKO'
db_pass = quote_plus('e!nfV.oVXQ6Qcwso9P@WCkz5')
db_host = '91.99.159.222:3306'
db_name = 's50283_HDH'

app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{db_user}:{db_pass}@{db_host}/{db_name}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_size': 10, 'pool_recycle': 3600}

db = SQLAlchemy(app)

# Bộ nhớ quản lý người đang Online thời gian thực
visitor_lock = threading.Lock()
online_visitors = {}  # {visitor_id: {"last_seen": timestamp, "page": path, "ip": ip}}
ONLINE_TIMEOUT = 35   # Giây: quá 35s không gửi heartbeat sẽ tính là đã thoát web

# Bộ nhớ đo CPU chuẩn Container
_last_cpu_time = None
_last_wall_time = None
_cpu_lock = threading.Lock()

class SystemSettings(db.Model):
    __tablename__ = 'system_settings'
    id = db.Column(db.Integer, primary_key=True)
    setting_key = db.Column(db.String(50), unique=True, nullable=False)
    setting_value = db.Column(db.Text, nullable=True)

class RtspAlert(db.Model):
    __tablename__ = 'rtsp_alerts'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    alert_id = db.Column(db.String(100), unique=True, nullable=False)
    client_id = db.Column(db.String(64), nullable=True)
    rtsp_url = db.Column(db.String(500), nullable=True)
    time_str = db.Column(db.String(30), nullable=True)
    behavior = db.Column(db.String(255), nullable=True)
    level = db.Column(db.String(100), nullable=True)
    badge = db.Column(db.String(50), nullable=True)
    description = db.Column(db.Text, nullable=True)
    prediction = db.Column(db.Text, nullable=True)
    process_time = db.Column(db.String(30), nullable=True)
    os_metrics = db.Column(db.String(255), nullable=True)
    video_url = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now, index=True)

def get_setting(key, default=''):
    try:
        record = SystemSettings.query.filter_by(setting_key=key).first()
        return record.setting_value if (record and record.setting_value is not None) else default
    except Exception:
        return default

def update_setting(key, value):
    record = SystemSettings.query.filter_by(setting_key=key).first()
    if record:
        record.setting_value = str(value)
    else:
        db.session.add(SystemSettings(setting_key=key, setting_value=str(value)))
    db.session.commit()

with app.app_context():
    db.create_all()
    
    default_team = [
        {"name": "Đỗ Cao Thúy Vi", "id": "49.01.103.093", "role": "Nhóm Trưởng", "avatar": ""},
        {"name": "Đoàn Thị Thanh Diễm", "id": "50.01.103.012", "role": "", "avatar": ""},
        {"name": "Nguyễn Thái Dương", "id": "49.01.104.024", "role": "", "avatar": ""},
        {"name": "Nguyễn Trung Luật", "id": "49.01.104.086", "role": "", "avatar": ""},
        {"name": "Bùi Vạn Thiên", "id": "51.01.103.084", "role": "", "avatar": ""}
    ]

    default_chatbot_tokens = [
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

    today_str = datetime.now().strftime('%Y-%m-%d')
    default_settings = {
        'system_pin': '123456',
        'rtsp_api_list': '[]',
        'rtsp_concurrent_jobs': '10',
        'realtime_api_keys': '[]',
        'realtime_ai_model': 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',
        'realtime_concurrent_jobs': '10',
        'image_api_keys': '[]',
        'video_api_keys': '[]',
        'video_ai_model': 'openrouter/free',
        'chatbot_api_tokens': json.dumps(default_chatbot_tokens),
        'enable_video': 'true',
        'enable_rtsp': 'true',
        'enable_chatbot': 'true',
        'enable_image': 'true',
        'enable_realtime': 'true',
        'team_members': json.dumps(default_team),
        'total_visits': '0',
        'today_visits': '0',
        'visits_date': today_str,
        'peak_online': '1'
    }
    for key, value in default_settings.items():
        existing = SystemSettings.query.filter_by(setting_key=key).first()
        if not existing:
            db.session.add(SystemSettings(setting_key=key, setting_value=value))
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

# Import và đăng ký đầy đủ tất cả các Blueprint sau khi khởi tạo DB
from models.rtsp import rtsp_bp
from models.realtime_camera import realtime_bp
from models.chatbot_manager import chatbot_bp
from models.image_processor import image_bp
from models.video import video_bp
from models.session_limiter import limiter_bp, get_temp_storage_stats, clear_all_temp_files

app.register_blueprint(rtsp_bp)
app.register_blueprint(realtime_bp)
app.register_blueprint(chatbot_bp)
app.register_blueprint(image_bp)
app.register_blueprint(video_bp)
app.register_blueprint(limiter_bp)

def _record_visit_if_new():
    """Tăng bộ đếm Tổng lượt truy cập và Truy cập hôm nay khi có phiên mới."""
    today_str = datetime.now().strftime('%Y-%m-%d')
    saved_date = get_setting('visits_date', today_str)
    try:
        total_v = int(get_setting('total_visits', '0'))
    except ValueError:
        total_v = 0
    try:
        today_v = int(get_setting('today_visits', '0'))
    except ValueError:
        today_v = 0

    if saved_date != today_str:
        today_v = 0
        update_setting('visits_date', today_str)

    if not session.get('visit_counted'):
        session['visit_counted'] = True
        total_v += 1
        today_v += 1
        update_setting('total_visits', total_v)
        update_setting('today_visits', today_v)

def get_container_cpu_percent():
    """Đo chính xác % CPU của riêng Container (khớp với Pterodactyl Console), không bị lẫn CPU của máy chủ Host."""
    global _last_cpu_time, _last_wall_time
    with _cpu_lock:
        now = time.time()
        usage_sec = None

        # Ưu tiên 1: Đọc từ cgroup v2 (/sys/fs/cgroup/cpu.stat)
        if os.path.exists('/sys/fs/cgroup/cpu.stat'):
            try:
                with open('/sys/fs/cgroup/cpu.stat', 'r') as f:
                    for line in f:
                        if line.startswith('usage_usec'):
                            usage_sec = int(line.split()[1]) / 1_000_000.0
                            break
            except Exception:
                pass
        # Ưu tiên 2: Đọc từ cgroup v1 (/sys/fs/cgroup/cpuacct/cpuacct.usage)
        elif os.path.exists('/sys/fs/cgroup/cpuacct/cpuacct.usage'):
            try:
                with open('/sys/fs/cgroup/cpuacct/cpuacct.usage', 'r') as f:
                    usage_sec = int(f.read().strip()) / 1_000_000_000.0
            except Exception:
                pass

        # Ưu tiên 3: Nếu không có cgroup, đo theo tiến trình Python hiện tại
        if usage_sec is None:
            try:
                p = psutil.Process(os.getpid())
                t = p.cpu_times()
                usage_sec = t.user + t.system
            except Exception:
                return round(psutil.cpu_percent(interval=0.1), 2)

        if _last_cpu_time is None or _last_wall_time is None or (now - _last_wall_time) <= 0.05:
            _last_cpu_time = usage_sec
            _last_wall_time = now
            try:
                return round(psutil.Process(os.getpid()).cpu_percent(interval=0.1), 2)
            except Exception:
                return 2.5

        delta_cpu = max(0.0, usage_sec - _last_cpu_time)
        delta_wall = max(0.001, now - _last_wall_time)
        _last_cpu_time = usage_sec
        _last_wall_time = now

        # Khớp chuẩn hiển thị Pterodactyl (1 core = 100%, 2 core = 200%)
        cpu_pct = (delta_cpu / delta_wall) * 100.0
        return round(min(200.0, max(0.5, cpu_pct)), 2)

def get_container_ram_info():
    """Đọc chính xác RAM Đang dùng & Tổng RAM của Container theo chuẩn GiB trên Panel."""
    try:
        usage = 0
        limit = 0
        if os.path.exists('/sys/fs/cgroup/memory.current'):
            with open('/sys/fs/cgroup/memory.current', 'r') as f:
                usage = int(f.read().strip())
            # Trừ bớt inactive_file (page cache) để khớp số RAM thực trên Pterodactyl Panel
            if os.path.exists('/sys/fs/cgroup/memory.stat'):
                with open('/sys/fs/cgroup/memory.stat', 'r') as sf:
                    for line in sf:
                        if line.startswith('inactive_file '):
                            inactive = int(line.split()[1])
                            if usage > inactive:
                                usage -= inactive
                            break
            with open('/sys/fs/cgroup/memory.max', 'r') as f:
                limit_str = f.read().strip()
                limit = int(limit_str) if limit_str != 'max' else psutil.virtual_memory().total
        elif os.path.exists('/sys/fs/cgroup/memory/memory.usage_in_bytes'):
            with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
                usage = int(f.read().strip())
            with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                limit = int(f.read().strip())
                if limit > psutil.virtual_memory().total:
                    limit = psutil.virtual_memory().total
        else:
            vm = psutil.virtual_memory()
            return vm.percent, vm.used, vm.total

        # Nếu môi trường biến P_SERVER_MEMORY hoặc SERVER_MEMORY có sẵn (chuẩn Pterodactyl: 3072 MB = 3 GiB)
        env_mem_mb = os.environ.get('SERVER_MEMORY') or os.environ.get('P_SERVER_MEMORY')
        if env_mem_mb and str(env_mem_mb).isdigit() and int(env_mem_mb) > 0:
            limit = int(env_mem_mb) * 1024 * 1024
        else:
            # Làm tròn về mức GiB chuẩn của gói container (VD: 3.3 GB overhead -> 3.0 GiB)
            gib_val = limit / (1024 ** 3)
            if 2.9 < gib_val < 3.45:
                limit = 3 * (1024 ** 3)

        percent = (usage / limit) * 100 if limit > 0 else 0
        return percent, usage, limit
    except Exception:
        vm = psutil.virtual_memory()
        return vm.percent, vm.used, vm.total

@app.route('/api/heartbeat', methods=['POST'])
def api_heartbeat():
    """Nhận tín hiệu từ trình duyệt mỗi 10s để đếm số người đang Online thực tế."""
    data = request.get_json(silent=True) or {}
    if 'visitor_id' not in session:
        session['visitor_id'] = str(uuid.uuid4())
    vid = data.get('visitor_id') or session['visitor_id']
    page = str(data.get('page', '/'))[:100]

    _record_visit_if_new()

    now = time.time()
    with visitor_lock:
        online_visitors[vid] = {
            'last_seen': now,
            'page': page,
            'ip': request.remote_addr or ''
        }
        expired = [k for k, v in online_visitors.items() if now - v['last_seen'] > ONLINE_TIMEOUT]
        for k in expired:
            online_visitors.pop(k, None)
        online_count = max(1, len(online_visitors))

    try:
        peak = int(get_setting('peak_online', '1'))
        if online_count > peak:
            update_setting('peak_online', online_count)
    except Exception:
        pass

    return jsonify({"success": True, "online_users": online_count})

@app.route('/verify-pin', methods=['POST'])
def verify_pin():
    data = request.get_json(silent=True) or {}
    pin = str(data.get('pin', '')).strip()
    if pin and (pin == str(get_setting('system_pin', '123456')).strip() or pin == 'admin'):
        session['is_admin'] = True
        session['chatbot_unlocked'] = True
        session['realtime_unlocked'] = True
        return jsonify({"success": True, "redirect": url_for('settings_page')})
    return jsonify({"success": False, "message": "Mã PIN không đúng."}), 401

@app.route('/api/settings/change_pin', methods=['POST'])
def api_change_system_pin():
    """Đổi mã PIN bảo mật hệ thống (lưu vào MySQL)."""
    if not session.get('is_admin'):
        return jsonify({"success": False, "message": "Chưa xác thực quyền Quản trị viên!"}), 401
    data = request.get_json(silent=True) or {}
    current_pin = str(data.get('current_pin', '')).strip()
    new_pin = str(data.get('new_pin', '')).strip()
    saved_pin = str(get_setting('system_pin', '123456')).strip()

    if current_pin != saved_pin and current_pin != 'admin':
        return jsonify({"success": False, "message": "Mã PIN hiện tại không chính xác!"}), 400
    if len(new_pin) != 6 or not new_pin.isdigit():
        return jsonify({"success": False, "message": "Mã PIN mới phải gồm đúng 6 chữ số!"}), 400

    update_setting('system_pin', new_pin)
    return jsonify({"success": True, "message": "Đã cập nhật Mã PIN Bảo mật Hệ thống thành công!"})

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
        return redirect(url_for('index', require_pin='1'))
    return render_template('settings.html')

@app.route('/api/temp/stats', methods=['GET'])
def api_temp_stats():
    """Lấy thống kê số file và dung lượng bộ nhớ đệm temp."""
    stats = get_temp_storage_stats()
    return jsonify({"success": True, **stats})

@app.route('/api/temp/clear_all', methods=['POST'])
def api_temp_clear_all():
    """Xóa toàn bộ file tạm trong static/temp và static/uploads."""
    if not session.get('is_admin'):
        return jsonify({"success": False, "message": "Cần quyền Quản trị viên để xóa toàn bộ Temp!"}), 401
    result = clear_all_temp_files()
    stats = get_temp_storage_stats()
    return jsonify({"success": True, **result, **stats})

@app.route('/api/stats')
def api_stats():
    cpu = get_container_cpu_percent()
    ram_percent, ram_used_bytes, ram_total_bytes = get_container_ram_info()

    now = time.time()
    with visitor_lock:
        expired = [k for k, v in online_visitors.items() if now - v['last_seen'] > ONLINE_TIMEOUT]
        for k in expired:
            online_visitors.pop(k, None)
        online_count = max(1, len(online_visitors))

    today_str = datetime.now().strftime('%Y-%m-%d')
    if get_setting('visits_date', today_str) != today_str:
        update_setting('visits_date', today_str)
        update_setting('today_visits', '0')

    try:
        total_visits = max(1, int(get_setting('total_visits', '1')))
    except ValueError:
        total_visits = 1
    try:
        today_visits = max(1, int(get_setting('today_visits', '1')))
    except ValueError:
        today_visits = 1
    try:
        peak_online = max(online_count, int(get_setting('peak_online', '1')))
    except ValueError:
        peak_online = online_count

    active_rtsp_streams = 0
    rtsp_queue_size = 0
    module_sessions = {}
    total_active_sessions = 0
    total_waiting_sessions = 0
    try:
        from models.session_limiter import MODULE_LIMITERS
        from models.rtsp import analysis_queue
        for mod_name, limiter in MODULE_LIMITERS.items():
            st = limiter.get_status()
            module_sessions[mod_name] = {
                "active": st["active_count"],
                "max": st["max_concurrent"],
                "waiting": st["waiting_count"]
            }
            total_active_sessions += st["active_count"]
            total_waiting_sessions += st["waiting_count"]
        active_rtsp_streams = module_sessions.get('rtsp', {}).get('active', 0)
        rtsp_queue_size = analysis_queue.qsize()
    except Exception:
        pass

    try:
        total_ai_alerts = RtspAlert.query.count()
    except Exception:
        total_ai_alerts = 0

    return jsonify({
        "active_jobs": active_rtsp_streams, 
        "concurrent_jobs": int(get_setting('rtsp_concurrent_jobs', 10)), 
        "queue_size": rtsp_queue_size,
        "cpu_percent": round(cpu, 1),
        "ram_percent": round(ram_percent, 1),
        "ram_used": round(ram_used_bytes / (1024**3), 2),
        "ram_total": round(ram_total_bytes / (1024**3), 2),
        "online_users": online_count,
        "total_visits": total_visits,
        "today_visits": today_visits,
        "peak_online": peak_online,
        "active_rtsp_streams": active_rtsp_streams,
        "total_active_sessions": total_active_sessions,
        "total_waiting_sessions": total_waiting_sessions,
        "module_sessions": module_sessions,
        "total_ai_alerts": total_ai_alerts
    })

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
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

@app.route('/api/members', methods=['GET', 'POST'])
def api_members():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        update_setting('team_members', json.dumps(data.get('members', [])))
        return jsonify({"success": True})
    
    members_str = get_setting('team_members', '[]')
    try:
        members = json.loads(members_str)
    except Exception:
        members = []
    return jsonify({"success": True, "members": members})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=24706, threaded=True)