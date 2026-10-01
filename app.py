from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from urllib.parse import quote_plus

# Import module RTSP
from models.rtsp import rtsp_bp

app = Flask(__name__)
app.secret_key = 'hdh_secret_key_2026'

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
    setting_value = db.Column(db.String(255), nullable=True)

with app.app_context():
    db.create_all()
    # Khai báo 5 chức năng chính và các thông số cần thiết
    default_settings = {
        'system_pin': '123456',
        'token1': '',
        'token2': '',
        'concurrent_jobs': '2',
        'enable_video': 'true',
        'enable_rtsp': 'true',
        'enable_chatbot': 'true',
        'enable_image': 'true',
        'enable_realtime': 'true'
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

@app.route('/verify-pin', methods=['POST'])
def verify_pin():
    data = request.get_json()
    if data and data.get('pin') == get_setting('system_pin'):
        session['is_admin'] = True
        return jsonify({"success": True, "redirect": url_for('settings_page')})
    return jsonify({"success": False}), 401

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/rtsp')
def rtsp():
    return render_template('rtsp.html')

@app.route('/image')
def image():
    return render_template('image.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/realtime')
def realtime():
    # Render giao diện web chờ cho tính năng này
    return render_template('realtime.html')

@app.route('/settings')
def settings_page():
    if not session.get('is_admin'):
        return redirect(url_for('index'))
    return render_template('settings.html')

# --- API THỐNG KÊ ĐÃ FIX LỖI NHẢY SỐ ---
@app.route('/api/stats')
def api_stats():
    max_jobs = int(get_setting('concurrent_jobs', 2))
    return jsonify({
        "active_jobs": 0,
        "concurrent_jobs": max_jobs,
        "queue_size": 0
    })

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    if request.method == 'POST':
        data = request.get_json()
        update_setting('token1', data.get('token1', ''))
        update_setting('token2', data.get('token2', ''))
        update_setting('concurrent_jobs', str(data.get('concurrent_jobs', 2)))
        update_setting('enable_video', 'true' if data.get('enable_video') else 'false')
        update_setting('enable_rtsp', 'true' if data.get('enable_rtsp') else 'false')
        update_setting('enable_chatbot', 'true' if data.get('enable_chatbot') else 'false')
        update_setting('enable_image', 'true' if data.get('enable_image') else 'false')
        update_setting('enable_realtime', 'true' if data.get('enable_realtime') else 'false')
        return jsonify({"success": True})
    
    return jsonify({
        "gemini_tokens": [get_setting('token1'), get_setting('token2')],
        "concurrent_jobs": int(get_setting('concurrent_jobs', 2)),
        "enable_video": get_setting('enable_video') == 'true',
        "enable_rtsp": get_setting('enable_rtsp') == 'true',
        "enable_chatbot": get_setting('enable_chatbot') == 'true',
        "enable_image": get_setting('enable_image') == 'true',
        "enable_realtime": get_setting('enable_realtime') == 'true',
    })

@app.route('/api/chat', methods=['POST'])
def api_chat():
    return jsonify({"response": "Mô-đun Chatbot đang chờ tích hợp AI."})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=24706, threaded=True)