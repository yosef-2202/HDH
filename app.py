from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Các route giao diện UI
@app.route('/')
def index():
    return render_template('video.html')

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/rtsp')
def rtsp():
    return render_template('rtsp.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/image')
def image():
    return render_template('image.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

# Các API xử lý Backend (Mô hình chưa được cài)
@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "Không có file"})
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # Mô hình xử lý video chưa được cài đặt ở đây
    return jsonify({"status": "success", "message": "Đã tải lên. Mô hình xử lý chưa được cài."})

@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "Không có file"})
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # Mô hình xử lý ảnh chưa được cài đặt ở đây
    return jsonify({"status": "success", "message": "Đã tải lên. Mô hình phân tích ảnh chưa được cài."})

@app.route('/api/chat', methods=['POST'])
def chat():
    user_msg = request.json.get('message')
    # Mô hình Chatbot/LLM chưa được cài đặt
    return jsonify({"response": f"Mô hình chatbot chưa được cài. Bạn vừa nói: {user_msg}"})

@app.route('/api/settings', methods=['POST'])
def save_settings():
    data = request.json
    # Lưu token và cấu hình model vào DB hoặc file config
    return jsonify({"status": "success", "message": "Đã lưu cài đặt!"})

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0')