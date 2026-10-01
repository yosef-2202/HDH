from flask import Blueprint, jsonify, request

chatbot_bp = Blueprint('chatbot_bp', __name__)

@chatbot_bp.route('/chat', methods=['POST'])
def handle_chat():
    user_message = request.json.get('message', '') if request.is_json else "Không có dữ liệu"
    return jsonify({
        "status": "standby",
        "reply": f"Tin nhắn đã nhận: '{user_message}'. Chatbot AI hiện đang bảo trì."
    })