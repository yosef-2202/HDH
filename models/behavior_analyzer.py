from flask import Blueprint, jsonify

behavior_bp = Blueprint('behavior_bp', __name__)

@behavior_bp.route('/analyze', methods=['POST', 'GET'])
def analyze_behavior():
    return jsonify({
        "status": "standby",
        "message": "Module phân tích hành vi đang chờ tích hợp model",
        "data": None
    })