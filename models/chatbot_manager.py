import random

class ChatbotManager:
    def __init__(self, settings_ref):
        self.settings = settings_ref

    def get_response(self, message):
        if not self.settings.chatbot_enabled:
            return "Xin lỗi, tính năng Chatbot hiện đang bị tắt trong phần Cài đặt."
        
        # Logic AI phản hồi cơ bản
        msg = message.lower()
        if "chào" in msg:
            return "Xin chào! Tôi là Trợ lý AI. Tôi có thể giúp gì cho bạn trong việc giám sát hành vi?"
        elif "hệ điều hành" in msg:
            return "Đây là Đồ án môn Hệ điều hành, ứng dụng tích hợp AI phân tích hành vi và xử lý đa luồng qua Flask!"
        elif "nguy hiểm" in msg or "cảnh báo" in msg:
            return "Hệ thống đang liên tục phân tích qua Camera/Video. Nếu có hành vi nguy hiểm, cảnh báo sẽ hiển thị realtime trên màn hình."
        elif "bật" in msg or "tắt" in msg or "cài đặt" in msg:
            return "Bạn có thể điều khiển bật/tắt các tính năng trong Menu Cài đặt (Icon Bánh răng)."
        else:
            responses = [
                "Tôi đã ghi nhận thông tin của bạn. Hệ thống đang hoạt động ổn định.",
                "Hệ thống phân tích hành vi đang theo dõi liên tục.",
                "Bạn cần tôi giải thích thêm về cấu trúc hệ thống hoặc tính năng nào không?"
            ]
            return random.choice(responses)
