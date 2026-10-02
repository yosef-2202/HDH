# 🚀 Hệ Thống AI Giám Sát Tập Trung (AI Platform)

Hệ thống AI Platform là một ứng dụng web dạng Modular được xây dựng bằng **Flask** (Python) và **Jinja2** kết hợp Bootstrap 5. Dự án cung cấp một bảng điều khiển tập trung để xử lý, phân tích dữ liệu đa phương tiện và hỗ trợ trợ lý ảo AI.

## 🌟 Các tính năng chính

Hệ thống được thiết kế linh hoạt, cho phép bật/tắt từng tính năng trực tiếp từ trang Quản trị:
- 🎥 **Xử lý Video:** Tải file video lên và áp dụng các model AI để phân tích.
- 📡 **Xử lý Luồng RTSP:** Kết nối trực tiếp với camera IP qua giao thức RTSP.
- 🤖 **Chatbot AI:** Trợ lý ảo tích hợp LLM hỗ trợ giải đáp và xử lý thông tin.
- 🖼️ **Xử lý Ảnh:** Tải ảnh lên để phân tích, nhận diện vật thể/khuôn mặt.
- ⏱️ **Phân tích Realtime:** Mô-đun xử lý dữ liệu thời gian thực (Đang phát triển).
- ⚙️ **Quản trị hệ thống:** Cấu hình API Key (Round-robin), số lượng Job đa luồng và bảo mật bằng mã PIN.

## 🛠️ Công nghệ sử dụng
- **Backend:** Python, Flask, SQLAlchemy, PyMySQL.
- **Frontend:** HTML5, CSS3, JS, Bootstrap 5, FontAwesome.
- **Database:** MySQL (Cấu hình kết nối từ xa).

## 🚀 Hướng dẫn cài đặt

**1. Clone dự án và di chuyển vào thư mục:**
```bash
git clone <url-repo-của-bạn>
cd HDH
