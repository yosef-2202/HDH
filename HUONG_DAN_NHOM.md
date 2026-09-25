# 🚀 Hướng Dẫn Phát Triển Đồ Án AI Giám Sát (Nhóm 5 Người)

Dự án này sử dụng kiến trúc **Modular** với Flask (Backend) và Jinja2 (Frontend). Layout chính đã được thiết kế và đóng gói. Các thành viên chỉ cần tập trung code logic trong các file/module được giao.

---

## 📁 1. Cấu trúc thư mục

/HDH
│── app.py                 # File chạy chính (Định tuyến API & Views)
│── requirements.txt       # Danh sách thư viện Python
│── static/
│   ├── css/style.css      # CSS dùng chung (Không tự ý sửa)
│   └── uploads/           # Nơi lưu file ảnh/video upload
│── templates/
│   ├── base.html          # (LÕI) Sidebar, Header & Terminal (Không sửa)
│   ├── video.html         # Giao diện Video (Thành viên 1)
│   ├── rtsp.html          # Giao diện RTSP (Thành viên 2)
│   ├── camera.html        # Giao diện Camera (Thành viên 3)
│   ├── image.html         # Giao diện Ảnh (Thành viên 4)
│   ├── chatbot.html       # Giao diện Chat (Thành viên 5)
│   └── settings.html      # Cài đặt cấu hình chung

---

2. Hướng dẫn Cài đặt & Chạy (Bắt buộc cho mọi thành viên)

Bước 1: Mở Terminal (hoặc VSCode Terminal) tại thư mục HDH.
Bước 2: Tạo môi trường ảo: python -m venv venv
Bước 3: Kích hoạt môi trường:
- Trên Windows: venv\Scripts\activate
- Trên Mac/Linux: source venv/bin/activate
Bước 4: Cài thư viện: pip install -r requirements.txt
Bước 5: Chạy server: python app.py
Bước 6: Mở trình duyệt và truy cập: http://127.0.0.1:5000

---

3. Quy tắc UI & Tránh Xung Đột Code

Để 5 người cùng code mà không bị lỗi giao diện hoặc ghi đè code của nhau, bắt buộc tuân thủ:

A. Quy tắc Frontend
- KHÔNG chỉnh sửa file base.html và style.css nếu không được sự đồng ý của cả nhóm.
- Mọi file html của từng module phải bắt đầu bằng lệnh kế thừa thẻ base.html và mở block content.
- Dùng class "card" và "card-header" đã cấu hình sẵn trong CSS để bọc các khối chức năng cho đồng bộ giao diện.

B. Quy tắc Backend (app.py)
- Định tuyến API của ai người đó viết, thêm tiền tố module để tránh trùng tên (Ví dụ: /api/video/upload).
- Nếu code logic AI quá dài, hãy tạo file python riêng trong thư mục models/ (ví dụ models/video_ai.py) và import vào app.py.

C. Quy tắc Git
- Tuyệt đối không code chung trên nhánh main.
- Mỗi người tự tạo nhánh riêng (ví dụ: feature-video, feature-chatbot...).
- Xong chức năng mới tạo Pull Request để gộp code.

---

## 🔊 4. Hướng dẫn gọi hàm System Log (Terminal UI)

Hệ thống có một Terminal giả lập ở góc dưới để hiển thị quá trình chạy AI. Các bạn **BẮT BUỘC** gọi hàm Javascript này trong file HTML của mình để log trạng thái ra UI.

**Cú pháp Javascript:**
```javascript
SystemLog.add(message, type);