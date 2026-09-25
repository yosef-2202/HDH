# 🛡️ Hệ thống AI Giám sát Hành vi Trực tuyến

**Hệ thống AI Giám sát Hành vi** là một ứng dụng web đa luồng, tích hợp trí tuệ nhân tạo để phát hiện và cảnh báo các hành vi bất thường (như vấp ngã, đánh nhau) theo thời gian thực. Đây là dự án thuộc học phần Hệ Điều Hành (HDH)[cite: 13], được thiết kế theo kiến trúc Modular nhằm tối ưu hóa hiệu suất xử lý hình ảnh trên máy chủ.

🔗 **Truy cập Website:** http://hdh.tbgroup.qzz.io:24706

## ⚙️ Tính năng cốt lõi
Hệ thống được chia thành 5 module xử lý độc lập:
*   **📹 Phân tích Video:** Tải lên tệp video và nhận diện hành vi qua từng khung hình (frame-by-frame).
*   **📡 Giám sát RTSP:** Trích xuất và xử lý luồng dữ liệu thời gian thực từ Camera IP.
*   **📷 Local Webcam:** Nhận diện tư thế và hành vi trực tiếp qua camera máy tính cá nhân.
*   **🖼️ Xử lý Ảnh Tĩnh:** Phân tích hình ảnh và trích xuất hệ thống khung xương (Pose Landmarks).
*   **🤖 Chatbot AI:** Trợ lý ảo LLM hỗ trợ giải đáp thông tin hệ thống và truy xuất trạng thái.

## 🛠️ Công nghệ sử dụng
*   **Backend & Xử lý đa luồng:** Python, Flask, OpenCV.
*   **AI/Deep Learning:** YOLOv8 Pose / MediaPipe.
*   **Frontend:** HTML5, CSS3, Vanilla JavaScript, Jinja2.

## 👥 Đội ngũ phát triển (Nhóm 5)
*   **Bùi Vạn Thiên (Yosef)** - Trưởng nhóm / Phát triển Core System, UI & Cấu hình máy chủ.
*   **[Tên Thành viên 2]** - Kỹ sư AI / Xử lý luồng dữ liệu Video.
*   **[Tên Thành viên 3]** - Kỹ sư Hệ thống / Tích hợp kết nối luồng mạng RTSP.
*   **[Tên Thành viên 4]** - Kỹ sư AI / Phân tích Camera Local & Xử lý ảnh tĩnh.
*   **[Tên Thành viên 5]** - Kỹ sư Tích hợp / Xây dựng logic Chatbot AI.
