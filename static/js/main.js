let statsChart;
let dataTableIndex = 1;
let pollInterval = null;

// ================= Khởi tạo Chart.js =================
function initChart() {
    const ctx = document.getElementById('statsChart').getContext('2d');
    statsChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Bình thường', 'Bất thường / Nguy hiểm'],
            datasets: [{
                data: [0, 0],
                backgroundColor: ['#198754', '#dc3545'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'bottom' }
            },
            cutout: '60%'
        }
    });
}

// ================= API & Settings Logic =================
async function fetchSettings() {
    try {
        let res = await fetch('/api/settings');
        let data = await res.json();
        document.getElementById('settingCamera').checked = data.camera_enabled;
        document.getElementById('settingDanger').checked = data.danger_analysis_enabled;
        document.getElementById('settingChatbot').checked = data.chatbot_enabled;
    } catch (e) {
        console.error("Lỗi tải cài đặt:", e);
    }
}

document.getElementById('saveSettingsBtn').addEventListener('click', async () => {
    let settings = {
        camera_enabled: document.getElementById('settingCamera').checked,
        danger_analysis_enabled: document.getElementById('settingDanger').checked,
        chatbot_enabled: document.getElementById('settingChatbot').checked
    };
    try {
        let res = await fetch('/api/settings', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(settings)
        });
        let data = await res.json();
        
        Swal.fire({ icon: 'success', title: 'Thành công', text: data.message, timer: 1500, showConfirmButton: false });
        
        // Đóng modal settings
        var modalEl = document.getElementById('settingsModal');
        var modal = bootstrap.Modal.getInstance(modalEl);
        modal.hide();
    } catch (e) {
        Swal.fire('Lỗi', 'Không thể lưu cài đặt. Xem console để biết thêm.', 'error');
    }
});

// ================= Phân tích Video Logic =================
document.getElementById('videoSource').addEventListener('change', function() {
    document.getElementById('rtspLink').classList.add('d-none');
    document.getElementById('videoFile').classList.add('d-none');
    
    if (this.value === 'rtsp') {
        document.getElementById('rtspLink').classList.remove('d-none');
    } else if (this.value === 'upload') {
        document.getElementById('videoFile').classList.remove('d-none');
    }
});

document.getElementById('startStreamBtn').addEventListener('click', async () => {
    const type = document.getElementById('videoSource').value;
    const imgEl = document.getElementById('videoStream');
    const placeholder = document.getElementById('videoPlaceholder');
    let source = '0'; // Default webcam
    
    // Kiểm tra cấu hình có cho phép bật camera không
    if (!document.getElementById('settingCamera').checked) {
        Swal.fire('Chú ý', 'Chức năng Camera đang bị tắt trong phần Cài đặt!', 'warning');
        return;
    }

    if (type === 'rtsp') {
        source = document.getElementById('rtspLink').value.trim();
        if(!source) {
            Swal.fire('Lỗi', 'Vui lòng nhập đường dẫn luồng RTSP/HTTP hợp lệ', 'warning');
            return;
        }
    } else if (type === 'upload') {
        const fileInput = document.getElementById('videoFile');
        if (fileInput.files.length === 0) {
            Swal.fire('Lỗi', 'Vui lòng chọn file video', 'warning');
            return;
        }
        
        let formData = new FormData();
        formData.append('file', fileInput.files[0]);
        Swal.fire({title: 'Đang tải lên...', allowOutsideClick: false, didOpen: () => Swal.showLoading()});
        
        try {
            let res = await fetch('/upload_video', {method: 'POST', body: formData});
            let data = await res.json();
            if(data.status === 'success') {
                Swal.close();
                source = data.filepath;
            } else {
                Swal.fire('Lỗi', data.message, 'error');
                return;
            }
        } catch (e) {
            Swal.fire('Lỗi', 'Tải file thất bại', 'error');
            return;
        }
    }
    
    // Gán src cho img thẻ để nhận luồng Multipart/x-mixed-replace (MJPEG)
    placeholder.style.display = 'none';
    imgEl.style.display = 'inline-block';
    imgEl.src = `/video_feed?type=${type}&source=${encodeURIComponent(source)}&t=${new Date().getTime()}`;
    
    // Bắt đầu Poll dữ liệu realtime
    if(pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(fetchLatestData, 1000);
    
    Swal.fire({ icon: 'success', title: 'Đã khởi tạo luồng', timer: 1500, showConfirmButton: false });
});

// Ngừng stream khi đóng modal phân tích
document.getElementById('closeAnalysisBtn').addEventListener('click', () => {
    document.getElementById('videoStream').src = '';
    document.getElementById('videoStream').style.display = 'none';
    document.getElementById('videoPlaceholder').style.display = 'block';
    if(pollInterval) clearInterval(pollInterval);
});

// Poll dữ liệu từ Server (Bảng & Biểu đồ)
async function fetchLatestData() {
    try {
        let res = await fetch('/api/latest_data');
        let json = await res.json();
        
        if (json.stats) {
            statsChart.data.datasets[0].data = [json.stats.normal, json.stats.abnormal];
            statsChart.update();
        }
        
        if (json.data) {
            const tableBody = document.querySelector('#dataTable tbody');
            const newRow = tableBody.insertRow(0); 
            
            let badgeClass = 'bg-success';
            if(json.data.danger_level === 'Cao') {
                badgeClass = 'bg-danger';
                newRow.classList.add('table-danger');
                
                // Alert nhỏ góc màn hình (Toast)
                Swal.mixin({
                    toast: true, position: 'top-end', showConfirmButton: false, timer: 3000, timerProgressBar: true
                }).fire({
                    icon: 'warning', title: 'Cảnh báo: Phát hiện hành vi nguy hiểm!'
                });
            } else if (json.data.danger_level === 'Không xác định') {
                badgeClass = 'bg-secondary';
            }
            
            newRow.innerHTML = `
                <td>${dataTableIndex++}</td>
                <td class="fw-bold">UID-${json.data.id}</td>
                <td>${json.data.features}</td>
                <td>${json.data.behavior}</td>
                <td><span class="badge ${badgeClass}">${json.data.danger_level}</span></td>
                <td>${json.data.time}</td>
                <td>${json.data.details}</td>
            `;
            
            // Chỉ giữ lại 15 dòng mới nhất tránh tràn DOM
            while (tableBody.rows.length > 15) {
                tableBody.deleteRow(tableBody.rows.length - 1);
            }
        }
    } catch (e) {
        // Ignored, maybe server restarting
    }
}

// ================= Chatbot Logic =================
document.getElementById('sendChatBtn').addEventListener('click', sendChatMessage);
document.getElementById('chatInput').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') sendChatMessage();
});

async function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const msg = input.value.trim();
    if(!msg) return;
    
    const chatBody = document.getElementById('chatBody');
    
    // Tin nhắn User
    chatBody.innerHTML += `<div class="chat-message user-message shadow-sm">${msg}</div>`;
    input.value = '';
    chatBody.scrollTop = chatBody.scrollHeight;
    
    // Hiển thị typing...
    const typingId = 'typing-' + Date.now();
    chatBody.innerHTML += `<div id="${typingId}" class="chat-message bot-message text-muted shadow-sm">
                            <i class="fas fa-circle-notch fa-spin"></i> Đang suy nghĩ...</div>`;
    chatBody.scrollTop = chatBody.scrollHeight;
    
    try {
        let res = await fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: msg})
        });
        let data = await res.json();
        
        document.getElementById(typingId).remove();
        
        let msgColor = data.enabled ? "" : "text-danger fw-bold";
        chatBody.innerHTML += `<div class="chat-message bot-message shadow-sm ${msgColor}">${data.response}</div>`;
        
    } catch(e) {
        document.getElementById(typingId).remove();
        chatBody.innerHTML += `<div class="chat-message bot-message text-danger shadow-sm">Lỗi kết nối tới máy chủ AI!</div>`;
    }
    chatBody.scrollTop = chatBody.scrollHeight;
}

// ================= Init =================
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    fetchSettings();
});
