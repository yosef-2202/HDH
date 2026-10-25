let statsChart = null;
let dataTableIndex = 1;
let pollInterval = null;

// Hàm tiện ích gắn sự kiện an toàn (chống lỗi crash JS khi chuyển giữa các trang)
function onElement(id, event, handler) {
    const el = document.getElementById(id);
    if (el) el.addEventListener(event, handler);
}

// ================= Khởi tạo Chart.js =================
function initChart() {
    const chartEl = document.getElementById('statsChart');
    if (!chartEl || typeof Chart === 'undefined') return;
    const ctx = chartEl.getContext('2d');
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
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom' }
            },
            cutout: '60%'
        }
    });
}

// ================= Xử lý Nhập Mã PIN vào Cài Đặt =================
async function promptSettingsPin(e) {
    if (e) e.preventDefault();

    // Nếu trên trang có sẵn Bootstrap Modal #pinModal thì mở và focus vào ô input
    const pinModalEl = document.getElementById('pinModal');
    if (pinModalEl && typeof bootstrap !== 'undefined') {
        const modal = bootstrap.Modal.getOrCreateInstance(pinModalEl);
        modal.show();
        setTimeout(() => {
            const pinInput = pinModalEl.querySelector('input[type="password"], input[type="tel"], input[type="text"], input[name="pin"], #pinInput');
            if (pinInput) {
                pinInput.value = '';
                pinInput.focus();
            }
        }, 350);
        return;
    }

    // Nếu không có #pinModal hoặc Modal bị lỗi trên đt, bật hộp thoại SweetAlert2 chuẩn Mobile
    if (typeof Swal !== 'undefined') {
        const { value: result } = await Swal.fire({
            title: 'Xác thực Quản trị viên',
            text: 'Vui lòng nhập mã PIN để vào Cài đặt hệ thống',
            input: 'password',
            inputPlaceholder: 'Nhập mã PIN...',
            inputAttributes: {
                autocapitalize: 'off',
                autocorrect: 'off',
                style: 'text-align: center; font-size: 1.25rem; letter-spacing: 4px;'
            },
            showCancelButton: true,
            confirmButtonText: 'Xác nhận',
            cancelButtonText: 'Hủy',
            confirmButtonColor: '#ffb7b2',
            allowOutsideClick: () => !Swal.isLoading(),
            preConfirm: async (inputPin) => {
                if (!inputPin) {
                    Swal.showValidationMessage('Vui lòng nhập mã PIN!');
                    return false;
                }
                try {
                    const res = await fetch('/verify-pin', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ pin: inputPin.trim() })
                    });
                    const data = await res.json();
                    if (!res.ok || !data.success) {
                        throw new Error(data.message || 'Mã PIN không đúng.');
                    }
                    return data;
                } catch (err) {
                    Swal.showValidationMessage(err.message);
                    return false;
                }
            }
        });

        if (result && result.redirect) {
            window.location.href = result.redirect;
        }
    } else {
        // Fallback mặc định của trình duyệt nếu không có SweetAlert2
        const inputPin = prompt('Nhập mã PIN quản trị:');
        if (!inputPin) return;
        try {
            const res = await fetch('/verify-pin', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pin: inputPin.trim() })
            });
            const data = await res.json();
            if (data.success && data.redirect) {
                window.location.href = data.redirect;
            } else {
                alert(data.message || 'Mã PIN không đúng!');
            }
        } catch (err) {
            alert('Lỗi kết nối máy chủ!');
        }
    }
}

// Hỗ trợ khi người dùng bấm nút xác nhận trong #pinModal có sẵn của HTML
function bindExistingPinModal() {
    const pinForm = document.getElementById('pinForm');
    const verifyPinBtn = document.getElementById('verifyPinBtn') || document.getElementById('submitPinBtn') || document.getElementById('confirmPinBtn');
    const pinInput = document.getElementById('pinInput') || document.querySelector('#pinModal input');

    async function submitPinFromModal() {
        const input = document.getElementById('pinInput') || document.querySelector('#pinModal input');
        if (!input) return;
        const pinVal = input.value.trim();
        if (!pinVal) return;

        try {
            const res = await fetch('/verify-pin', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pin: pinVal })
            });
            const data = await res.json();
            if (data.success && data.redirect) {
                window.location.href = data.redirect;
            } else {
                const errEl = document.getElementById('pinError');
                if (errEl) {
                    errEl.textContent = data.message || 'Mã PIN không đúng!';
                    errEl.classList.remove('d-none');
                } else if (typeof Swal !== 'undefined') {
                    Swal.fire('Lỗi', data.message || 'Mã PIN không đúng!', 'error');
                } else {
                    alert(data.message || 'Mã PIN không đúng!');
                }
            }
        } catch (e) {
            console.error('Lỗi xác thực PIN:', e);
        }
    }

    if (pinForm) {
        pinForm.addEventListener('submit', (e) => {
            e.preventDefault();
            submitPinFromModal();
        });
    }
    if (verifyPinBtn) {
        verifyPinBtn.addEventListener('click', (e) => {
            e.preventDefault();
            submitPinFromModal();
        });
    }
    if (pinInput) {
        pinInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                submitPinFromModal();
            }
        });
    }
}

// ================= API & Settings Logic =================
async function fetchSettings() {
    try {
        let res = await fetch('/api/config');
        if (!res.ok) return;
        let data = await res.json();

        const setCheck = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.checked = Boolean(val);
        };

        // Đồng bộ cả ID mới (/api/config) và ID cũ nếu có
        setCheck('enable_video', data.enable_video);
        setCheck('enable_rtsp', data.enable_rtsp);
        setCheck('enable_chatbot', data.enable_chatbot);
        setCheck('enable_image', data.enable_image);
        setCheck('enable_realtime', data.enable_realtime);

        setCheck('settingCamera', data.enable_realtime);
        setCheck('settingDanger', data.enable_rtsp);
        setCheck('settingChatbot', data.enable_chatbot);
    } catch (e) {
        console.error("Lỗi tải cài đặt:", e);
    }
}

onElement('saveSettingsBtn', 'click', async () => {
    const getCheck = (id, fallbackId) => {
        const el = document.getElementById(id) || (fallbackId ? document.getElementById(fallbackId) : null);
        return el ? el.checked : true;
    };

    let settings = {
        enable_video: getCheck('enable_video'),
        enable_rtsp: getCheck('enable_rtsp', 'settingDanger'),
        enable_chatbot: getCheck('enable_chatbot', 'settingChatbot'),
        enable_image: getCheck('enable_image'),
        enable_realtime: getCheck('enable_realtime', 'settingCamera')
    };

    try {
        let res = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings)
        });
        let data = await res.json();

        if (typeof Swal !== 'undefined') {
            Swal.fire({ icon: 'success', title: 'Thành công', text: data.message || 'Đã lưu cài đặt hệ thống!', timer: 1500, showConfirmButton: false });
        }

        const modalEl = document.getElementById('settingsModal');
        if (modalEl && typeof bootstrap !== 'undefined') {
            const modal = bootstrap.Modal.getInstance(modalEl);
            if (modal) modal.hide();
        }
    } catch (e) {
        if (typeof Swal !== 'undefined') {
            Swal.fire('Lỗi', 'Không thể lưu cài đặt. Xem console để biết thêm.', 'error');
        }
    }
});

// ================= Phân tích Video Logic =================
onElement('videoSource', 'change', function () {
    const rtspEl = document.getElementById('rtspLink');
    const fileEl = document.getElementById('videoFile');
    if (rtspEl) rtspEl.classList.add('d-none');
    if (fileEl) fileEl.classList.add('d-none');

    if (this.value === 'rtsp' && rtspEl) {
        rtspEl.classList.remove('d-none');
    } else if (this.value === 'upload' && fileEl) {
        fileEl.classList.remove('d-none');
    }
});

onElement('startStreamBtn', 'click', async () => {
    const videoSourceEl = document.getElementById('videoSource');
    const type = videoSourceEl ? videoSourceEl.value : 'webcam';
    const imgEl = document.getElementById('videoStream');
    const placeholder = document.getElementById('videoPlaceholder');
    let source = '0';

    const settingCamEl = document.getElementById('settingCamera');
    if (settingCamEl && !settingCamEl.checked) {
        Swal.fire('Chú ý', 'Chức năng Camera đang bị tắt trong phần Cài đặt!', 'warning');
        return;
    }

    if (type === 'rtsp') {
        const rtspLinkEl = document.getElementById('rtspLink');
        source = rtspLinkEl ? rtspLinkEl.value.trim() : '';
        if (!source) {
            Swal.fire('Lỗi', 'Vui lòng nhập đường dẫn luồng RTSP/HTTP hợp lệ', 'warning');
            return;
        }
    } else if (type === 'upload') {
        const fileInput = document.getElementById('videoFile');
        if (!fileInput || fileInput.files.length === 0) {
            Swal.fire('Lỗi', 'Vui lòng chọn file video', 'warning');
            return;
        }

        let formData = new FormData();
        formData.append('file', fileInput.files[0]);
        Swal.fire({ title: 'Đang tải lên...', allowOutsideClick: false, didOpen: () => Swal.showLoading() });

        try {
            let res = await fetch('/upload_video', { method: 'POST', body: formData });
            let data = await res.json();
            if (data.status === 'success') {
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

    if (placeholder) placeholder.style.display = 'none';
    if (imgEl) {
        imgEl.style.display = 'inline-block';
        imgEl.src = `/video_feed?type=${type}&source=${encodeURIComponent(source)}&t=${new Date().getTime()}`;
    }

    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(fetchLatestData, 1000);

    Swal.fire({ icon: 'success', title: 'Đã khởi tạo luồng', timer: 1500, showConfirmButton: false });
});

onElement('closeAnalysisBtn', 'click', () => {
    const imgEl = document.getElementById('videoStream');
    const placeholder = document.getElementById('videoPlaceholder');
    if (imgEl) {
        imgEl.src = '';
        imgEl.style.display = 'none';
    }
    if (placeholder) placeholder.style.display = 'block';
    if (pollInterval) clearInterval(pollInterval);
});

async function fetchLatestData() {
    try {
        let res = await fetch('/api/latest_data');
        if (!res.ok) return;
        let json = await res.json();

        if (json.stats && statsChart) {
            statsChart.data.datasets[0].data = [json.stats.normal, json.stats.abnormal];
            statsChart.update();
        }

        if (json.data) {
            const tableBody = document.querySelector('#dataTable tbody');
            if (!tableBody) return;
            const newRow = tableBody.insertRow(0);

            let badgeClass = 'bg-success';
            if (json.data.danger_level === 'Cao') {
                badgeClass = 'bg-danger';
                newRow.classList.add('table-danger');

                if (typeof Swal !== 'undefined') {
                    Swal.mixin({
                        toast: true, position: 'top-end', showConfirmButton: false, timer: 3000, timerProgressBar: true
                    }).fire({
                        icon: 'warning', title: 'Cảnh báo: Phát hiện hành vi nguy hiểm!'
                    });
                }
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

            while (tableBody.rows.length > 15) {
                tableBody.deleteRow(tableBody.rows.length - 1);
            }
        }
    } catch (e) {
        // Bỏ qua lỗi khi server đang khởi động lại
    }
}

// ================= Chatbot Logic =================
onElement('sendChatBtn', 'click', sendChatMessage);
onElement('chatInput', 'keypress', function (e) {
    if (e.key === 'Enter') sendChatMessage();
});

async function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const chatBody = document.getElementById('chatBody');
    if (!input || !chatBody) return;

    const msg = input.value.trim();
    if (!msg) return;

    chatBody.innerHTML += `<div class="chat-message user-message shadow-sm">${msg}</div>`;
    input.value = '';
    chatBody.scrollTop = chatBody.scrollHeight;

    const typingId = 'typing-' + Date.now();
    chatBody.innerHTML += `<div id="${typingId}" class="chat-message bot-message text-muted shadow-sm">
                            <i class="fas fa-circle-notch fa-spin"></i> Đang suy nghĩ...</div>`;
    chatBody.scrollTop = chatBody.scrollHeight;

    try {
        let res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg })
        });
        let data = await res.json();

        const typingEl = document.getElementById(typingId);
        if (typingEl) typingEl.remove();

        let msgColor = data.enabled ? "" : "text-danger fw-bold";
        chatBody.innerHTML += `<div class="chat-message bot-message shadow-sm ${msgColor}">${data.response}</div>`;
    } catch (e) {
        const typingEl = document.getElementById(typingId);
        if (typingEl) typingEl.remove();
        chatBody.innerHTML += `<div class="chat-message bot-message text-danger shadow-sm">Lỗi kết nối tới máy chủ AI!</div>`;
    }
    chatBody.scrollTop = chatBody.scrollHeight;
}

// ================= Heartbeat Online =================
function startHeartbeat() {
    let vid = localStorage.getItem('visitor_id');
    if (!vid) {
        vid = 'v_' + Math.random().toString(36).substring(2, 12) + Date.now();
        localStorage.setItem('visitor_id', vid);
    }
    const sendBeat = () => {
        fetch('/api/heartbeat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ visitor_id: vid, page: window.location.pathname })
        }).catch(() => {});
    };
    sendBeat();
    setInterval(sendBeat, 10000);
}

// ================= Init =================
document.addEventListener('DOMContentLoaded', () => {
    startHeartbeat();
    initChart();
    fetchSettings();
    bindExistingPinModal();

    // Bắt sự kiện khi bấm vào nút/link Cài đặt (/settings) trên thanh menu
    document.querySelectorAll('a[href="/settings"], a[href$="/settings"], [data-target="settings"], #openSettingsBtn').forEach(link => {
        link.addEventListener('click', (e) => {
            if (window.location.pathname !== '/settings') {
                promptSettingsPin(e);
            }
        });
    });

    // Tự động mở khung nhập mã PIN nếu vừa bị redirect từ /settings về trang chủ
    const params = new URLSearchParams(window.location.search);
    if (params.get('require_pin') === '1') {
        window.history.replaceState({}, document.title, window.location.pathname);
        promptSettingsPin();
    }
});