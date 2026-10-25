# -*- coding: utf-8 -*-
# File: models/session_limiter.py
import threading
import time
from flask import Blueprint, request, jsonify

limiter_bp = Blueprint('limiter_bp', __name__)


class ModuleQueueManager:
    """
    Quản lý tối đa 10 phiên đồng thời (Active Slots) và Hàng đợi FIFO (Waiting Queue)
    cho từng module độc lập (Chuẩn cơ chế Semaphore + FIFO Scheduling của HĐH).
    """
    def __init__(self, module_name: str, max_concurrent: int = 10, lease_timeout: float = 25.0):
        self.module_name = module_name
        self.max_concurrent = max_concurrent
        self.lease_timeout = lease_timeout
        self.cond = threading.Condition(threading.Lock())
        self.active_clients = {}   # {client_id: last_seen_timestamp}
        self.waiting_queue = []    # [client_id_1, client_id_2, ...]
        self.wait_seen = {}        # {client_id: last_seen_timestamp}

    def _cleanup_unlocked(self):
        now = time.time()
        # 1. Giải phóng các phiên đang chạy nhưng đã đóng tab / mất kết nối quá lease_timeout
        expired_active = [
            cid for cid, ts in self.active_clients.items()
            if now - ts > self.lease_timeout
        ]
        for cid in expired_active:
            del self.active_clients[cid]

        # 2. Xóa những người trong hàng chờ đã thoát trang quá 20s
        expired_wait = [
            cid for cid in self.waiting_queue
            if now - self.wait_seen.get(cid, now) > 20.0
        ]
        for cid in expired_wait:
            self.waiting_queue.remove(cid)
            self.wait_seen.pop(cid, None)

        # 3. Đẩy người đứng đầu hàng đợi (FIFO) vào slot trống
        while len(self.active_clients) < self.max_concurrent and self.waiting_queue:
            next_cid = self.waiting_queue.pop(0)
            self.wait_seen.pop(next_cid, None)
            self.active_clients[next_cid] = now

        if expired_active or expired_wait:
            self.cond.notify_all()

    def try_acquire_stream(self, client_id: str) -> dict:
        """Dành cho các module giữ phiên lâu (Realtime, RTSP, Video)."""
        now = time.time()
        with self.cond:
            self._cleanup_unlocked()

            # Nếu đã nằm trong 10 slot đang phục vụ -> Gia hạn heartbeat
            if client_id in self.active_clients:
                self.active_clients[client_id] = now
                return {
                    "allowed": True,
                    "status": "ACTIVE",
                    "position": 0,
                    "active_count": len(self.active_clients),
                    "max_concurrent": self.max_concurrent,
                    "waiting_count": len(self.waiting_queue)
                }

            # Nếu còn slot trống và không có ai xếp hàng trước
            if len(self.active_clients) < self.max_concurrent and not self.waiting_queue:
                self.active_clients[client_id] = now
                return {
                    "allowed": True,
                    "status": "ACTIVE",
                    "position": 0,
                    "active_count": len(self.active_clients),
                    "max_concurrent": self.max_concurrent,
                    "waiting_count": len(self.waiting_queue)
                }

            # Đã đủ 10/10 người -> Đưa người thứ 11+ vào hàng đợi FIFO
            if client_id not in self.waiting_queue:
                self.waiting_queue.append(client_id)
            self.wait_seen[client_id] = now
            pos = self.waiting_queue.index(client_id) + 1

            return {
                "allowed": False,
                "status": "WAITING",
                "position": pos,
                "active_count": len(self.active_clients),
                "max_concurrent": self.max_concurrent,
                "waiting_count": len(self.waiting_queue),
                "message": (
                    f"Module {self.module_name.upper()} đang phục vụ tối đa "
                    f"{len(self.active_clients)}/{self.max_concurrent} người cùng lúc. "
                    f"Bạn đang ở vị trí chờ #{pos}, hệ thống sẽ tự động kết nối ngay khi có phiên trống!"
                )
            }

    def acquire_blocking_task(self, client_id: str, timeout: float = 45.0) -> tuple[bool, dict]:
        """
        Dành cho các module xử lý theo lượt (Image, Chatbot).
        Nếu đang đủ 10 người xử lý cùng lúc, người thứ 11 sẽ đợi trong hàng chờ tới khi có người xong.
        """
        deadline = time.time() + timeout
        with self.cond:
            while True:
                self._cleanup_unlocked()
                now = time.time()

                if client_id in self.active_clients:
                    self.active_clients[client_id] = now
                    return True, {
                        "active_count": len(self.active_clients),
                        "max_concurrent": self.max_concurrent,
                        "waiting_count": len(self.waiting_queue)
                    }

                is_first_in_line = (not self.waiting_queue) or (self.waiting_queue[0] == client_id)
                if len(self.active_clients) < self.max_concurrent and is_first_in_line:
                    if client_id in self.waiting_queue:
                        self.waiting_queue.remove(client_id)
                    self.wait_seen.pop(client_id, None)
                    self.active_clients[client_id] = now
                    return True, {
                        "active_count": len(self.active_clients),
                        "max_concurrent": self.max_concurrent,
                        "waiting_count": len(self.waiting_queue)
                    }

                if client_id not in self.waiting_queue:
                    self.waiting_queue.append(client_id)
                self.wait_seen[client_id] = now

                remaining = deadline - now
                if remaining <= 0:
                    pos = self.waiting_queue.index(client_id) + 1 if client_id in self.waiting_queue else 1
                    return False, {
                        "status": "WAITING",
                        "position": pos,
                        "active_count": len(self.active_clients),
                        "max_concurrent": self.max_concurrent,
                        "waiting_count": len(self.waiting_queue),
                        "message": (
                            f"Module {self.module_name.upper()} đang phục vụ {self.max_concurrent}/{self.max_concurrent} người cùng lúc "
                            f"(Vị trí chờ #{pos}). Hệ thống đang giữ chỗ và sẽ tự động chạy khi tới lượt!"
                        )
                    }

                self.cond.wait(timeout=min(1.0, remaining))

    def release(self, client_id: str):
        """Giải phóng slot ngay lập tức và đánh thức người thứ 11."""
        with self.cond:
            removed = False
            if client_id in self.active_clients:
                del self.active_clients[client_id]
                removed = True
            if client_id in self.waiting_queue:
                self.waiting_queue.remove(client_id)
                self.wait_seen.pop(client_id, None)
                removed = True

            self._cleanup_unlocked()
            if removed:
                self.cond.notify_all()

    def get_status(self, client_id: str = "") -> dict:
        with self.cond:
            self._cleanup_unlocked()
            pos = 0
            state = "IDLE"
            if client_id in self.active_clients:
                state = "ACTIVE"
                self.active_clients[client_id] = time.time()
            elif client_id in self.waiting_queue:
                state = "WAITING"
                pos = self.waiting_queue.index(client_id) + 1
                self.wait_seen[client_id] = time.time()

            return {
                "module": self.module_name,
                "status": state,
                "position": pos,
                "active_count": len(self.active_clients),
                "max_concurrent": self.max_concurrent,
                "waiting_count": len(self.waiting_queue)
            }


# Khởi tạo giới hạn chuẩn 10 người đồng thời cho từng Module
MODULE_LIMITERS = {
    "realtime": ModuleQueueManager("realtime", max_concurrent=10, lease_timeout=18.0),
    "rtsp": ModuleQueueManager("rtsp", max_concurrent=10, lease_timeout=20.0),
    "video": ModuleQueueManager("video", max_concurrent=10, lease_timeout=120.0),
    "image": ModuleQueueManager("image", max_concurrent=10, lease_timeout=45.0),
    "chatbot": ModuleQueueManager("chatbot", max_concurrent=10, lease_timeout=60.0),
}


def get_limiter(module_name: str) -> ModuleQueueManager:
    return MODULE_LIMITERS[module_name]


@limiter_bp.route('/api/module_queue/acquire', methods=['POST'])
def api_acquire_slot():
    data = request.get_json(silent=True) or {}
    mod = str(data.get('module', '')).strip().lower()
    cid = str(data.get('client_id', '')).strip()
    if mod not in MODULE_LIMITERS or not cid:
        return jsonify({"success": False, "message": "Tham số không hợp lệ"}), 400
    info = MODULE_LIMITERS[mod].try_acquire_stream(cid)
    return jsonify({"success": True, **info}), (200 if info["allowed"] else 429)


@limiter_bp.route('/api/module_queue/release', methods=['POST'])
def api_release_slot():
    data = request.get_json(force=True, silent=True) or {}
    mod = str(data.get('module', '')).strip().lower()
    cid = str(data.get('client_id', '')).strip()
    if mod in MODULE_LIMITERS and cid:
        MODULE_LIMITERS[mod].release(cid)
    return jsonify({"success": True})


@limiter_bp.route('/api/module_queue/status', methods=['GET'])
def api_queue_status():
    mod = str(request.args.get('module', '')).strip().lower()
    cid = str(request.args.get('client_id', '')).strip()
    if mod in MODULE_LIMITERS:
        return jsonify({"success": True, **MODULE_LIMITERS[mod].get_status(cid)})
    return jsonify({
        "success": True,
        "modules": {k: v.get_status() for k, v in MODULE_LIMITERS.items()}
    })