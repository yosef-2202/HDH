# -*- coding: utf-8 -*-
# File: models/session_limiter.py
import os
import glob
import re
import threading
import time
from flask import Blueprint, request, jsonify, session

limiter_bp = Blueprint('limiter_bp', __name__)

BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEMP_FOLDER = os.path.join(BASE_PROJECT_DIR, 'static', 'temp')
UPLOAD_FOLDER = os.path.join(BASE_PROJECT_DIR, 'static', 'uploads')
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

_temp_counter_lock = threading.Lock()
_session_file_counters = {}  # { "rtsp_client_xyz": 1, "img_client_xyz": 2 }

SOURCE_PREFIX_MAP = {
    'rtsp': 'rtsp',
    'rstp': 'rtsp',
    'realtime': 'realtime',
    'video': 'video',
    'image': 'img',
    'img': 'img',
    'chatbot': 'chat',
    'chat': 'chat',
}


def sanitize_id(raw_id: str) -> str:
    clean = re.sub(r'[^a-zA-Z0-9_-]', '', str(raw_id or 'guest'))
    return clean[:48] if clean else 'guest'


def normalize_prefix(module_prefix: str) -> str:
    raw = sanitize_id(module_prefix).lower()
    return SOURCE_PREFIX_MAP.get(raw, raw or 'temp')


def build_temp_filename(module_prefix: str, client_id: str, ext: str = '.webm') -> tuple[str, str]:
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    prefix = normalize_prefix(module_prefix)
    cid = sanitize_id(client_id)
    key = f"{prefix}_{cid}"
    with _temp_counter_lock:
        seq = _session_file_counters.get(key, 0) + 1
        _session_file_counters[key] = seq

    if not ext.startswith('.'):
        ext = '.' + ext
    base_id = f"{prefix}_{cid}_{seq}"
    filepath = os.path.join(TEMP_FOLDER, f"{base_id}{ext}")
    return base_id, filepath


def cleanup_session_temp_files(module_prefix: str, client_id: str) -> int:
    if not client_id:
        return 0
    cid = sanitize_id(client_id)
    prefixes = (
        [normalize_prefix(module_prefix)]
        if module_prefix and module_prefix != '*'
        else list(set(SOURCE_PREFIX_MAP.values()))
    )

    removed_count = 0
    for prefix in prefixes:
        key = f"{prefix}_{cid}"
        with _temp_counter_lock:
            _session_file_counters.pop(key, None)

        for folder in (TEMP_FOLDER, UPLOAD_FOLDER):
            if not os.path.exists(folder):
                continue
            patterns = [f"{prefix}_{cid}_*", f"{prefix}_{cid}.*"]
            if prefix == 'rtsp':
                patterns.append(f"rstp_{cid}_*")
            elif prefix == 'img':
                patterns.append(f"image_{cid}_*")

            for pat in patterns:
                pattern = os.path.join(folder, pat)
                for fpath in glob.glob(pattern):
                    try:
                        if os.path.isfile(fpath):
                            os.remove(fpath)
                            removed_count += 1
                    except Exception:
                        pass
    return removed_count


def cleanup_source_temp_files(module_prefix: str) -> dict:
    prefix = normalize_prefix(module_prefix)
    with _temp_counter_lock:
        keys_to_remove = [k for k in _session_file_counters if k.startswith(f"{prefix}_")]
        for k in keys_to_remove:
            _session_file_counters.pop(k, None)

    deleted_files = 0
    freed_bytes = 0
    for folder in (TEMP_FOLDER, UPLOAD_FOLDER):
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if not os.path.isfile(fpath) or fname.startswith('.'):
                continue
            low = fname.lower()
            match = (
                low.startswith(f"{prefix}_") or
                (prefix == 'img' and low.startswith('image_')) or
                (prefix == 'rtsp' and low.startswith('rstp_'))
            )
            if match:
                try:
                    freed_bytes += os.path.getsize(fpath)
                    os.remove(fpath)
                    deleted_files += 1
                except Exception:
                    pass

    return {
        "source": prefix,
        "deleted_files": deleted_files,
        "freed_kb": round(freed_bytes / 1024, 1),
        "freed_mb": round(freed_bytes / (1024 * 1024), 2)
    }


def cleanup_all_client_temp_files(client_id: str) -> int:
    return cleanup_session_temp_files('*', client_id)


def get_temp_storage_stats() -> dict:
    total_files = 0
    total_bytes = 0
    by_source = {"rtsp": 0, "realtime": 0, "video": 0, "img": 0, "chat": 0, "other": 0}
    recent_files = []

    for folder in (TEMP_FOLDER, UPLOAD_FOLDER):
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if not os.path.isfile(fpath) or fname.startswith('.'):
                continue
            try:
                sz = os.path.getsize(fpath)
                mtime = os.path.getmtime(fpath)
            except OSError:
                sz = 0
                mtime = 0
            total_files += 1
            total_bytes += sz
            low = fname.lower()
            if low.startswith(('rtsp_', 'rstp_')):
                by_source['rtsp'] += 1
            elif low.startswith('realtime_'):
                by_source['realtime'] += 1
            elif low.startswith('video_'):
                by_source['video'] += 1
            elif low.startswith(('img_', 'image_')):
                by_source['img'] += 1
            elif low.startswith(('chat_', 'chatbot_')):
                by_source['chat'] += 1
            else:
                by_source['other'] += 1

            recent_files.append({
                "name": fname,
                "size_kb": round(sz / 1024, 1),
                "mtime": mtime
            })

    recent_files.sort(key=lambda x: x["mtime"], reverse=True)

    return {
        "total_files": total_files,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "total_kb": round(total_bytes / 1024, 1),
        "by_source": by_source,
        "recent_files": recent_files[:12]
    }


def clear_all_temp_files() -> dict:
    with _temp_counter_lock:
        _session_file_counters.clear()

    deleted_files = 0
    freed_bytes = 0
    for folder in (TEMP_FOLDER, UPLOAD_FOLDER):
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if not os.path.isfile(fpath) or fname.startswith('.'):
                continue
            try:
                freed_bytes += os.path.getsize(fpath)
                os.remove(fpath)
                deleted_files += 1
            except Exception:
                pass

    return {
        "deleted_files": deleted_files,
        "freed_mb": round(freed_bytes / (1024 * 1024), 2),
        "freed_kb": round(freed_bytes / 1024, 1)
    }


class ModuleQueueManager:
    def __init__(self, module_name: str, max_concurrent: int = 10, lease_timeout: float = 25.0, auto_clean_on_timeout: bool = True):
        self.module_name = module_name
        self.max_concurrent = max_concurrent
        self.lease_timeout = lease_timeout
        self.auto_clean_on_timeout = auto_clean_on_timeout
        self.cond = threading.Condition(threading.Lock())
        self.active_clients = {}
        self.waiting_queue = []
        self.wait_seen = {}

    def _cleanup_unlocked(self):
        now = time.time()
        expired_active = [
            cid for cid, ts in self.active_clients.items()
            if now - ts > self.lease_timeout
        ]
        for cid in expired_active:
            del self.active_clients[cid]
            # Với module 'image', giữ nguyên bộ nhớ đệm cho đến khi người dùng thoát Web hoặc Admin xóa
            if self.auto_clean_on_timeout:
                threading.Thread(
                    target=cleanup_session_temp_files,
                    args=(self.module_name, cid),
                    daemon=True
                ).start()

        expired_wait = [
            cid for cid in self.waiting_queue
            if now - self.wait_seen.get(cid, now) > 20.0
        ]
        for cid in expired_wait:
            self.waiting_queue.remove(cid)
            self.wait_seen.pop(cid, None)

        while len(self.active_clients) < self.max_concurrent and self.waiting_queue:
            next_cid = self.waiting_queue.pop(0)
            self.wait_seen.pop(next_cid, None)
            self.active_clients[next_cid] = now

        if expired_active or expired_wait:
            self.cond.notify_all()

    def try_acquire_stream(self, client_id: str) -> dict:
        now = time.time()
        with self.cond:
            self._cleanup_unlocked()

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

    def release(self, client_id: str, cleanup_temp: bool = False):
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

        if cleanup_temp and client_id:
            cleanup_session_temp_files(self.module_name, client_id)

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


# Riêng module 'image' đặt auto_clean_on_timeout=False để chỉ xóa temp khi người dùng thoát Web hoặc Admin ra lệnh xóa
MODULE_LIMITERS = {
    "realtime": ModuleQueueManager("realtime", max_concurrent=10, lease_timeout=18.0, auto_clean_on_timeout=True),
    "rtsp": ModuleQueueManager("rtsp", max_concurrent=10, lease_timeout=20.0, auto_clean_on_timeout=True),
    "video": ModuleQueueManager("video", max_concurrent=10, lease_timeout=120.0, auto_clean_on_timeout=True),
    "image": ModuleQueueManager("image", max_concurrent=10, lease_timeout=25.0, auto_clean_on_timeout=False),
    "chatbot": ModuleQueueManager("chatbot", max_concurrent=10, lease_timeout=60.0, auto_clean_on_timeout=True),
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
    cid = str(data.get('client_id', '')).strip() or request.headers.get('X-Client-ID', '').strip()
    if mod in MODULE_LIMITERS and cid:
        # Không tự xóa bộ nhớ đệm của image khi chỉ giải phóng slot; chỉ xóa khi thoát web (/api/session/exit)
        should_clean = (mod != 'image')
        MODULE_LIMITERS[mod].release(cid, cleanup_temp=should_clean)
    return jsonify({"success": True})


@limiter_bp.route('/api/session/exit', methods=['POST'])
def api_session_exit():
    """Endpoint nhận tín hiệu sendBeacon khi người dùng đóng tab/thoát trình duyệt -> Xóa sạch temp của phiên."""
    data = request.get_json(force=True, silent=True) or {}
    cid = str(data.get('client_id', '')).strip() or request.headers.get('X-Client-ID', '').strip()
    mod = str(data.get('module', '')).strip().lower()
    removed = 0
    if cid:
        if mod and mod in MODULE_LIMITERS:
            MODULE_LIMITERS[mod].release(cid, cleanup_temp=True)
            removed = cleanup_session_temp_files(mod, cid)
        else:
            for limiter in MODULE_LIMITERS.values():
                limiter.release(cid, cleanup_temp=False)
            removed = cleanup_all_client_temp_files(cid)
    return jsonify({"success": True, "removed_temp_files": removed})


@limiter_bp.route('/api/temp/clear_source', methods=['POST'])
def api_clear_source_temp():
    """Chỉ Admin mới được phép ra lệnh xóa bộ nhớ đệm theo nguồn (ví dụ: 'img')."""
    if not session.get('is_admin'):
        return jsonify({
            "success": False,
            "message": "Chỉ Quản trị viên (Admin) mới có quyền xóa bộ nhớ đệm!"
        }), 403

    data = request.get_json(silent=True) or {}
    source = str(data.get('source', 'img')).strip().lower()
    result = cleanup_source_temp_files(source)
    stats = get_temp_storage_stats()
    return jsonify({"success": True, **result, **stats})


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