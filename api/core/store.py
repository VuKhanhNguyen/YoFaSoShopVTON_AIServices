"""Manifest runtime bền vững, không kết nối SQL và không tạo entity ngoài ERD.

API và một worker dùng chung volume trên MỘT host. Khóa file liên process;
ghi temp + fsync + os.replace. Không giữ khóa khi chạy inference.
"""
import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import uuid4

from api.core.errors import ServiceError


def now():
    return datetime.now(timezone.utc).isoformat()


_locks = {}
_guard = threading.Lock()


class Store:
    def __init__(self, settings):
        self.settings = settings
        self.root = settings.runtime_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "runtime.json"
        with _guard:
            self.lock = _locks.setdefault(str(self.root), threading.RLock())

    @contextmanager
    def state(self):
        """Rollback thay đổi trong RAM nếu domain error; file lỗi được báo, không tự xóa."""
        with self.lock, (self.root / "manifest.lock").open("a+b") as handle:
            if handle.seek(0, 2) == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            deadline = time.monotonic() + 10
            while True:
                try:
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise ServiceError("STORAGE_BUSY", "Runtime đang bận.", 503, retryable=True)
                    time.sleep(.02)
            try:
                data = json.loads(self.manifest.read_text("utf-8")) if self.manifest.exists() else {
                    "version": 1, "assets": {}, "jobs": {}, "references": {}, "heartbeat": {}}
                if data.get("version") != 1:
                    raise ValueError("Unsupported runtime version")
                before = json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False)
                yield data
                encoded = json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False)
                if before != encoded or not self.manifest.exists():
                    temporary = self.root / (uuid4().hex + ".json.tmp")
                    try:
                        with temporary.open("x", encoding="utf-8") as output:
                            output.write(encoded)
                            output.flush()
                            os.fsync(output.fileno())
                        os.replace(temporary, self.manifest)
                    finally:
                        temporary.unlink(missing_ok=True)
            except (OSError, ValueError, KeyError) as exc:
                raise ServiceError("STORAGE_UNAVAILABLE", "Manifest runtime lỗi; phục hồi từ bản sao lưu.", 503) from exc
            finally:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def check_schema(self):
        """Chỉ kiểm tra manifest, không chạy migration/database."""
        with self.state():
            pass

    def asset(self, asset_id):
        if not isinstance(asset_id, str) or len(asset_id) != 32 or any(c not in "0123456789abcdef" for c in asset_id):
            raise ServiceError("ASSET_NOT_FOUND", "Không tìm thấy ảnh.", 404)
        with self.state() as data:
            row = data["assets"].get(asset_id)
        if not row:
            raise ServiceError("ASSET_NOT_FOUND", "Không tìm thấy ảnh.", 404)
        return row

    def register_asset(self, row, job_id=None):
        with self.state() as data:
            if job_id and job_id not in data["jobs"]:
                raise ServiceError("JOB_NOT_FOUND", "Không tìm thấy job.", 404)
            data["assets"][row["id"]] = row
            if job_id:
                refs = data["references"].setdefault(job_id, [])
                refs.append(row["id"])

    def delete_asset(self, asset_id):
        with self.state() as data:
            if asset_id not in data["assets"]:
                raise ServiceError("ASSET_NOT_FOUND", "Không tìm thấy ảnh.", 404)
            if any(asset_id in refs for refs in data["references"].values()):
                raise ServiceError("ASSET_IN_USE", "Ảnh được job tham chiếu.", 409)
            del data["assets"][asset_id]

    def public_asset(self, row):
        return {"assetId": row["id"], "role": row["role"], "width": row["width"], "height": row["height"],
                "bytes": row["bytes"], "sha256": row["sha256"], "createdAt": row["created_at"],
                "contentUrl": f"/internal/v1/assets/{row['id']}/content", "simulated": bool(row["simulated"])}

    def job(self, job_id):
        with self.state() as data:
            row = data["jobs"].get(job_id)
        if not row:
            raise ServiceError("JOB_NOT_FOUND", "Không tìm thấy công việc.", 404)
        return row

    def find_job(self, kind, business_id):
        with self.state() as data:
            row = next((r for r in data["jobs"].values() if r["kind"] == kind and r["business_id"] == business_id), None)
        if not row:
            raise ServiceError("JOB_NOT_FOUND", "Không tìm thấy công việc.", 404)
        return row

    def list_jobs(self, limit, offset, status=None):
        with self.state() as data:
            rows = sorted((r for r in data["jobs"].values() if not status or r["status"] == status),
                          key=lambda r: (r["created_at"], r["id"]), reverse=True)
        return {"items": [self.public_job(r) for r in rows[offset:offset+limit]], "total": len(rows), "limit": limit, "offset": offset}

    def public_job(self, row):
        return {"jobId": row["id"], "kind": row["kind"], "businessId": row["business_id"], "status": row["status"],
                "createdAt": row["created_at"], "updatedAt": row["updated_at"], "attempts": row["attempts"],
                "simulated": bool(row["simulated"]), "result": json.loads(row["result"]) if row["result"] else None,
                "error": json.loads(row["error"]) if row["error"] else None}

    def submit(self, kind, business_id, payload, asset_ids, mode, max_queue):
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        fingerprint = hashlib.sha256((mode + encoded).encode()).hexdigest()
        with self.state() as data:
            for row in data["jobs"].values():
                if row["kind"] == kind and row["business_id"] == business_id:
                    if row["fingerprint"] != fingerprint:
                        raise ServiceError("IDEMPOTENCY_CONFLICT", "ID đã dùng với nội dung khác.", 409)
                    return row
            if sum(r["status"] in {"QUEUED", "PROCESSING"} for r in data["jobs"].values()) >= max_queue:
                raise ServiceError("QUEUE_FULL", "Hàng đợi đã đầy.", 429, retryable=True)
            if any(a not in data["assets"] for a in asset_ids):
                raise ServiceError("ASSET_NOT_FOUND", "Ảnh không tồn tại.", 404)
            identifier, timestamp = uuid4().hex, now()
            row = dict(id=identifier, kind=kind, business_id=business_id, payload=encoded, fingerprint=fingerprint,
                       status="QUEUED", attempts=0, created_at=timestamp, updated_at=timestamp,
                       simulated=int(mode == "mock"), result=None, error=None, claim_token=None)
            data["jobs"][identifier] = row
            data["references"][identifier] = sorted(set(asset_ids))
        return row

    def claim(self):
        with self.state() as data:
            rows = sorted((r for r in data["jobs"].values() if r["status"] == "QUEUED"), key=lambda r: (r["created_at"], r["id"]))
            if not rows:
                return None
            row = rows[0]
            row.update(status="PROCESSING", attempts=row["attempts"]+1, updated_at=now(), claim_token=uuid4().hex)
        return row

    def finish(self, job_id, *, claim_token, result=None, error=None):
        with self.state() as data:
            row = data["jobs"].get(job_id)
            if not row or row["status"] != "PROCESSING" or row["claim_token"] != claim_token:
                raise ServiceError("JOB_LEASE_LOST", "Job không còn thuộc lượt xử lý này.", 409)
            row.update(status="FAILED" if error else "COMPLETED", result=json.dumps(result, allow_nan=False) if result is not None else None,
                       error=json.dumps(error) if error else None, updated_at=now(), claim_token=None)

    def transition(self, job_id, action, max_queue, max_attempts):
        with self.state() as data:
            row = data["jobs"].get(job_id)
            if not row:
                raise ServiceError("JOB_NOT_FOUND", "Không tìm thấy công việc.", 404)
            if action == "cancel":
                if row["status"] == "CANCELLED":
                    return row
                if row["status"] != "QUEUED":
                    raise ServiceError("JOB_NOT_CANCELLABLE", "Chỉ hủy được job đang chờ.", 409)
                status = "CANCELLED"
            elif action == "retry":
                if row["status"] != "FAILED" or not json.loads(row["error"] or "{}").get("retryable") or row["attempts"] >= max_attempts:
                    raise ServiceError("JOB_NOT_RETRYABLE", "Job không cho phép thử lại.", 409)
                if sum(r["status"] in {"QUEUED", "PROCESSING"} for r in data["jobs"].values()) >= max_queue:
                    raise ServiceError("QUEUE_FULL", "Hàng đợi đã đầy.", 429, retryable=True)
                status = "QUEUED"
            else:
                raise ServiceError("ACTION_INVALID", "Thao tác không hợp lệ.")
            row.update(status=status, result=None, error=None, updated_at=now(), claim_token=None)
        return row

    def heartbeat(self, timestamp=None, mode=None, *, write=False):
        with self.state() as data:
            if write:
                data["heartbeat"] = {"heartbeat": timestamp, "mode": mode}
            return data["heartbeat"]

    def recover(self):
        with self.state() as data:
            for row in data["jobs"].values():
                if row["status"] == "PROCESSING":
                    row.update(status="FAILED", claim_token=None, updated_at=now(), error=json.dumps({
                        "errorCode": "WORKER_INTERRUPTED", "message": "Worker đã dừng; có thể retry.", "retryable": True}))
