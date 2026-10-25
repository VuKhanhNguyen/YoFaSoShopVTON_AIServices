"""SQLite lưu metadata và queue bền vững trên một máy, transaction chống enqueue trùng."""
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from api.core.errors import ServiceError


def now() -> str:
    """Timestamp UTC thống nhất cho API, SQLite và worker."""
    return datetime.now(timezone.utc).isoformat()


class Store:
    """Mỗi thao tác mở connection riêng; API/worker dùng chung file trên local disk."""

    def __init__(self, root: Path):
        """Tạo schema nếu chưa có, giữ nguyên job/asset khi API restart."""
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.db = root / "state.sqlite3"
        with self.connection() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS assets (
              id TEXT PRIMARY KEY, role TEXT NOT NULL, width INT, height INT,
              bytes INT, sha256 TEXT, created_at TEXT, simulated INT DEFAULT 0);
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY, kind TEXT, business_id TEXT, payload TEXT,
              fingerprint TEXT, status TEXT, created_at TEXT, updated_at TEXT,
              attempts INT DEFAULT 0, simulated INT, result TEXT, error TEXT,
              UNIQUE(kind,business_id));
            CREATE TABLE IF NOT EXISTS job_assets (
              job_id TEXT REFERENCES jobs(id) ON DELETE CASCADE,
              asset_id TEXT REFERENCES assets(id), PRIMARY KEY(job_id,asset_id));
            CREATE INDEX IF NOT EXISTS queue_status ON jobs(status,created_at);
            CREATE TABLE IF NOT EXISTS worker (id INT PRIMARY KEY, heartbeat REAL, mode TEXT);
            """)

    @contextmanager
    def connection(self):
        """Bật foreign key, timeout và commit/rollback; không chia connection giữa thread."""
        db = sqlite3.connect(self.db, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def asset(self, asset_id: str) -> dict:
        """Chỉ đọc ID đã tồn tại; không nhận đường dẫn do client cung cấp."""
        with self.connection() as db:
            row = db.execute("SELECT * FROM assets WHERE id=?", (asset_id,)).fetchone()
        if not row:
            raise ServiceError("ASSET_NOT_FOUND", "Không tìm thấy ảnh.", 404)
        return dict(row)

    def public_asset(self, row: dict) -> dict:
        """Che path server và tên file upload khỏi response."""
        return {"assetId": row["id"], "role": row["role"], "width": row["width"], "height": row["height"],
                "bytes": row["bytes"], "sha256": row["sha256"], "createdAt": row["created_at"],
                "contentUrl": f"/internal/v1/assets/{row['id']}/content", "simulated": bool(row["simulated"])}

    def job(self, job_id: str) -> dict:
        """Đọc snapshot công việc; lỗi 404 thay vì response null không rõ nghĩa."""
        with self.connection() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise ServiceError("JOB_NOT_FOUND", "Không tìm thấy công việc.", 404)
        return dict(row)

    def public_job(self, row: dict) -> dict:
        """Giải mã result/error; không công bố input chứa thông tin riêng tư."""
        return {"jobId": row["id"], "kind": row["kind"], "businessId": row["business_id"],
                "status": row["status"], "createdAt": row["created_at"], "updatedAt": row["updated_at"],
                "attempts": row["attempts"], "simulated": bool(row["simulated"]),
                "result": json.loads(row["result"]) if row["result"] else None,
                "error": json.loads(row["error"]) if row["error"] else None}

    def submit(self, kind: str, business_id: str, payload: dict, asset_ids: list[str], mode: str, max_queue: int) -> dict:
        """Tạo job và giữ asset trong cùng transaction; ID cũ khác payload trả 409."""
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256((mode + encoded).encode()).hexdigest()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM jobs WHERE kind=? AND business_id=?", (kind, business_id)).fetchone()
            if old:
                if old["fingerprint"] != fingerprint:
                    raise ServiceError("IDEMPOTENCY_CONFLICT", "ID này đã được dùng với nội dung khác.", 409)
                return dict(old)
            count = db.execute("SELECT count(*) FROM jobs WHERE status IN ('QUEUED','PROCESSING')").fetchone()[0]
            if count >= max_queue:
                raise ServiceError("QUEUE_FULL", "Hàng đợi đã đầy, vui lòng thử sau.", 429, retryable=True)
            for asset_id in set(asset_ids):
                if not db.execute("SELECT 1 FROM assets WHERE id=?", (asset_id,)).fetchone():
                    raise ServiceError("ASSET_NOT_FOUND", "Một ảnh đầu vào không tồn tại.", 404)
            job_id, timestamp = uuid4().hex, now()
            db.execute("INSERT INTO jobs(id,kind,business_id,payload,fingerprint,status,created_at,updated_at,simulated) VALUES(?,?,?,?,?,'QUEUED',?,?,?)",
                       (job_id, kind, business_id, encoded, fingerprint, timestamp, timestamp, int(mode == "mock")))
            db.executemany("INSERT INTO job_assets VALUES(?,?)", [(job_id, aid) for aid in set(asset_ids)])
        return self.job(job_id)

    def claim(self) -> dict | None:
        """Claim atomically; worker file lock bảo đảm chỉ một GPU worker trên runtime."""
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE status='QUEUED' ORDER BY created_at LIMIT 1").fetchone()
            if not row:
                return None
            db.execute("UPDATE jobs SET status='PROCESSING', attempts=attempts+1, updated_at=? WHERE id=?", (now(), row["id"]))
        return self.job(row["id"])

    def finish(self, job_id: str, result=None, error=None):
        """Chỉ publish kết quả sau khi pipeline và ghi asset đã thành công."""
        with self.connection() as db:
            db.execute("UPDATE jobs SET status=?,result=?,error=?,updated_at=? WHERE id=? AND status='PROCESSING'",
                       ("FAILED" if error else "COMPLETED", json.dumps(result, ensure_ascii=False) if result is not None else None,
                        json.dumps(error, ensure_ascii=False) if error else None, now(), job_id))

    def transition(self, job_id: str, action: str, max_queue: int, max_attempts: int) -> dict:
        """Cancel chỉ queued; retry có giới hạn, không tự lặp ảnh sai/chất lượng lỗi."""
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                raise ServiceError("JOB_NOT_FOUND", "Không tìm thấy công việc.", 404)
            if action == "cancel":
                if row["status"] == "CANCELLED":
                    return dict(row)
                if row["status"] != "QUEUED":
                    raise ServiceError("JOB_NOT_CANCELLABLE", "Chỉ hủy được job đang chờ; không ngắt CUDA giữa chừng.", 409)
                status = "CANCELLED"
            else:
                error = json.loads(row["error"] or "{}")
                if row["status"] != "FAILED" or not error.get("retryable") or row["attempts"] >= max_attempts:
                    raise ServiceError("JOB_NOT_RETRYABLE", "Job không cho phép thử lại hoặc đã hết lượt.", 409)
                if db.execute("SELECT count(*) FROM jobs WHERE status IN ('QUEUED','PROCESSING')").fetchone()[0] >= max_queue:
                    raise ServiceError("QUEUE_FULL", "Hàng đợi đã đầy.", 429, retryable=True)
                status = "QUEUED"
            db.execute("UPDATE jobs SET status=?,result=NULL,error=NULL,updated_at=? WHERE id=?", (status, now(), job_id))
        return self.job(job_id)
