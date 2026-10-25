"""Retention ảnh/job; mặc định dry-run, --apply mới xóa dữ liệu hết hạn.

Chỉ dùng runtime riêng của AI. Không dùng script này với thư mục model/catalog.
"""
import argparse
from datetime import datetime, timedelta, timezone

from api.core.config import Settings
from api.core.store import Store


def cleanup(settings, apply=False):
    """Xóa terminal job quá hạn, rồi asset quá hạn không còn được job tham chiếu."""
    store = Store(settings.runtime_dir)
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=settings.retention_hours)).isoformat()
    with store.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        jobs = db.execute("SELECT id FROM jobs WHERE status IN ('COMPLETED','FAILED','CANCELLED') AND updated_at<?", (cutoff,)).fetchall()
        if apply:
            db.executemany("DELETE FROM jobs WHERE id=?", [(j["id"],) for j in jobs])
        assets = db.execute("SELECT id FROM assets WHERE created_at<? AND id NOT IN (SELECT asset_id FROM job_assets)", (cutoff,)).fetchall()
        if apply:
            for row in assets:
                target = settings.runtime_dir / "assets" / f"{row['id']}.png"
                target.unlink(missing_ok=True)
                db.execute("DELETE FROM assets WHERE id=?", (row["id"],))
            for job in jobs:
                folder = settings.runtime_dir / "jobs" / job["id"]
                # Chỉ file do pipeline này tạo, không recursive-delete đường dẫn tính toán.
                for name in ("source.png", "prepared.png"):
                    (folder / name).unlink(missing_ok=True)
                if folder.exists() and not any(folder.iterdir()):
                    folder.rmdir()
    return {"apply": apply, "expiredJobs": len(jobs), "unreferencedExpiredAssets": len(assets)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    print(cleanup(Settings(), parser.parse_args().apply))
