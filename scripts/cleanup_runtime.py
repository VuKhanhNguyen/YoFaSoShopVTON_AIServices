"""Retention ảnh/job; mặc định dry-run, --apply mới xóa dữ liệu hết hạn.

Chỉ dùng runtime riêng của AI. Không dùng script này với thư mục model/catalog.
"""
import argparse
import re
from datetime import datetime, timedelta, timezone

from api.core.config import Settings
from api.core.store import Store


def cleanup(settings, apply=False):
    """Xóa terminal job quá hạn, rồi asset quá hạn không còn được job tham chiếu."""
    store = Store(settings)
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=settings.retention_hours)).isoformat()
    with store.state() as state:
        jobs = [r for r in state["jobs"].values() if r["status"] in {"COMPLETED","FAILED","CANCELLED"} and r["updated_at"] < cutoff]
        removed = {r["id"] for r in jobs}
        # Dry-run tính đúng như apply, nhưng không đổi manifest.
        references = {a for j, refs in state["references"].items() if j not in removed for a in refs}
        assets = [r for r in state["assets"].values() if r["created_at"] < cutoff and r["id"] not in references]
        if apply:
            for job in jobs:
                state["jobs"].pop(job["id"])
                state["references"].pop(job["id"],None)
            for asset in assets:
                state["assets"].pop(asset["id"])
    # Chỉ xóa file SAU commit metadata: DB rollback không làm mất ảnh đang được tham chiếu.
    if apply:
        for row in assets:
            if re.fullmatch("[0-9a-f]{32}", row["id"]):
                (settings.runtime_dir / "assets" / f"{row['id']}.png").unlink(missing_ok=True)
        for job in jobs:
            if not re.fullmatch("[0-9a-f]{32}", job["id"]):
                continue
            folder = settings.runtime_dir / "jobs" / job["id"]
            if folder.is_symlink():
                continue
            for name in ("source.png", "prepared.png"):
                (folder / name).unlink(missing_ok=True)
            if folder.exists() and not any(folder.iterdir()):
                folder.rmdir()
    return {"apply": apply, "expiredJobs": len(jobs), "unreferencedExpiredAssets": len(assets)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    print(cleanup(Settings(), parser.parse_args().apply))
