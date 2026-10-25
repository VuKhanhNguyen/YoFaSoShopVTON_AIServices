"""Download the runnable OOTDiffusion Hugging Face Space snapshot.

The Week 4 notebook calls the public Space remotely. This script is only for
an optional local/self-hosted installation and never accepts a token as a
command-line argument; use the HF_TOKEN environment variable or `hf auth login`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


REPO_ID = "levihsu/OOTDiffusion"
REPO_TYPE = "space"
MIN_FREE_GIB = 20
ALLOW_PATTERNS = (
    "checkpoints/**",
    "ootd/**",
    "preprocess/**",
    "run/*.py",
    "requirements.txt",
    "README.md",
)


def configure_console() -> None:
    """Keep Vietnamese status text printable in legacy Windows terminals."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


def default_target() -> Path:
    return Path(__file__).resolve().parents[1] / "api" / "models" / "ootdiffusion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tải code và checkpoint OOTDiffusion để chạy local/self-hosted."
    )
    parser.add_argument("--local-dir", type=Path, default=default_target())
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ in kế hoạch tải, không kết nối hoặc ghi dữ liệu.",
    )
    parser.add_argument(
        "--force-low-disk",
        action="store_true",
        help=f"Cho phép tiếp tục khi dung lượng trống dưới {MIN_FREE_GIB} GiB.",
    )
    return parser.parse_args()


def print_plan(target: Path, revision: str) -> None:
    print("=" * 72)
    print("OOTDiffusion local snapshot (tùy chọn, không cần cho remote Space API)")
    print(f"Nguồn      : https://huggingface.co/spaces/{REPO_ID}/tree/{revision}")
    print(f"Repo type  : {REPO_TYPE}")
    print(f"Thư mục đích: {target}")
    print("Thành phần : code chạy + preprocess + checkpoints + requirements")
    print("Bảo mật    : token chỉ đọc từ HF_TOKEN hoặc cache của `hf auth login`")
    print("=" * 72)


def main() -> int:
    configure_console()
    args = parse_args()
    target = args.local_dir.expanduser().resolve()
    print_plan(target, args.revision)

    if args.dry_run:
        print("[DRY-RUN] Không có dữ liệu nào được tải hoặc thay đổi.")
        return 0

    target.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(target).free / 1024**3
    if free_gib < MIN_FREE_GIB and not args.force_low_disk:
        print(
            f"[LỖI] Chỉ còn {free_gib:.1f} GiB trống; cần tối thiểu "
            f"{MIN_FREE_GIB} GiB. Dùng --force-low-disk nếu đã tự kiểm tra.",
            file=sys.stderr,
        )
        return 2

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "[LỖI] Thiếu huggingface_hub. Cài bằng: pip install -U huggingface_hub",
            file=sys.stderr,
        )
        return 3

    token = os.getenv("HF_TOKEN") or None
    try:
        downloaded = snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            revision=args.revision,
            local_dir=target,
            allow_patterns=list(ALLOW_PATTERNS),
            token=token,
        )
    except Exception as exc:  # Hugging Face raises several transport errors.
        print(f"[LỖI] Không tải được OOTDiffusion: {exc}", file=sys.stderr)
        print("Có thể chạy lại cùng lệnh; snapshot_download sẽ tận dụng cache.")
        return 1

    print(f"[OK] OOTDiffusion đã tải vào: {downloaded}")
    print("Lưu ý: cài dependencies và làm theo README Linux của dự án trước khi chạy local.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
