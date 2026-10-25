"""Tạo .env phát triển với khóa ngẫu nhiên, không in khóa hoặc ghi đè file cũ."""
import secrets
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    """Khởi tạo cấu hình mock an toàn cho lượt chạy đầu tiên trên laptop."""
    root = Path(__file__).resolve().parents[1]
    target = root / ".env"
    if target.exists():
        print(".env đã tồn tại; giữ nguyên cấu hình.")
        return
    text = (root / ".env.example").read_text(encoding="utf-8")
    target.write_text(text.replace("CHANGE_ME_RANDOM_SECRET_AT_LEAST_32_CHARACTERS", secrets.token_urlsafe(32)), encoding="utf-8")
    print("Đã tạo .env development/mock. Xem khóa ở file này, không đưa lên Git.")


if __name__ == "__main__":
    main()
