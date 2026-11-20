"""Tải trọng số DeepFashion2 theo .env (đã bỏ OOTDiffusion; dùng setup_gpu_models.py cho trọn bộ GPU)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    """Chỉ tải DeepFashion2 khi người dùng chủ động chạy."""
    from dotenv import load_dotenv
    from huggingface_hub import hf_hub_download

    from api.core.config import Settings

    load_dotenv(ROOT / ".env", override=False)
    settings = Settings()
    os.environ["HF_HOME"] = str(settings.hf_home)
    token = settings.hf_token.get_secret_value() if settings.hf_token else None
    target = hf_hub_download(
        repo_id=os.getenv("DEEPFASHION_REPO_ID", "Bingsu/adetailer"),
        filename=settings.deepfashion_weights.name,
        revision=os.getenv("DEEPFASHION_REVISION", "main"),
        local_dir=settings.deepfashion_weights.parent,
        token=token,
    )
    assert Path(target).resolve() == settings.deepfashion_weights
    print("DeepFashion2 checkpoint download complete. Chạy scripts/setup_gpu_models.py để tải đầy đủ SAM 2.1 Large + Qwen3-VL-4B + YOLO-Pose.")


if __name__ == "__main__":
    main()
