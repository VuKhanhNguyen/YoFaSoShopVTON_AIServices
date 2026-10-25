"""Tải checkpoint chính thức theo .env, không ghi đè code OOTDiffusion đã sửa."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    """Chỉ tải khi người dùng chủ động chạy; cần requirements GPU/Hub đã cài."""
    from dotenv import load_dotenv
    from api.core.config import Settings
    load_dotenv(ROOT / ".env", override=False)
    settings = Settings()
    os.environ["HF_HOME"] = str(settings.hf_home)
    from huggingface_hub import hf_hub_download, snapshot_download
    if not (settings.ootd_root / "ootd").is_dir():
        raise RuntimeError("Cần đưa code OOTDiffusion vào OOTD_ROOT trước; script này chỉ tải checkpoint.")
    token = settings.hf_token.get_secret_value() if settings.hf_token else None
    snapshot_download(repo_id=os.getenv("OOTD_REPO_ID", "levihsu/OOTDiffusion"), repo_type="space",
                      revision=os.getenv("OOTD_REVISION", "main"), local_dir=settings.ootd_root,
                      allow_patterns=["checkpoints/**"], token=token)
    target = hf_hub_download(repo_id=os.getenv("DEEPFASHION_REPO_ID", "Bingsu/adetailer"),
                             filename=settings.deepfashion_weights.name, revision=os.getenv("DEEPFASHION_REVISION", "main"),
                             local_dir=settings.deepfashion_weights.parent, token=token)
    assert Path(target).resolve() == settings.deepfashion_weights
    print("Checkpoint download complete. Run real inference acceptance tests next.")


if __name__ == "__main__":
    main()
