"""
Upload model artifacts to Hugging Face Hub.
Run: .venv/bin/python scripts/upload_models_to_hf.py
"""

from pathlib import Path
from huggingface_hub import HfApi

REPO_ID    = "AnnGur/mic-creep-kpneumoniae"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

FILES = [
    ("xgb_tuned_kpneumoniae.pkl",    "xgb_tuned_kpneumoniae.pkl"),
    ("feature_names_kpneumoniae.json","feature_names_kpneumoniae.json"),
    ("xgb_tuned_abaumannii.pkl",     "xgb_tuned_abaumannii.pkl"),
    ("feature_names_abaumannii.json", "feature_names_abaumannii.json"),
]

api = HfApi()

for local_name, repo_name in FILES:
    local_path = MODELS_DIR / local_name
    if not local_path.exists():
        print(f"SKIP  {local_name} (not found)")
        continue
    print(f"Uploading {local_name} ({local_path.stat().st_size / 1e6:.1f} MB)...")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=repo_name,
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"  done -> {REPO_ID}/{repo_name}")

print("\nAll uploads complete.")
