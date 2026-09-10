import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    BASE_DIR / "Dataset" / "merge_dataset.py",
    BASE_DIR / "Preprocessing" / "preprocessing.py",
    BASE_DIR / "Label" / "Labeling.py",
    BASE_DIR / "Klasifikasi" / "naive_bayes_model.py",
    BASE_DIR / "Visualisasi" / "generate_visualisasi.py",
]


def run_script(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(f"Script tidak ditemukan: {script_path}")

    print(f"\n========================================")
    print(f"Menjalankan: {script_path.name}")
    print("========================================")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR),
        text=True,
        capture_output=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Script gagal: {script_path.name} (kode: {result.returncode})")


if __name__ == "__main__":
    try:
        for script in SCRIPTS:
            run_script(script)
        print("\n========================================")
        print("Semua proses pipeline selesai berhasil.")
        print("========================================")
    except Exception as exc:
        print(f"\nPipeline gagal: {exc}")
        raise
