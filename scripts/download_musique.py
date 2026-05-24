"""Download and extract the MuSiQue v1.0 dataset."""

import argparse
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


GOOGLE_DRIVE_FILE_ID = "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"
ZIP_PATH = Path("musique_v1.0.zip")
DATA_DIR = Path("data")
DEV_FILE = DATA_DIR / "musique_ans_v1.0_dev.jsonl"


def download_with_gdown(path: Path) -> None:
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    command = [sys.executable, "-m", "gdown", url, "-O", str(path)]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to download MuSiQue with gdown. The official MuSiQue repo "
            "uses gdown; install it with `python3 -m pip install gdown` and retry."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ZIP_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if DEV_FILE.exists():
        print(f"MuSiQue already exists: {DEV_FILE}")
        return

    if not args.archive.exists():
        print(f"Downloading official MuSiQue archive -> {args.archive}")
        download_with_gdown(args.archive)
    else:
        print(f"Using existing archive: {args.archive}")

    print(f"Extracting {args.archive} -> .")
    with ZipFile(args.archive) as archive:
        archive.extractall(".")

    if not DEV_FILE.exists():
        raise RuntimeError(f"Expected file was not extracted: {DEV_FILE}")

    print(f"Ready: {DEV_FILE}")


if __name__ == "__main__":
    main()
