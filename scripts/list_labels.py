"""Scans a folder of sign-language videos and lists every sign (label).

Folder layout (confirmed from the Zenodo previews):
    INCLUDE/<Category>/<NN. Sign name>/<video file>
e.g. Pronouns/40. I/MVI_0001.MOV
The leading number ("40. ") is removed to get the sign name.

Run:  python scripts/list_labels.py
Optional:  python scripts/list_labels.py --root data/raw/INCLUDE
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def clean_name(folder_name: str) -> str:
    """Remove numbering like '12. ' or '12_' from the start of a folder name."""
    return re.sub(r"^\s*\d+\s*[._\-)]\s*", "", folder_name).strip()


def scan(root: Path) -> pd.DataFrame:
    rows = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            relative = path.relative_to(root)
            parts = relative.parts
            if len(parts) >= 3:
                category, sign_folder = parts[-3], parts[-2]
            elif len(parts) == 2:
                category, sign_folder = "(none)", parts[-2]
            else:
                category, sign_folder = "(none)", "(none)"
            rows.append(
                {
                    "category": clean_name(category),
                    "sign": clean_name(sign_folder),
                    "folder": sign_folder,
                    "file": path.name,
                    "sample_path": str(relative),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/INCLUDE", help="folder with the extracted videos")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Folder not found: {root.resolve()}")
        print("Extract the INCLUDE zips there first.")
        return 1

    df = scan(root)
    if df.empty:
        print(f"No video files found under {root.resolve()}")
        print(f"Looked for extensions: {sorted(VIDEO_EXTENSIONS)}")
        return 1
    
    summary = (
        df.groupby(["category", "folder", "sign"])
        .agg(videos=("file", "count"), sample_path=("sample_path", "first"))
        .reset_index()
        .sort_values(["category", "folder"])
    )
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "label_inventory.csv"
    summary.to_csv(out_file, index=False)

    print(f"\nFound {len(df)} videos, {summary['sign'].nunique()} signs, "
          f"{summary['category'].nunique()} categories.\n")
    for category, group in summary.groupby("category"):
        print(f"== {category} ({len(group)} signs)")
        print(group[["folder", "videos"]].to_string(index=False))
        print()
    print(f"Saved inventory to {out_file}")
    print("Example file path (to check the folder structure):")
    print(f"  {summary['sample_path'].iloc[0]}")
    return 0
if __name__ == "__main__":
    sys.exit(main())