"""Merge the results_*.zip files downloaded from Kaggle into the repo's results folders.

usage (from anywhere):  python merge_results.py path/to/results_00_recipe.zip path/to/results_01_paper_no_dp.zip ...
Lines are appended to track_*/results/results.jsonl (duplicates skipped), and best_recipe.json is copied
into shared/ if present.  Then run `python -m compare.compare` from our_work/.
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # our_work/


def main(zips):
    added = {}
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                norm = name.replace("\\", "/")
                data = z.read(name).decode("utf8")
                if norm.endswith("results.jsonl") or norm.endswith("timing.jsonl"):
                    dest = ROOT / norm
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    have = set(dest.read_text(encoding="utf8").splitlines()) if dest.exists() else set()
                    new = [l for l in data.splitlines() if l.strip() and l not in have]
                    if new:
                        with open(dest, "a", encoding="utf8") as f:
                            f.write("\n".join(new) + "\n")
                    added[norm] = added.get(norm, 0) + len(new)
                elif norm == "shared/best_recipe.json":
                    (ROOT / norm).write_text(data, encoding="utf8")
                    print("copied", norm)
    for k, v in added.items():
        print(f"{k}: +{v} lines")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
