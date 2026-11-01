"""Submit experiment test predictions to Kaggle (max 5 per day) and record the leaderboard scores.

    uv run python scripts/submit.py cnn_convnext_t_adamw vit_swin_t ...   # submit results/exps/<name>/test.csv
    uv run python scripts/submit.py --final                               # results/submissions/final.csv
    uv run python scripts/submit.py --refresh                             # only pull scores into results/lb.csv
"""

from __future__ import annotations

import io
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soilgs.metric import COLS, is_valid  # noqa: E402

COMP = "soil-grain-size-from-photos"
LB = ROOT / "results/lb.csv"


def kaggle(*args: str) -> str:
    return subprocess.run(["kaggle", *args], check=True, capture_output=True, text=True).stdout


def to_submission(src: Path, dst: Path) -> Path:
    """Reorder rows / rename columns exactly like sample_submission.csv and validate."""
    pred = pd.read_csv(src)
    tmpl = pd.read_csv(ROOT / "results/data/sample_submission.csv")
    pred = pred.set_index("sample_id").loc[tmpl.iloc[:, 0].astype(str), COLS]
    assert is_valid(pred.to_numpy(float)), f"{src} is not a valid set of curves"
    out = pd.DataFrame(pred.to_numpy().round(4), columns=tmpl.columns[1:])
    out.insert(0, tmpl.columns[0], tmpl.iloc[:, 0].to_numpy())
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    return dst


def refresh() -> pd.DataFrame:
    raw = kaggle("competitions", "submissions", "-c", COMP, "--csv")
    subs = pd.read_csv(io.StringIO(raw[raw.index("fileName"):] if "fileName" in raw else raw))
    rows = []
    for _, r in subs.iterrows():
        rows.append({"name": str(r.get("description", "")), "file": r.get("fileName"), "date": r.get("date"),
                     "status": r.get("status"), "public": pd.to_numeric(r.get("publicScore"), errors="coerce"),
                     "private": pd.to_numeric(r.get("privateScore"), errors="coerce")})
    df = pd.DataFrame(rows)
    LB.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LB, index=False)
    print(df.to_string(index=False))
    return df


def main() -> None:
    args = sys.argv[1:]
    if args == ["--refresh"]:
        refresh()
        return
    if args == ["--final"]:
        import json
        name = json.loads((ROOT / "results/final_choice.json").read_text())["name"]
        jobs = [(name, ROOT / "results/submissions/final.csv")]
    else:
        jobs = [(n, ROOT / "results/exps" / n / "test.csv") for n in args]
    for name, src in jobs:
        dst = to_submission(src, ROOT / "results/submissions" / f"{name}.csv")
        print(kaggle("competitions", "submit", "-c", COMP, "-f", str(dst), "-m", name))
    time.sleep(20)
    refresh()


if __name__ == "__main__":
    main()
