"""OOF error correlation between approaches and ensembles built from diverse models, selected by CV only."""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soilgs.metric import COLS, DLOG  # noqa: E402
from soilgs.train import score  # noqa: E402

EXPS = ROOT / "results/exps"


def load_all() -> dict[str, dict]:
    out = {}
    for p in sorted(EXPS.glob("*/metrics.json")):
        m = json.loads(p.read_text())
        # block-8 follow-ups (soil-mixup, test-time harmonised copies) are judged on their own and kept out of the
        # search, so the submitted ensemble stays reproducible
        if (p.parent / "oof.csv").exists() and not m["name"].startswith(("ens_", "mix_")) and not m["name"].endswith("_tth"):
            m["oof"] = pd.read_csv(p.parent / "oof.csv").sort_values("sample_id").reset_index(drop=True)
            m["test"] = pd.read_csv(p.parent / "test.csv") if (p.parent / "test.csv").exists() else None
            out[m["name"]] = m
    return out


def residual(m: dict, labels: pd.DataFrame) -> np.ndarray:
    """Signed, log-width-weighted residual per sample and sieve (the quantity whose |.| sum is the EMD)."""
    y = labels.set_index("sample_id").loc[m["oof"].sample_id, COLS].to_numpy(float)
    return ((m["oof"][COLS].to_numpy(float) - y)[:, :-1] * DLOG)


def blend(members: list[dict], weights=None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    w = np.ones(len(members)) / len(members) if weights is None else np.asarray(weights) / np.sum(weights)
    oof = members[0]["oof"][["sample_id", "fold"]].copy()
    oof[COLS] = sum(wi * m["oof"][COLS].to_numpy(float) for wi, m in zip(w, members))
    test = None
    if all(m["test"] is not None for m in members):
        t = [m["test"].sort_values("sample_id").reset_index(drop=True) for m in members]
        test = t[0][["sample_id"]].copy()
        test[COLS] = sum(wi * x[COLS].to_numpy(float) for wi, x in zip(w, t))
        test[COLS[-1]] = 100.0
    return oof, test


def save(name: str, members: list[str], oof: pd.DataFrame, test, labels) -> dict:
    out = EXPS / name
    out.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out / "oof.csv", index=False)
    if test is not None:
        test.to_csv(out / "test.csv", index=False)
    m = score(oof, labels)
    m.update(name=name, family="ensemble", members=members, gpu="-", gpu_hours=0.0, peak_mem_gb=0.0)
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    return m


def main() -> None:
    labels = pd.read_csv(ROOT / "results/data/labels.csv")
    labels.columns = ["sample_id"] + COLS
    allm = load_all()
    fams = {"cnn", "vit", "ssl"}
    cands = {n: m for n, m in allm.items() if m.get("family") in fams}
    best = {f: min((m for m in cands.values() if m["family"] == f), key=lambda m: m["cv_mean"])["name"]
            for f in fams if any(m["family"] == f for m in cands.values())}
    top = sorted(cands, key=lambda n: cands[n]["cv_mean"])[:12]
    # error correlation on the per-sample, per-sieve weighted residuals
    res = {n: residual(cands[n], labels).ravel() for n in top}
    corr = pd.DataFrame(np.corrcoef(np.stack(list(res.values()))), index=list(res), columns=list(res))
    corr.round(3).to_csv(ROOT / "results/oof_error_corr.csv")
    fam_res = {f: residual(cands[n], labels).ravel() for f, n in best.items()}
    fam_corr = pd.DataFrame(np.corrcoef(np.stack(list(fam_res.values()))), index=list(fam_res), columns=list(fam_res))
    rows = []
    single = min(cands.values(), key=lambda m: m["cv_mean"])
    rows.append({"ensemble": "best single", "members": [single["name"]], "cv_mean": single["cv_mean"],
                 "cv_std": single["cv_std"]})
    plans = {"ens_best_per_family": list(best.values()), "ens_top3": top[:3], "ens_top5": top[:5]}
    # diverse pairs/triples among the top-12: lowest mean residual correlation, then lowest CV
    for r in (2, 3):
        combos = sorted(combinations(top, r), key=lambda c: np.mean([corr.loc[a, b] for a, b in combinations(c, 2)]))
        plans[f"ens_diverse{r}"] = list(combos[0])
    out = {}
    for name, members in plans.items():
        oof, test = blend([cands[n] for n in members])
        m = save(name, members, oof, test, labels)
        out[name] = m
        rows.append({"ensemble": name, "members": members, "cv_mean": m["cv_mean"], "cv_std": m["cv_std"],
                     "mean_pair_corr": float(np.mean([corr.loc[a, b] if a in corr.index and b in corr.index
                                                      else fam_corr.loc[cands[a]["family"], cands[b]["family"]]
                                                      for a, b in combinations(members, 2)]))})
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "results/ensembles.csv", index=False)
    choice = min(out.values(), key=lambda m: m["cv_mean"])
    final = choice if choice["cv_mean"] < single["cv_mean"] else single
    (ROOT / "results/final_choice.json").write_text(json.dumps(
        {"chosen_by": "CV mean EMD", "name": final["name"], "cv_mean": final["cv_mean"], "cv_std": final["cv_std"],
         "best_single": single["name"], "best_single_cv": single["cv_mean"], "family_best": best,
         "family_error_corr": fam_corr.round(3).to_dict()}, indent=1))
    print(table.to_string())
    print("family error correlation\n", fam_corr.round(3))
    print("final:", final["name"])


if __name__ == "__main__":
    main()
