"""Results tables (results/tables.md, results/results.csv, results/compute_log.csv), figures and the final submission."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soilgs.metric import COLS, SIEVES, is_valid  # noqa: E402

R = ROOT / "results"
FIG = R / "figures"


def collect() -> pd.DataFrame:
    rows = []
    lb = pd.read_csv(R / "lb.csv") if (R / "lb.csv").exists() else pd.DataFrame(columns=["name", "public", "private"])
    lb = lb.drop_duplicates("name", keep="last").set_index("name")
    for p in sorted((R / "exps").glob("*/metrics.json")):
        m = json.loads(p.read_text())
        cfg = json.loads((p.parent / "config.json").read_text()) if (p.parent / "config.json").exists() else {}
        rob = m.get("robustness", {})
        rows.append({
            "name": m["name"], "family": m.get("family"), "arch": cfg.get("arch", ""),
            "setup": ", ".join(f"{k}={cfg[k]}" for k in ("opt", "lr", "layer_decay", "norm", "aug", "sampler", "batch",
                                                         "mode", "head", "features", "img", "tile_mm")
                               if cfg.get(k) not in (None, "")),
            "cv_mean": m["cv_mean"], "cv_std": m["cv_std"], "D50_err_dec": m.get("D50"), "D10_err_dec": m.get("D10"),
            "lb_public": lb.public.get(m["name"], np.nan) if len(lb) else np.nan,
            "lb_private": lb.private.get(m["name"], np.nan) if len(lb) else np.nan,
            "ctrl_shuffle": m.get("controls", {}).get("shuffle"), "ctrl_grey": m.get("controls", {}).get("grey"),
            "scale_ref": rob.get("scale_ref"), "res_0.5": rob.get("resolution_0.5"), "res_0.25": rob.get("resolution_0.25"),
            "ppm_x0.5": rob.get("ppm_error_0.5"), "ppm_x2": rob.get("ppm_error_2"),
            "cka": m.get("cka_base_vs_tuned"), "gpu": m.get("gpu"), "gpu_hours": m.get("gpu_hours"),
            "peak_mem_gb": m.get("peak_mem_gb"), "params_m": m.get("params_m"), "trainable_m": m.get("trainable_m"),
        })
    return pd.DataFrame(rows)


def fmt_table(df: pd.DataFrame, cols: list[str]) -> str:
    d = df[cols].copy()
    d.insert(1, "CV (mean ± std)", [f"{a:.3f} ± {b:.3f}" for a, b in zip(df.cv_mean, df.cv_std)])
    d = d.drop(columns=[c for c in ("cv_mean", "cv_std") if c in d])
    return d.to_markdown(index=False, floatfmt=".3f")


def plots(df: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(R / "data/labels.csv")
    curves = pd.read_csv(R / "eda/curves.csv") if (R / "eda/curves.csv").exists() else None
    # grading curves of all training samples
    fig, ax = plt.subplots(figsize=(7, 4))
    colours = {"fine": "tab:brown", "sand": "tab:orange", "gravel": "tab:gray"}
    for i, row in labels.iterrows():
        c = colours.get(curves["class"].iloc[i], "k") if curves is not None else "k"
        ax.semilogx(SIEVES, row.iloc[1:].to_numpy(float), color=c, alpha=0.7)
    ax.set(xlabel="grain diameter [mm]", ylabel="% passing", title="Training grading curves (colour = soil class)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "eda_curves.png", dpi=110)
    plt.close(fig)
    # scale robustness for the best model of each family
    fams = df[df.family.isin(["cnn", "vit", "ssl"]) & df.scale_ref.notna()]
    if len(fams):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for fam, g in fams.groupby("family"):
            b = g.loc[g.cv_mean.idxmin()]
            m = json.loads((R / "exps" / b["name"] / "metrics.json").read_text())["robustness"]
            res = [(0.25, m.get("resolution_0.25")), (0.5, m.get("resolution_0.5")), (1.0, m.get("scale_ref"))]
            axes[0].plot([r for r, _ in res], [v for _, v in res], "o-", label=f"{fam}: {b['name']}")
            err = [(0.5, m.get("ppm_error_0.5")), (0.71, m.get("ppm_error_0.71")), (1.0, m.get("scale_ref")),
                   (1.41, m.get("ppm_error_1.41")), (2.0, m.get("ppm_error_2"))]
            axes[1].plot([r for r, _ in err], [v for _, v in err], "o-", label=f"{fam}: {b['name']}")
        axes[0].set(xscale="log", xlabel="pixel density relative to the cache (correct ppm)", ylabel="CV EMD",
                    title="Same photo, fewer pixels per mm")
        axes[1].set(xscale="log", xlabel="ppm error factor (pipeline believes ppm × k)", ylabel="CV EMD",
                    title="Mis-calibrated scale")
        for a in axes:
            a.grid(alpha=0.3)
            a.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / "scale_robustness.png", dpi=110)
        plt.close(fig)
    # training curves of the main experiments
    names = [n for n in df.name if (R / "exps" / n / "train_log.json").exists()]
    if names:
        fams_order = ["cnn", "vit", "ssl"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, fam in zip(axes, fams_order):
            for n in df[(df.family == fam) & df.name.isin(names)].sort_values("cv_mean").name[:6]:
                log = json.loads((R / "exps" / n / "train_log.json").read_text())["0"]
                s = pd.DataFrame(log)
                s = s[s.stage == s.stage.iloc[-1]]
                ax.plot(s.step, s.loss.rolling(5, min_periods=1).mean(), label=n)
            ax.set(title=f"{fam}: fold-0 training EMD", xlabel="step", ylabel="train EMD (smoothed)")
            ax.legend(fontsize=6)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "training_curves.png", dpi=110)
        plt.close(fig)
    # domain adaptation run
    for p in (R / "exps").glob("da_*/dino_log.json"):
        info = json.loads(p.read_text())
        log = pd.DataFrame(info["log"])
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
        t = log.dropna(subset=["loss"]) if "loss" in log else log
        ax[0].plot(t.step, t.loss, label="DINO loss")
        ax[0].plot(t.step, t.teacher_entropy, label="teacher entropy")
        ax[0].legend()
        ax[0].set(xlabel="step", title=f"{p.parent.name}: continued DINO")
        pr = log.dropna(subset=["probe_ridge_cv"]) if "probe_ridge_cv" in log else None
        if pr is not None and len(pr):
            ax[1].plot(pr.step, pr.probe_ridge_cv, "o-")
        ax[1].set(xlabel="step", ylabel="ridge CV EMD (frozen teacher)", title="downstream probe during adaptation")
        for a in ax:
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / f"{p.parent.name}_curves.png", dpi=110)
        plt.close(fig)
    # where each approach fails: per-sample EMD of the family bests, by soil class
    if curves is not None:
        best = {f: g.loc[g.cv_mean.idxmin(), "name"] for f, g in df[df.family.isin(["cnn", "vit", "ssl"])].groupby("family")}
        rows = []
        for f, n in best.items():
            ps = json.loads((R / "exps" / n / "metrics.json").read_text())["per_sample"]
            for s, e in ps.items():
                rows.append({"family": f, "sample_id": s, "emd": e})
        ps = pd.DataFrame(rows).merge(curves[["sample_id", "class"]])
        ps.groupby(["class", "family"]).emd.mean().unstack().round(2).to_csv(R / "error_by_class.csv")


def main() -> None:
    df = collect().sort_values(["family", "cv_mean"])
    df.to_csv(R / "results.csv", index=False)
    df[["name", "family", "gpu", "gpu_hours", "peak_mem_gb", "params_m", "trainable_m"]].to_csv(
        R / "compute_log.csv", index=False)
    parts = ["# Results\n", "CV = mean ± std over the 6 group folds of the per-sample log-EMD (lower is better). "
             "LB = Kaggle public leaderboard (private is hidden until the competition ends).\n"]
    main_cols = ["name", "cv_mean", "cv_std", "D50_err_dec", "lb_public", "gpu_hours", "peak_mem_gb", "setup"]
    for fam, title in (("baseline", "Baselines and shortcut controls"), ("cnn", "Part 1 — CNNs"),
                       ("vit", "Part 2 — ViTs"), ("ssl", "Part 3 — self-supervised encoders"),
                       ("ensemble", "Part 7 — ensembles")):
        sub = df[df.family == fam]
        if len(sub):
            cols = main_cols + (["cka"] if fam == "ssl" else [])
            parts += [f"\n## {title}\n", fmt_table(sub, cols), ""]
    rob = df[df.scale_ref.notna()]
    if len(rob):
        parts += ["\n## Controls and scale robustness (CV EMD)\n",
                  rob[["name", "cv_mean", "ctrl_shuffle", "ctrl_grey", "scale_ref", "res_0.5", "res_0.25", "ppm_x0.5",
                       "ppm_x2"]].to_markdown(index=False, floatfmt=".2f")]
    (R / "tables.md").write_text("\n".join(parts) + "\n")
    plots(df)
    fc = R / "final_choice.json"
    if fc.exists():
        name = json.loads(fc.read_text())["name"]
        sub = pd.read_csv(R / "exps" / name / "test.csv")
        sub = sub[["sample_id"] + COLS]
        assert is_valid(sub[COLS].to_numpy(float)), "final submission is not a valid curve set"
        (R / "submissions").mkdir(exist_ok=True)
        sub.round(4).to_csv(R / "submissions/final.csv", index=False)
        print("final submission:", name)
    print((R / "tables.md").read_text()[:3000])


if __name__ == "__main__":
    main()
