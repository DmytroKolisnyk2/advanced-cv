"""Run experiments from config.yaml (skips finished ones; every result lands in results/exps/<name>/)."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soilgs.data import cache_name  # noqa: E402
from soilgs.dino import dino_adapt  # noqa: E402
from soilgs.frozen import extract, ridge_fit_predict, sample_means  # noqa: E402
from soilgs.metric import COLS, emd  # noqa: E402
from soilgs.train import DEVICE, TileStore, run_experiment  # noqa: E402

_STORES: dict[str, TileStore] = {}


def store(cache_dir: Path, photos=None) -> TileStore:
    key = str(cache_dir)
    if key not in _STORES:
        _STORES[key] = TileStore(cache_dir, DEVICE, photos)
    return _STORES[key]


def stores_for(conf: dict, tile_mm: float) -> dict[str, TileStore]:
    cache = ROOT / conf["data"]["cache"]
    prefix = str(cache / cache_name(tile_mm))
    if prefix not in _STORES:  # switching tile size: drop the other caches from GPU memory
        _STORES.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out = {"main": store(cache / cache_name(tile_mm))}
    if tile_mm == conf["data"]["tile_mm"]:
        rb = conf["robustness"]
        for e in rb["ppm_error"]:
            out[f"ppm_error_{e:g}"] = store(cache / cache_name(tile_mm, ppm_error=e))
        for r in rb["resolution"]:
            out[f"resolution_{r:g}"] = store(cache / cache_name(tile_mm, resolution=r))
        out["scale_ref"] = store(cache / (cache_name(tile_mm) + "_ref"))
    return out


def dino_probe(labels: pd.DataFrame, folds: pd.DataFrame, st: TileStore):
    """Downstream monitor for the DINO run: ridge on frozen sample-mean features, same folds."""
    targets = labels.set_index("sample_id")[COLS]
    train_idx = st.index[st.index.split == "train"]

    def probe(backbone) -> dict:
        sub = TileStore.__new__(TileStore)
        keep = (st.index.split == "train").to_numpy()
        sub.index, sub.x = train_idx.reset_index(drop=True), st.x[torch.from_numpy(np.where(keep)[0]).to(st.x.device)]
        f = sample_means(sub.index, extract(backbone, sub)["cls+patch"])
        errs = []
        for k in sorted(folds.fold.unique()):
            va = folds.sample_id[folds.fold == k].tolist()
            tr = folds.sample_id[folds.fold != k].tolist()
            (p,), _ = ridge_fit_predict(f.loc[tr].to_numpy(), targets.loc[tr].to_numpy(), [f.loc[va].to_numpy()])
            errs.append(emd(targets.loc[va].to_numpy(), p).mean())
        backbone.train()
        return {"probe_ridge_cv": float(np.mean(errs))}
    return probe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--family", default="")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    conf = yaml.safe_load((ROOT / "config.yaml").read_text())
    labels = pd.read_csv(ROOT / "results/data/labels.csv")
    labels.columns = ["sample_id"] + COLS
    folds = pd.read_csv(ROOT / "results/data/folds.csv")
    out_root, ckpt_root = ROOT / "results/exps", ROOT / "checkpoints"
    names = [n for n, e in conf["experiments"].items()
             if args.all or n in args.only.split(",") or (args.family and e["family"] == args.family)]
    for name in names:
        e = {**conf["defaults"], **conf["experiments"][name]}
        e.setdefault("tile_mm", conf["data"]["tile_mm"])
        print(f"=== {name}", flush=True)
        try:
            kind = e.get("type", "train")
            if e.get("init_ckpt"):
                e["init_ckpt"] = str(ROOT / e["init_ckpt"])
            st = stores_for(conf, e["tile_mm"])
            if kind == "frozen":
                from soilgs.frozen import run_frozen
                res = run_frozen(name.removeprefix("frz_"), e, st, labels, folds, out_root, ROOT / "cache/features")
                for m in res:
                    print(f"  {m['name']:40s} CV {m['cv_mean']:.3f} +- {m['cv_std']:.3f}", flush=True)
                continue
            if kind == "dino":
                done = ckpt_root / name / "backbone.pt"
                if not done.exists():
                    info = dino_adapt(e, st["main"], ckpt_root / name, dino_probe(labels, folds, st["main"]))
                    (out_root / name).mkdir(parents=True, exist_ok=True)
                    (out_root / name / "dino_log.json").write_text(json.dumps(info, indent=1))
                continue
            m = run_experiment(name, e, st, labels, folds, out_root, ckpt_root)
            print(f"  CV {m['cv_mean']:.3f} +- {m['cv_std']:.3f}  ({m['gpu_hours'] * 60:.1f} min, "
                  f"{m['peak_mem_gb']} GB)", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"!!! FAILED {name}", flush=True)


if __name__ == "__main__":
    main()
