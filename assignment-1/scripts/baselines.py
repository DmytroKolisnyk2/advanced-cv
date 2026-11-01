"""Image-free and shortcut baselines on the same folds (CPU):
uniform (Kaggle baseline), training mean, camera/scale only (metadata shortcut), colour statistics only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soilgs.data import cache_name  # noqa: E402
from soilgs.frozen import ridge_fit_predict  # noqa: E402
from soilgs.metric import COLS, UNIFORM_BASELINE  # noqa: E402
from soilgs.train import score  # noqa: E402


def colour_features(cache: Path) -> pd.DataFrame:
    """Per-sample colour statistics (RGB/HSV mean and std, 8-bin RGB histograms), averaged over tiles."""
    idx = pd.read_csv(cache / "tiles.csv")
    arr = np.load(cache / "tiles.npy", mmap_mode="r")
    feats = []
    for i in range(0, len(idx), 512):
        x = np.asarray(arr[i:i + 512], dtype=np.float32) / 255
        mx, mn = x.max(-1), x.min(-1)
        s = (mx - mn) / (mx + 1e-6)
        f = [x.mean((1, 2)), x.std((1, 2)), mx.mean((1, 2))[:, None], s.mean((1, 2))[:, None], s.std((1, 2))[:, None]]
        for c in range(3):
            f.append(np.stack([np.histogram(t[..., c], 8, (0, 1))[0] for t in x]) / (x.shape[1] * x.shape[2]))
        feats.append(np.concatenate(f, 1))
    df = pd.DataFrame(np.concatenate(feats))
    df["photo"], df["sample_id"] = idx.photo, idx.sample_id
    return df.groupby(["sample_id", "photo"]).mean().groupby("sample_id").mean()


def write(name: str, oof: pd.DataFrame, test: pd.DataFrame | None, labels: pd.DataFrame, extra: dict) -> None:
    out = ROOT / "results/exps" / name
    out.mkdir(parents=True, exist_ok=True)
    oof = oof.sort_values("sample_id").reset_index(drop=True)
    oof[["sample_id", "fold"] + COLS].to_csv(out / "oof.csv", index=False)
    m = score(oof, labels)
    m.update(name=name, family="baseline", gpu="cpu", gpu_hours=0.0, peak_mem_gb=0.0, **extra)
    if test is not None:
        test = test.groupby("sample_id", as_index=False)[COLS].mean()
        test[COLS[-1]] = 100.0
        test.to_csv(out / "test.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    print(f"{name:28s} CV {m['cv_mean']:.3f} +- {m['cv_std']:.3f}")


def main() -> None:
    conf = yaml.safe_load((ROOT / "config.yaml").read_text())
    labels = pd.read_csv(ROOT / "results/data/labels.csv")
    labels.columns = ["sample_id"] + COLS
    folds = pd.read_csv(ROOT / "results/data/folds.csv")
    photos = pd.read_csv(ROOT / "results/data/photos_index.csv")
    y = labels.set_index("sample_id")[COLS]
    test_ids = sorted(photos.sample_id[photos.split == "test"].unique())
    k = int(folds.fold.max()) + 1

    def per_fold(fn):
        oof, test = [], []
        for f in range(k):
            va = folds.sample_id[folds.fold == f].tolist()
            tr = folds.sample_id[folds.fold != f].tolist()
            pv, pt = fn(tr, va, test_ids)
            oof.append(pd.DataFrame(pv, columns=COLS).assign(sample_id=va, fold=f))
            if len(test_ids):
                test.append(pd.DataFrame(pt, columns=COLS).assign(sample_id=test_ids))
        return pd.concat(oof), (pd.concat(test) if test else None)

    write("base_uniform", *per_fold(lambda tr, va, te: (np.tile(UNIFORM_BASELINE, (len(va), 1)),
                                                        np.tile(UNIFORM_BASELINE, (len(te), 1)))), labels, {})
    write("base_train_mean", *per_fold(lambda tr, va, te: (np.tile(y.loc[tr].mean().to_numpy(), (len(va), 1)),
                                                           np.tile(y.loc[tr].mean().to_numpy(), (len(te), 1)))),
          labels, {})

    # metadata shortcut: which cameras photographed the sample (and therefore its ppm), nothing else
    cams = pd.crosstab(photos.sample_id, photos.camera).clip(upper=1)

    def camera_only(tr, va, te):
        means = {}
        for c in cams.columns:
            with_c = [s for s in tr if cams.loc[s, c]]
            means[c] = y.loc[with_c].mean().to_numpy() if with_c else y.loc[tr].mean().to_numpy()
        pred = lambda sids: np.stack([np.mean([means[c] for c in cams.columns if cams.loc[s, c]], 0) for s in sids])
        return pred(va), (pred(te) if len(te) else None)
    write("ctrl_camera_only", *per_fold(camera_only), labels, {"control": "camera / ppm only"})

    col = colour_features(ROOT / conf["data"]["cache"] / cache_name(conf["data"]["tile_mm"]))

    def colour_only(tr, va, te):
        sets = [col.loc[va].to_numpy()] + ([col.loc[te].to_numpy()] if len(te) else [])
        preds, _ = ridge_fit_predict(col.loc[tr].to_numpy(), y.loc[tr].to_numpy(), sets)
        return preds[0], (preds[1] if len(te) else None)
    write("ctrl_colour_only", *per_fold(colour_only), labels, {"control": "colour statistics only"})


if __name__ == "__main__":
    main()
