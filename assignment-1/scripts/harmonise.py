"""Block 8, H8.1: test-time harmonisation. Put the test photos through the same path as the training photos
(1600 px long side + JPEG at the quality measured on the training files), re-tile them with the same pipeline and
re-predict with the already trained models (no retraining, so CV is unchanged).

Also measures the domain gap without labels (DINOv3 feature distance to the nearest training photo, sharpness) and
checks the effect on H374, the only training soil photographed at native resolution.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soilgs.data import build_cache, cache_name  # noqa: E402
from soilgs.frozen import extract, knn_fit_predict, ridge_fit_predict, sample_means  # noqa: E402
from soilgs.metric import COLS, emd  # noqa: E402
from soilgs.models import add_lora, build_model  # noqa: E402
from soilgs.train import DEVICE, TileStore, aggregate, predict_tiles  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
EXPS = ROOT / "results/exps"
TRAINED = ["ssl_dinov3_s_ft", "vit_scratch", "cnn_r18_bs8_grouped_frozenbn"]
FROZEN = {"dinov3_s_knn_clspatch": ("vit_small_patch16_dinov3.lvd1689m", None, "dinov3_s", "knn", "cls+patch"),
          "dinov2_s_da_ridge_block4": ("vit_small_patch14_dinov2.lvd142m", "checkpoints/da_dinov2_s/backbone.pt",
                                       "dinov2_s_da", "ridge", "block4")}
ENSEMBLE = TRAINED + list(FROZEN)


def jpeg_quality(path: str) -> int:
    """IJG quality whose scaled luminance table is closest to the file's (standard libjpeg scaling)."""
    base = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                     14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
                     92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], float)
    with Image.open(path) as im:
        q = np.array(sorted(im.quantization[0]), float) if hasattr(im, "quantization") else None
    if q is None:
        return 95
    best, err = 95, 1e18
    for quality in range(10, 101):
        scale = 5000 / quality if quality < 50 else 200 - 2 * quality
        t = np.clip(np.floor((base * scale + 50) / 100), 1, 255)
        e = np.abs(np.sort(t) - q).sum()
        if e < err:
            best, err = quality, e
    return best


def messenger(src: str, dst: Path, long_side: int, quality: int) -> float:
    """Resize to `long_side` and re-encode as JPEG. Returns the size ratio (new / old long side)."""
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    r = long_side / max(im.size)
    im = im.resize((round(im.size[0] * r), round(im.size[1] * r)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    dst.write_bytes(buf.getvalue())
    return r


def sharpness(store: TileStore, ids: np.ndarray) -> float:
    x = store.batch(ids[:2000]).mean(1, keepdim=True)
    lap = torch.nn.functional.conv2d(x, torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=x.dtype, device=x.device))
    return float(lap.abs().mean())


def load_trained(name: str, fold: int):
    cfg = json.loads((EXPS / name / "config.json").read_text())
    cfg.update(pretrained=False, init_ckpt=None)
    model = build_model(cfg)
    if cfg.get("mode") == "lora":
        add_lora(model, cfg.get("lora_r", 8), cfg.get("lora_alpha", 16))
    model.load_state_dict(torch.load(ROOT / "checkpoints" / f"{name}_fold{fold}.pt", map_location="cpu"))
    return model.to(DEVICE).eval(), cfg


def write_exp(name: str, test: pd.DataFrame, extra: dict) -> None:
    """<name>_tth: the original OOF/metrics (CV unchanged) with harmonised test predictions."""
    out = EXPS / f"{name}_tth"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(EXPS / name / "oof.csv", out / "oof.csv")
    if (EXPS / name / "config.json").exists():
        shutil.copy(EXPS / name / "config.json", out / "config.json")
    m = json.loads((EXPS / name / "metrics.json").read_text())
    m.update(name=f"{name}_tth", test_time="messenger harmonisation of test photos", **extra)
    test = test.groupby("sample_id", as_index=False)[COLS].mean()
    test[COLS[-1]] = 100.0
    test.to_csv(out / "test.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(m, indent=1))


def main() -> None:
    conf = yaml.safe_load((ROOT / "config.yaml").read_text())
    d = conf["data"]
    photos = pd.read_csv(ROOT / "results/data/photos_index.csv")
    labels = pd.read_csv(ROOT / "results/data/labels.csv")
    labels.columns = ["sample_id"] + COLS
    folds = pd.read_csv(ROOT / "results/data/folds.csv")
    fold_of = dict(zip(folds.sample_id, folds.fold))
    k = int(folds.fold.max()) + 1

    # 1. the training path: long side and JPEG quality of the messenger-compressed training photos
    train = photos[photos.split == "train"]
    comp = train[train.camera.isin(["motorolaedge", "samsunga52"])]
    long_side = int(np.median(np.maximum(comp.width, comp.height)))
    qualities = {c: int(np.median([jpeg_quality(p) for p in g.path.head(10)])) for c, g in comp.groupby("camera")}
    quality = int(np.median(list(qualities.values())))
    print("training path:", long_side, "px, JPEG quality", qualities)

    # 2. harmonised copies of the test photos and of H374 (native-resolution training soil)
    hdir = ROOT / "data/harmonised"
    hdir.mkdir(parents=True, exist_ok=True)
    todo = photos[(photos.split == "test") | (photos.camera == "motorolaedge60fusion")].copy()
    rows = []
    for r in todo.itertuples():
        dst = hdir / f"{r.photo}.jpg"
        ratio = messenger(r.path, dst, long_side, quality)
        rows.append({**r._asdict(), "path": str(dst), "ppm": r.ppm * ratio})
    hphotos = pd.DataFrame(rows).drop(columns=["Index"])
    hname = cache_name(d["tile_mm"]) + "_harmonised"
    build_cache(hphotos, ROOT / d["cache"] / hname, d["tile_mm"], d["out_px"], d["overlap"], d["roi"], d["max_tiles"])
    main_store = TileStore(ROOT / d["cache"] / cache_name(d["tile_mm"]), DEVICE)
    h_store = TileStore(ROOT / d["cache"] / hname, DEVICE)
    mi, hi = main_store.index, h_store.index
    test_ids, htest_ids = np.where(mi.split == "test")[0], np.where(hi.split == "test")[0]
    h374, hh374 = np.where(mi.sample_id == "H374")[0], np.where(hi.sample_id == "H374")[0]
    comp_ids = np.where(mi.camera.isin(["motorolaedge", "samsunga52"]))[0]

    # 3. label-free domain gap: sharpness and DINOv3 feature distance to the nearest training photo
    gap = {"jpeg_quality": qualities, "long_side": long_side,
           "sharpness": {"train (messenger)": sharpness(main_store, comp_ids), "test original": sharpness(main_store, test_ids),
                         "test harmonised": sharpness(h_store, htest_ids)}}
    enc = build_model({"arch": "vit_small_patch16_dinov3.lvd1689m", "pretrained": True}).to(DEVICE)
    fm = extract(enc, main_store)["cls+patch"]
    fh = extract(enc, h_store)["cls+patch"]
    pm = sample_means(mi.assign(sample_id=mi.photo), fm)  # per photo
    ph = sample_means(hi.assign(sample_id=hi.photo), fh)
    tr_photos = mi.photo[mi.split == "train"].unique()
    te_photos = mi.photo[mi.split == "test"].unique()
    a = pm.loc[tr_photos].to_numpy()
    a = a / np.linalg.norm(a, axis=1, keepdims=True)

    def nn_dist(b):
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        return float((1 - (b @ a.T).max(1)).mean())
    gap["dinov3_nn_cosine_distance"] = {"test original": nn_dist(pm.loc[te_photos].to_numpy()),
                                        "test harmonised": nn_dist(ph.loc[te_photos].to_numpy())}
    del enc
    print(json.dumps(gap, indent=1))

    # 4. trained models: re-predict harmonised test tiles with every fold model; H374 check with its fold model
    h374_rows = []
    for name in TRAINED:
        parts = []
        for f in range(k):
            model, cfg = load_trained(name, f)
            img = cfg.get("img")
            parts.append(aggregate(hi.iloc[htest_ids], predict_tiles(model, h_store, htest_ids, img))[1])
            if fold_of["H374"] == f:
                y = labels.set_index("sample_id").loc[["H374"], COLS].to_numpy()
                e0 = emd(y, aggregate(mi.iloc[h374], predict_tiles(model, main_store, h374, img))[1][COLS].to_numpy())[0]
                e1 = emd(y, aggregate(hi.iloc[hh374], predict_tiles(model, h_store, hh374, img))[1][COLS].to_numpy())[0]
                h374_rows.append({"model": name, "H374_original": round(float(e0), 2), "H374_harmonised": round(float(e1), 2)})
            del model
        write_exp(name, pd.concat(parts), {})

    # 5. frozen members: refit per fold on the cached training features, predict harmonised test features
    targets = labels.set_index("sample_id")[COLS]
    for name, (arch, ckpt, enc_name, head, fset) in FROZEN.items():
        cached = np.load(ROOT / "cache/features" / f"{enc_name}.npz")[f"main_{fset}"]
        enc = build_model({"arch": arch, "pretrained": True, "init_ckpt": str(ROOT / ckpt) if ckpt else None}).to(DEVICE)
        fh = extract(enc, h_store)[fset]
        del enc
        tr_means = sample_means(mi, cached)
        te_means = sample_means(hi.iloc[htest_ids], fh[htest_ids])
        parts = []
        for f in range(k):
            tr = sorted(s for s, g in fold_of.items() if g != f)
            fn = knn_fit_predict if head == "knn" else ridge_fit_predict
            (p,), _ = fn(tr_means.loc[tr].to_numpy(), targets.loc[tr].to_numpy(), [te_means.to_numpy()])
            parts.append(pd.DataFrame(p, columns=COLS).assign(sample_id=list(te_means.index)))
        write_exp(name, pd.concat(parts), {})

    # 6. the final ensemble with harmonised members
    tests = [pd.read_csv(EXPS / f"{n}_tth" / "test.csv").sort_values("sample_id").reset_index(drop=True) for n in ENSEMBLE]
    ens = tests[0][["sample_id"]].copy()
    ens[COLS] = np.mean([t[COLS].to_numpy() for t in tests], 0)
    ens[COLS[-1]] = 100.0
    out = EXPS / "ens_top5_tth"
    out.mkdir(exist_ok=True)
    shutil.copy(EXPS / "ens_top5" / "oof.csv", out / "oof.csv")
    ens.to_csv(out / "test.csv", index=False)
    m = json.loads((EXPS / "ens_top5" / "metrics.json").read_text())
    m.update(name="ens_top5_tth", members=[f"{n}_tth" for n in ENSEMBLE], test_time="messenger harmonisation")
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    gap["H374_check"] = h374_rows
    # how much the test predictions moved (mean EMD between original and harmonised test curves)
    gap["test_shift_emd"] = {n: float(emd(pd.read_csv(EXPS / n / "test.csv").sort_values("sample_id")[COLS].to_numpy(),
                                          pd.read_csv(EXPS / f"{n}_tth" / "test.csv").sort_values("sample_id")[COLS].to_numpy()).mean())
                             for n in ENSEMBLE + ["ens_top5"] if (EXPS / f"{n}_tth").exists() or n == "ens_top5"}
    (ROOT / "results/harmonisation.json").write_text(json.dumps(gap, indent=1))
    print(json.dumps({k: v for k, v in gap.items() if k != "jpeg_quality"}, indent=1))


if __name__ == "__main__":
    main()
