"""Photo index, EDA statistics, group-aware folds and every tile cache (built once, reused by all experiments)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soilgs.data import build_cache, cache_name, index_photos, load_labels, load_ppm, make_folds  # noqa: E402
from soilgs.metric import COLS, d_value  # noqa: E402
from soilgs.tiles import load_calibrated  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def eda(photos: pd.DataFrame, labels: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sizes = []
    for p in photos.path:
        with Image.open(p) as im:
            sizes.append(im.size)
    photos = photos.assign(width=[s[0] for s in sizes], height=[s[1] for s in sizes])
    photos["mpix"] = photos.width * photos.height / 1e6
    photos["field_w_mm"] = photos.width / photos.ppm
    photos["field_h_mm"] = photos.height / photos.ppm
    photos.drop(columns=["path"]).to_csv(out / "photos.csv", index=False)
    y = labels[COLS].to_numpy(float)
    curves = labels.assign(fines_lt_0063=y[:, 3], gravel_gt_2=100 - y[:, 6],
                           log10_d50=d_value(y, 50), log10_d10=d_value(y, 10), log10_d60=d_value(y, 60))
    curves["class"] = np.select([curves.fines_lt_0063 >= 35, curves.gravel_gt_2 >= 50], ["fine", "gravel"], "sand")
    curves.to_csv(out / "curves.csv", index=False)
    summary = {
        "n_train_samples": int((photos.split == "train").groupby(photos.sample_id).any().sum()),
        "n_test_samples": int(photos[photos.split == "test"].sample_id.nunique()),
        "n_photos": photos.split.value_counts().to_dict(),
        "photos_per_sample": photos.groupby("sample_id").size().describe().round(2).to_dict(),
        "cameras": photos.groupby(["camera", "split"]).size().unstack(fill_value=0).to_dict(),
        "ppm_by_camera": photos.groupby("camera").ppm.first().round(3).to_dict(),
        "mpix_by_camera": photos.groupby("camera").mpix.median().round(1).to_dict(),
        "field_of_view_mm_median": [float(photos.field_w_mm.median()), float(photos.field_h_mm.median())],
        "class_counts": curves["class"].value_counts().to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    # thumbnails: one photo per camera and per class, plus calibrated crops at three physical scales
    th = out / "thumbs"
    th.mkdir(exist_ok=True)
    picks = pd.concat([photos.groupby("camera").head(1),
                       photos[photos.split == "train"].merge(curves[["sample_id", "class"]]).groupby("class").head(1)])
    for r in picks.drop_duplicates("photo").itertuples():
        with Image.open(r.path) as im:
            im.draft("RGB", (1024, 1024))
            im = im.convert("RGB")
            im.thumbnail((768, 768))
            im.save(th / f"{r.photo}.jpg", quality=85)
        img = load_calibrated(r.path, r.ppm, 224 / 20)  # 11.2 px/mm
        h, w = img.shape[:2]
        for mm in (10, 20, 40):
            t = int(mm * 224 / 20)
            crop = img[h // 2 - t // 2: h // 2 + t // 2, w // 2 - t // 2: w // 2 + t // 2]
            Image.fromarray(crop).resize((224, 224), Image.BICUBIC).save(th / f"{r.photo}__{mm}mm.jpg", quality=90)


def main() -> None:
    conf = yaml.safe_load((ROOT / "config.yaml").read_text())
    d = conf["data"]
    data_dir = ROOT / d["dir"]
    res = ROOT / "results/data"
    res.mkdir(parents=True, exist_ok=True)
    photos = index_photos(data_dir)
    photos.to_csv(res / "photos_index.csv", index=False)
    labels = load_labels(data_dir)
    labels.to_csv(res / "labels.csv", index=False)
    load_ppm(data_dir).to_csv(res / "ppm.csv", index=False)
    sample_sub = next(data_dir.rglob("sample_submission*.csv"), None)
    if sample_sub is not None:  # exact header / row order expected by Kaggle
        shutil.copy(sample_sub, res / "sample_submission.csv")
    folds = make_folds(labels, d["folds"], d["fold_seed"])
    folds.to_csv(res / "folds.csv", index=False)
    if "--eda" in sys.argv or not (ROOT / "results/eda/summary.json").exists():
        eda(photos, labels, ROOT / "results/eda")
    if "--no-cache" in sys.argv:
        return
    cache = ROOT / d["cache"]
    for mm in [d["tile_mm"]] + d["tile_mm_ablation"]:
        build_cache(photos, cache / cache_name(mm), mm, d["out_px"], d["overlap"], d["roi"], d["max_tiles"])
    rb = conf["robustness"]
    train = photos[photos.split == "train"]
    for e in rb["ppm_error"]:
        build_cache(train, cache / cache_name(d["tile_mm"], ppm_error=e), d["tile_mm"], d["out_px"], 0.0, d["roi"],
                    rb["max_tiles"], ppm_error=e)
    for r in rb["resolution"]:
        build_cache(train, cache / cache_name(d["tile_mm"], resolution=r), d["tile_mm"], d["out_px"], 0.0, d["roi"],
                    rb["max_tiles"], resolution=r)
    # reference for the robustness curves: same protocol (no overlap, 12 tiles), correct scale
    build_cache(train, cache / (cache_name(d["tile_mm"]) + "_ref"), d["tile_mm"], d["out_px"], 0.0, d["roi"],
                rb["max_tiles"])


if __name__ == "__main__":
    main()
