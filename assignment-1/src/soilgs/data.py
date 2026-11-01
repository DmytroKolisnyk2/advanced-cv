"""Competition files -> photo table (sample, camera, ppm), labels, group-aware folds and tile caches."""

from __future__ import annotations

import json
import os
import re
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .metric import COLS, d_value
from .tiles import tile_photo

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _norm_col(c: str) -> str:
    try:
        return f"{float(c):g}"
    except ValueError:
        return c.strip()


def load_labels(data_dir: Path) -> pd.DataFrame:
    """Training curves, one row per sample, columns sample_id + COLS (percent passing)."""
    files = sorted(p for p in data_dir.rglob("*.csv") if "label" in p.name.lower() or p.name.lower() == "train.csv")
    df = pd.read_csv(files[0])
    df.columns = [_norm_col(c) for c in df.columns]
    df = df.rename(columns={df.columns[0]: "sample_id"})
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    return df[["sample_id"] + COLS].reset_index(drop=True)


def load_test_ids(data_dir: Path) -> list[str]:
    for name in ("test.csv", "sample_submission.csv"):
        hits = list(data_dir.rglob(name))
        if hits:
            return pd.read_csv(hits[0]).iloc[:, 0].astype(str).str.strip().tolist()
    raise FileNotFoundError("test.csv / sample_submission.csv")


def load_ppm(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(next(data_dir.rglob("ppm*.csv")))
    df.columns = [c.strip() for c in df.columns]
    return df


def _camera_col(ppm: pd.DataFrame) -> str:
    return next(c for c in ppm.columns if not pd.api.types.is_numeric_dtype(ppm[c]))


def _ppm_col(ppm: pd.DataFrame) -> str:
    num = [c for c in ppm.columns if pd.api.types.is_numeric_dtype(ppm[c])]
    return next((c for c in num if re.search(r"ppm|px.*mm|pixel", c, re.I)), num[0])


def _key(s: str) -> str:
    s = s.lower().replace("ü", "ue").replace("ö", "oe").replace("ä", "ae").replace("ß", "ss")
    return re.sub(r"[^a-z0-9]", "", s)


def _strip_photo_suffix(stem: str) -> str:
    """'iPhone14_HPC_Audorfring (3)' -> 'iPhone14_HPC_Audorfring', 'Motorola_Edge_F827_01' -> 'Motorola_Edge_F827'."""
    return re.sub(r"(\s*\(\d+\)|_\d{1,2})$", "", stem)


def index_photos(data_dir: Path, extra_ppm: dict[str, float] | None = None) -> pd.DataFrame:
    """Every photo with its split, sample_id, camera and pixels-per-mm.

    ppm.csv gives ppm at the camera's native resolution (width x height); photos stored at a lower resolution
    (e.g. 1600 px instead of 4000) get ppm scaled by the size ratio. Cameras missing from ppm.csv take their
    value from `extra_ppm` (pixels per mm at the stored resolution)."""
    labels, test_ids, ppm = load_labels(data_dir), load_test_ids(data_dir), load_ppm(data_dir)
    cam_col = _camera_col(ppm)
    ref = {_key(r[cam_col]): r for _, r in ppm.iterrows()}
    extra = {_key(k): v for k, v in (extra_ppm or {}).items()}
    cams = sorted(set(ref) | set(extra), key=len, reverse=True)
    ids = {**{_key(s): (s, "train") for s in labels.sample_id}, **{_key(s): (s, "test") for s in test_ids}}
    rows = []
    for p in sorted(data_dir.rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        key = _key(_strip_photo_suffix(p.stem))
        cam = next((c for c in cams if key.startswith(c)), None)
        rest = key[len(cam):] if cam else key
        sid, split = ids.get(rest, (None, None))
        with Image.open(p) as im:
            w, h = im.size
        if cam in ref:
            r = ref[cam]
            ppm_v = float(r["ppm"]) * max(w, h) / max(float(r["width"]), float(r["height"]))
        else:
            ppm_v = extra.get(cam, np.nan)
        rows.append({"photo": p.stem, "path": str(p), "sample_id": sid, "split": split, "camera": cam,
                     "width": w, "height": h, "ppm": ppm_v})
    df = pd.DataFrame(rows)
    bad = df[df.sample_id.isna() | df.camera.isna() | df.ppm.isna()]
    if len(bad):
        print(f"WARNING: dropping {len(bad)} photos without sample/camera/ppm: {bad.photo.head(8).tolist()}")
    return df.drop(bad.index).reset_index(drop=True)


def make_folds(labels: pd.DataFrame, k: int, seed: int = 0) -> pd.DataFrame:
    """Group-aware folds over physical samples, stratified by D50: sort by D50, deal each block of k
    consecutive samples to the k folds in random order (every fold gets fine and coarse soils)."""
    d50 = d_value(labels[COLS].to_numpy(), 50)
    order = np.argsort(d50, kind="stable")
    rng = np.random.default_rng(seed)
    fold = np.empty(len(labels), int)
    for start in range(0, len(order), k):
        block = order[start:start + k]
        fold[block] = rng.permutation(k)[:len(block)]
    return pd.DataFrame({"sample_id": labels.sample_id, "fold": fold, "log10_d50": d50})


def build_cache(photos: pd.DataFrame, out: Path, tile_mm: float, out_px: int, overlap: float, roi: float,
                max_tiles: int | None, ppm_error: float = 1.0, resolution: float = 1.0) -> None:
    """Tile every photo once; tiles.npy (uint8 [N,px,px,3]) + tiles.csv (photo, sample_id, split)."""
    out.mkdir(parents=True, exist_ok=True)
    if (out / "tiles.csv").exists():
        return
    jobs = [(r.path, r.ppm, tile_mm, out_px, overlap, roi, max_tiles, ppm_error, resolution,
             zlib.crc32(r.photo.encode())) for r in photos.itertuples()]
    with Pool(min(8, os.cpu_count() or 1)) as pool:
        chunks = list(tqdm(pool.imap(_tile_job, jobs), total=len(jobs), desc=out.name))
    rows = []
    for r, t in zip(photos.itertuples(), chunks):
        rows += [{"photo": r.photo, "sample_id": r.sample_id, "split": r.split, "camera": r.camera}] * len(t)
    np.save(out / "tiles.npy", np.concatenate(chunks))
    pd.DataFrame(rows).to_csv(out / "tiles.csv", index=False)
    (out / "meta.json").write_text(json.dumps({"tile_mm": tile_mm, "out_px": out_px, "overlap": overlap,
                                               "roi": roi, "max_tiles": max_tiles, "ppm_error": ppm_error,
                                               "resolution": resolution}))


def _tile_job(args) -> np.ndarray:
    *a, seed = args
    return tile_photo(*a, seed=seed)


def cache_name(tile_mm: float, ppm_error: float = 1.0, resolution: float = 1.0) -> str:
    name = f"tiles_{tile_mm:g}mm"
    if ppm_error != 1.0:
        name += f"_ppmerr{ppm_error:g}"
    if resolution != 1.0:
        name += f"_res{resolution:g}"
    return name
