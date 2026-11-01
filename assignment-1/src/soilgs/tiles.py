"""Physically calibrated tiling: every tile covers tile_mm x tile_mm of soil and is resampled to out_px."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None


def load_calibrated(path: str, ppm: float, target_ppm: float) -> np.ndarray:
    """Decode a photo at target_ppm pixels per mm (uint8 HxWx3).

    JPEG draft mode decodes at 1/2, 1/4 or 1/8 directly (much faster for 48 MP photos); the exact
    resize afterwards accounts for the draft factor, so the physical scale stays exact.
    """
    img = Image.open(path)
    w0, h0 = img.size
    scale = target_ppm / ppm
    if img.format == "JPEG" and scale < 1:
        img.draft("RGB", (max(1, int(w0 * scale)), max(1, int(h0 * scale))))
    eff_ppm = ppm * img.size[0] / w0  # draft() resizes at once and keeps the aspect ratio
    img = ImageOps.exif_transpose(img).convert("RGB")
    s = target_ppm / eff_ppm
    new = (max(1, round(img.size[0] * s)), max(1, round(img.size[1] * s)))
    if new != img.size:
        img = img.resize(new, Image.LANCZOS if s > 1 else Image.BOX, reducing_gap=None)
    return np.asarray(img)


def roi_box(h: int, w: int, roi: float) -> tuple[int, int, int, int]:
    """Central region keeping `roi` of each side (drops tray edges / rulers near the borders)."""
    dy, dx = int(h * (1 - roi) / 2), int(w * (1 - roi) / 2)
    return dy, dx, h - dy, w - dx


def grid_tiles(img: np.ndarray, tile_px: int, stride_px: int, roi: float = 1.0,
               max_tiles: int | None = None, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Cut a regular grid of tile_px tiles inside the ROI. Returns (tiles [N,t,t,3], yx [N,2])."""
    y0, x0, y1, x1 = roi_box(*img.shape[:2], roi)
    ys = list(range(y0, y1 - tile_px + 1, stride_px)) or [max(0, (img.shape[0] - tile_px) // 2)]
    xs = list(range(x0, x1 - tile_px + 1, stride_px)) or [max(0, (img.shape[1] - tile_px) // 2)]
    yx = np.array([(y, x) for y in ys for x in xs])
    if max_tiles and len(yx) > max_tiles:
        keep = np.random.default_rng(seed).choice(len(yx), max_tiles, replace=False)
        yx = yx[np.sort(keep)]
    tiles = np.stack([_crop(img, y, x, tile_px) for y, x in yx])
    return tiles, yx


def _crop(img: np.ndarray, y: int, x: int, t: int) -> np.ndarray:
    tile = img[y:y + t, x:x + t]
    if tile.shape[:2] != (t, t):  # photo smaller than a tile: reflect-pad
        tile = np.pad(tile, ((0, t - tile.shape[0]), (0, t - tile.shape[1]), (0, 0)), mode="reflect")
    return tile


def tile_photo(path: str, ppm: float, tile_mm: float, out_px: int = 224, overlap: float = 0.0,
               roi: float = 1.0, max_tiles: int | None = None, ppm_error: float = 1.0,
               resolution: float = 1.0, seed: int = 0) -> np.ndarray:
    """Tiles of one photo.

    ppm_error  - the pipeline believes ppm * ppm_error (a mis-calibrated scale): each tile then covers
                 tile_mm * ppm_error of real soil, so grains look ppm_error times smaller.
    resolution - the photo was taken at `resolution` x the pixel density with the correct ppm: tiles cover the
                 right physical area but carry less detail (down- then up-sampled).
    """
    target_ppm = out_px / tile_mm
    img = load_calibrated(path, ppm * ppm_error, target_ppm * resolution)
    t = max(8, round(out_px * resolution))
    tiles, _ = grid_tiles(img, t, max(1, round(t * (1 - overlap))), roi, max_tiles, seed)
    if t != out_px:
        tiles = np.stack([np.asarray(Image.fromarray(x).resize((out_px, out_px), Image.BICUBIC)) for x in tiles])
    return tiles
