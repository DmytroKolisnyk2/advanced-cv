"""Label-free check of a camera's pixels-per-mm (Samsung A52) against ppm_updated.csv.

Almost every training soil was photographed with both the Motorola Edge (known ppm) and the Samsung A52, from the
same 21 cm. The same soil at the right scale must have the same texture spectrum in cycles per mm. For a candidate
Samsung ppm s we resample its photos to a common grid, compute the radially averaged power spectrum (shape only,
normalised), and compare it with the Motorola spectrum of the same sample. The s with the smallest mismatch wins.
A self-check rescales Motorola photos by known factors and checks that the method recovers them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soilgs.data import index_photos  # noqa: E402
from soilgs.tiles import load_calibrated, roi_box  # noqa: E402

GRID_PPM = 2.5            # common resolution (below every candidate, so always a downsampling)
FREQ = np.geomspace(0.06, 1.0, 24)  # cycles per mm (Nyquist at 2.5 px/mm = 1.25)


def spectrum(img: np.ndarray, roi: float = 0.6) -> np.ndarray:
    g = img.astype(np.float32).mean(-1)
    y0, x0, y1, x1 = roi_box(*g.shape, roi)
    g = g[y0:y1, x0:x1]
    g = (g - g.mean()) / (g.std() + 1e-6)
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    p = np.abs(np.fft.fftshift(np.fft.fft2(g * win))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(g.shape[0], d=1 / GRID_PPM))
    fx = np.fft.fftshift(np.fft.fftfreq(g.shape[1], d=1 / GRID_PPM))
    r = np.hypot(*np.meshgrid(fy, fx, indexing="ij"))
    edges = np.concatenate([[FREQ[0] / 1.06], np.sqrt(FREQ[1:] * FREQ[:-1]), [FREQ[-1] * 1.06]])
    out = np.array([p[(r >= a) & (r < b)].mean() for a, b in zip(edges[:-1], edges[1:])])
    out = np.log(out + 1e-9)
    return out - out.mean()


def sample_spectra(photos, ppm_scale: float = 1.0) -> dict[str, np.ndarray]:
    out = {}
    for sid, g in photos.groupby("sample_id"):
        out[sid] = np.mean([spectrum(load_calibrated(r.path, r.ppm * ppm_scale, GRID_PPM)) for r in g.itertuples()], 0)
    return out


def mismatch(a: dict, b: dict) -> float:
    common = sorted(set(a) & set(b))
    return float(np.mean([np.mean((a[s] - b[s]) ** 2) for s in common]))


def main() -> None:
    conf = yaml.safe_load((ROOT / "config.yaml").read_text())
    photos = index_photos(ROOT / conf["data"]["dir"], {"Samsung A52": 1.0})  # placeholder ppm, scanned below
    ref_cam, new_cam = "motorolaedge", "samsunga52"
    moto = photos[photos.camera == ref_cam]
    sams = photos[photos.camera == new_cam]
    official = float(sams.ppm.iloc[0])  # ppm_updated.csv scaled to the stored photo size
    ref = sample_spectra(moto)
    grid = np.round(np.arange(3.0, 8.01, 0.1), 2)
    curve = [mismatch(ref, sample_spectra(sams.assign(ppm=s))) for s in grid]
    best = float(grid[int(np.argmin(curve))])
    # self-check on different photos of the same soils: Motorola photo #1 (true ppm) vs the other Motorola photos
    # with a pretended ppm factor k; the minimum must sit at k = 1
    first = moto.groupby("sample_id").head(1)
    rest = moto.drop(first.index)
    ref1 = sample_spectra(first)
    check_grid = np.round(np.arange(0.6, 1.61, 0.05), 2)
    check = [mismatch(ref1, sample_spectra(rest, ppm_scale=k)) for k in check_grid]
    res = {"camera": "Samsung A52", "estimated_ppm": best, "grid": grid.tolist(), "mismatch": curve,
           "self_check_scale_grid": check_grid.tolist(), "self_check_mismatch": check,
           "self_check_best": float(check_grid[int(np.argmin(check))]),
           "optics_prior_ppm": round(1599 / 290.0, 2), "table_ppm": round(official, 3),
           "n_pairs": len(set(ref) & set(sams.sample_id))}
    out = ROOT / "results/calibration"
    out.mkdir(parents=True, exist_ok=True)
    (out / "samsung_ppm.json").write_text(json.dumps(res, indent=1))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(grid, curve, "o-", ms=3)
    ax[0].axvline(best, color="r", ls="--", label=f"best {best}")
    ax[0].axvline(res["optics_prior_ppm"], color="g", ls=":", label=f"optics prior {res['optics_prior_ppm']}")
    ax[0].axvline(official, color="k", ls="-.", label=f"ppm_updated.csv {official:.2f}")
    ax[0].set(xlabel="Samsung A52 ppm candidate", ylabel="spectrum mismatch vs Motorola (same soils)")
    ax[0].legend()
    ax[1].plot(check_grid, check, "o-", ms=3)
    ax[1].set(xlabel="pretended ppm factor on Motorola photos", ylabel="mismatch", title="self-check (true = 1.0)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "samsung_ppm.png", dpi=110)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=1))


if __name__ == "__main__":
    main()
