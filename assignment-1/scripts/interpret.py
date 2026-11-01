"""Grad-CAM (CNNs, hierarchical ViTs) and attention rollout (plain ViTs / SSL encoders) on the same validation
photos for the best model of each family (fold-0 models, so every photo is unseen by its model)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soilgs.metric import COLS, LOG_D, d_value  # noqa: E402
from soilgs.models import add_lora, build_model  # noqa: E402
from soilgs.tiles import load_calibrated  # noqa: E402
from soilgs.train import DEVICE, normalize  # noqa: E402

# log10 centre of each of the 11 mass bins; their probability-weighted mean = predicted mean grain size
BIN_CENTRES = torch.tensor(np.concatenate([[LOG_D[0] - 0.25], (LOG_D[:-1] + LOG_D[1:]) / 2]), dtype=torch.float32)


def load_fold0(name: str) -> torch.nn.Module:
    cfg = json.loads((ROOT / "results/exps" / name / "config.json").read_text())
    cfg["pretrained"] = False
    if cfg.get("init_ckpt"):
        cfg["init_ckpt"] = None
    model = build_model(cfg)
    if cfg.get("mode") == "lora":
        add_lora(model, cfg.get("lora_r", 8), cfg.get("lora_alpha", 16))
    sd = torch.load(ROOT / "checkpoints" / f"{name}_fold0.pt", map_location="cpu")
    model.load_state_dict(sd)
    return model.to(DEVICE).eval()


def coarseness(logits: torch.Tensor) -> torch.Tensor:
    return (logits.float().softmax(-1) * BIN_CENTRES.to(logits.device)).sum(-1)


def grad_cam(model, x: torch.Tensor) -> np.ndarray:
    """Grad-CAM of the predicted mean log-diameter w.r.t. the last feature map (positive = 'coarser')."""
    fmap = model.forward_features(x)
    nhwc = fmap.ndim == 4 and fmap.shape[-1] == model.num_features and fmap.shape[1] != model.num_features
    if nhwc:
        fmap = fmap.permute(0, 3, 1, 2)  # NHWC (Swin) -> NCHW
    fmap.retain_grad()
    out = model.forward_head(fmap.permute(0, 2, 3, 1) if nhwc else fmap)
    coarseness(out).sum().backward()
    a, g = fmap, fmap.grad
    cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return (cam / (cam.amax((1, 2), keepdim=True) + 1e-8)).detach().cpu().numpy()


def rollout(model, x: torch.Tensor) -> np.ndarray:
    """Attention rollout: average heads, add identity, renormalise, multiply over layers; [CLS] row on patches."""
    maps = []
    hooks = []
    for blk in model.blocks:
        blk.attn.fused_attn = False
        hooks.append(blk.attn.attn_drop.register_forward_hook(lambda m, i, o: maps.append(i[0].float().mean(1))))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    n = maps[0].shape[-1]
    r = torch.eye(n, device=x.device).expand(len(x), n, n)
    for a in maps:
        a = a + torch.eye(n, device=x.device)
        r = (a / a.sum(-1, keepdim=True)) @ r
    npt = getattr(model, "num_prefix_tokens", 1)
    cls = r[:, 0, npt:]
    s = int(cls.shape[-1] ** 0.5)
    cam = cls.reshape(len(x), 1, s, s)
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam - cam.amin((1, 2), keepdim=True)
    return (cam / (cam.amax((1, 2), keepdim=True) + 1e-8)).cpu().numpy()


def explain(model, tiles: np.ndarray, img: int | None) -> tuple[np.ndarray, np.ndarray]:
    x = torch.from_numpy(tiles).permute(0, 3, 1, 2).float().div(255).to(DEVICE)
    if img and img != x.shape[-1]:
        x = F.interpolate(x, size=(img, img), mode="bilinear", antialias=True)
    x = normalize(x)
    plain_vit = hasattr(model, "blocks") and hasattr(model, "cls_token") or type(model).__name__ in ("TinyViT", "Eva")
    m = rollout(model, x) if plain_vit else grad_cam(model, x)
    with torch.no_grad():
        curve = (100 * torch.cumsum(model(x).float().softmax(-1), -1)).mean(0).cpu().numpy()
    if m.shape[-1] != tiles.shape[1]:
        m = F.interpolate(torch.from_numpy(m)[:, None], size=tiles.shape[1:3], mode="bilinear")[:, 0].numpy()
    return m, curve


def main() -> None:
    conf = yaml.safe_load((ROOT / "config.yaml").read_text())
    tile_mm, px = conf["data"]["tile_mm"], conf["data"]["out_px"]
    metrics = [json.loads(p.read_text()) for p in (ROOT / "results/exps").glob("*/metrics.json")]
    cfg_of = lambda n: json.loads((ROOT / "results/exps" / n / "config.json").read_text())
    trained = [m for m in metrics if (ROOT / "checkpoints" / f"{m['name']}_fold0.pt").exists()
               and cfg_of(m["name"]).get("tile_mm", tile_mm) == tile_mm]
    best = {}
    for fam in ("cnn", "vit", "ssl"):
        cands = [m for m in trained if m.get("family") == fam]
        if cands:
            best[fam] = min(cands, key=lambda m: m["cv_mean"])["name"]
    print("maps for", best)
    labels = pd.read_csv(ROOT / "results/data/labels.csv")
    labels.columns = ["sample_id"] + COLS
    folds = pd.read_csv(ROOT / "results/data/folds.csv")
    photos = pd.read_csv(ROOT / "results/data/photos_index.csv")
    val = folds.sample_id[folds.fold == 0].tolist()
    ref = json.loads((ROOT / "results/exps" / best["cnn"] / "metrics.json").read_text())["per_sample"]
    order = sorted(val, key=lambda s: ref[s])
    chosen = [order[0], order[len(order) // 2], order[-1]]  # best, median, worst (failure case)
    models = {f: load_fold0(n) for f, n in best.items()}
    imgs = {f: json.loads((ROOT / "results/exps" / n / "config.json").read_text()).get("img") for f, n in best.items()}
    fig, axes = plt.subplots(len(chosen), 1 + len(models), figsize=(4 * (1 + len(models)), 4 * len(chosen)))
    axes = np.atleast_2d(axes)
    record = []
    for r, sid in enumerate(chosen):
        ph = photos[photos.sample_id == sid].iloc[0]
        img = load_calibrated(ph.path, ph.ppm, px / tile_mm)
        h, w = img.shape[:2]
        n = 3 if min(h, w) >= 3 * px else 1
        y0, x0 = h // 2 - n * px // 2, w // 2 - n * px // 2
        region = img[y0:y0 + n * px, x0:x0 + n * px]
        tiles = np.stack([region[i * px:(i + 1) * px, j * px:(j + 1) * px] for i in range(n) for j in range(n)])
        true = labels.set_index("sample_id").loc[sid, COLS].to_numpy(float)
        axes[r, 0].imshow(region)
        axes[r, 0].set_title(f"{sid} ({ph.camera})\ntrue D50={10 ** d_value(true, 50)[0]:.3g} mm, "
                             f"{n * tile_mm:g} mm wide", fontsize=9)
        for c, (fam, model) in enumerate(models.items(), start=1):
            m, curve = explain(model, tiles, imgs[fam])
            full = np.block([[m[i * n + j] for j in range(n)] for i in range(n)])
            axes[r, c].imshow(region)
            axes[r, c].imshow(full, cmap="jet", alpha=0.45)
            kind = "rollout" if type(model).__name__ in ("VisionTransformer", "TinyViT", "Eva") else "Grad-CAM"
            axes[r, c].set_title(f"{fam}: {best[fam]}\n{kind}, pred D50={10 ** d_value(curve, 50)[0]:.3g} mm",
                                 fontsize=9)
            record.append({"sample_id": sid, "family": fam, "model": best[fam],
                           "pred_d50_mm": float(10 ** d_value(curve, 50)[0]),
                           "true_d50_mm": float(10 ** d_value(true, 50)[0]),
                           "map_centre_vs_border": float(full[n * px // 4: 3 * n * px // 4, n * px // 4: 3 * n * px // 4].mean()
                                                         / (full.mean() + 1e-8))})
        for a in axes[r]:
            a.axis("off")
    out = ROOT / "results/figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "interpretation_maps.png", dpi=110)
    pd.DataFrame(record).to_csv(ROOT / "results/interpretation.csv", index=False)
    print(pd.DataFrame(record))


if __name__ == "__main__":
    main()
