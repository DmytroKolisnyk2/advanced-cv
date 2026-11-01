"""Inference: photo(s) of one soil sample + pixels-per-mm -> valid grain-size curve.

    uv run python scripts/predict.py --ppm 21.4 photo1.jpg photo2.jpg --exps cnn_convnext_t_adamw,vit_swin_t

Uses every saved fold checkpoint of each experiment (checkpoints/<exp>_fold<k>.pt) and averages the curves.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soilgs.metric import COLS, make_valid  # noqa: E402
from soilgs.models import add_lora, build_model, logits_to_curve  # noqa: E402
from soilgs.tiles import tile_photo  # noqa: E402
from soilgs.train import DEVICE, normalize  # noqa: E402


@torch.no_grad()
def predict(photos: list[str], ppm: list[float], exps: list[str]) -> np.ndarray:
    conf = yaml.safe_load((ROOT / "config.yaml").read_text())
    d = conf["data"]
    curves = []
    for name in exps:
        cfg = json.loads((ROOT / "results/exps" / name / "config.json").read_text())
        tile_mm = cfg.get("tile_mm", d["tile_mm"])
        tiles = [tile_photo(p, s, tile_mm, d["out_px"], d["overlap"], d["roi"], d["max_tiles"]) for p, s in zip(photos, ppm)]
        cfg.update(pretrained=False, init_ckpt=None)
        for ck in sorted((ROOT / "checkpoints").glob(f"{name}_fold*.pt")):
            model = build_model(cfg)
            if cfg.get("mode") == "lora":
                add_lora(model, cfg.get("lora_r", 8), cfg.get("lora_alpha", 16))
            model.load_state_dict(torch.load(ck, map_location="cpu"))
            model.to(DEVICE).eval()
            per_photo = []
            for t in tiles:
                x = torch.from_numpy(t).permute(0, 3, 1, 2).float().div(255).to(DEVICE)
                if cfg.get("img") and cfg["img"] != x.shape[-1]:
                    x = torch.nn.functional.interpolate(x, size=(cfg["img"],) * 2, mode="bilinear", antialias=True)
                per_photo.append(logits_to_curve(model(normalize(x))).mean(0).cpu().numpy())
            curves.append(np.mean(per_photo, 0))
    return make_valid(np.mean(curves, 0))[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--ppm", type=float, nargs="+", required=True, help="pixels per mm (one value or one per photo)")
    ap.add_argument("--exps", required=True)
    a = ap.parse_args()
    ppm = a.ppm * len(a.photos) if len(a.ppm) == 1 else a.ppm
    curve = predict(a.photos, ppm, a.exps.split(","))
    print(",".join(COLS))
    print(",".join(f"{v:.2f}" for v in curve))


if __name__ == "__main__":
    main()
