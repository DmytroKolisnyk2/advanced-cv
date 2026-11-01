"""Continued DINO-style self-distillation of a pretrained ViT on unlabelled training tiles (domain adaptation).

Scale-safe multi-crop: global views are whole cached tiles, local views are smaller crops cut WITHOUT resizing,
so every view keeps the physical mm-per-pixel of the photo. Only dihedral flips/rotations and mild colour
jitter are used; random-resized crops would change the apparent grain size, which is the label.
"""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import build_model, gpu_augment
from .train import TileStore, gpu_name, normalize, peak_gb, reset_peak


class DINOHead(nn.Module):
    def __init__(self, dim: int, out: int = 4096, hidden: int = 1024, bottleneck: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, bottleneck))
        self.last = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck, out, bias=False))
        self.last.parametrizations.weight.original0.data.fill_(1)
        self.last.parametrizations.weight.original0.requires_grad = False

    def forward(self, x):
        return self.last(F.normalize(self.mlp(x), dim=-1))


class Wrap(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, x):
        return self.head(self.backbone.forward_head(self.backbone.forward_features(x), pre_logits=True))


def local_crops(x: torch.Tensor, n: int, size: int) -> list[torch.Tensor]:
    out = []
    for _ in range(n):
        y0, x0 = np.random.randint(0, x.shape[-1] - size + 1, 2)
        out.append(x[:, :, y0:y0 + size, x0:x0 + size])
    return out


def dino_adapt(cfg: dict, store: TileStore, out_dir: Path, probe=None) -> dict:
    """Returns the training log; writes the adapted backbone state dict (head removed) to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = store.x.device
    torch.manual_seed(cfg.get("seed", 0))
    np.random.seed(cfg.get("seed", 0))
    backbone = build_model({**cfg, "pretrained": True})
    backbone.reset_classifier(0)
    student = Wrap(backbone, DINOHead(backbone.num_features, cfg.get("out_dim", 4096))).to(dev)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad = False
    center = torch.zeros(1, cfg.get("out_dim", 4096), device=dev)
    steps, bs, n_local = cfg["steps"], cfg["batch"], cfg.get("n_local", 4)
    head_only = cfg.get("head_warmup", 200)
    opt = torch.optim.AdamW([
        {"params": student.backbone.parameters(), "lr": cfg["lr"], "name": "backbone"},
        {"params": [p for p in student.head.parameters() if p.requires_grad], "lr": cfg.get("head_lr", 5e-4), "name": "head"},
    ], weight_decay=0.04)
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    tr_ids = np.where(store.index.split == "train")[0]
    t_s, t_t = 0.1, cfg.get("teacher_temp", 0.04)
    log, t0 = [], time.time()
    reset_peak()
    if probe is not None:
        log.append({"step": 0, **probe(teacher.backbone)})
    for step in range(steps):
        cos = 0.5 * (1 + math.cos(math.pi * step / steps))
        m = 1 - (1 - cfg.get("momentum", 0.996)) * cos
        for g in opt.param_groups:
            warm = min(1.0, (step + 1) / max(1, int(0.1 * steps)))
            g["lr"] = g["base_lr"] * warm * (0.1 + 0.9 * cos)
            if g["name"] == "backbone" and step < head_only:
                g["lr"] = 0.0
        x = store.batch(np.random.choice(tr_ids, bs))
        g1, g2 = gpu_augment(x.clone(), "basic"), gpu_augment(x.clone(), "basic")
        locs = [gpu_augment(c.contiguous(), "basic") for c in local_crops(x, n_local, cfg.get("local_px", 98))]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t_out = teacher(normalize(torch.cat([g1, g2]))).float()
            s_glob = student(normalize(torch.cat([g1, g2]))).float()
            s_loc = student(normalize(torch.cat(locs))).float() if locs else None
        t_prob = F.softmax((t_out - center) / t_t, -1).chunk(2)
        s_views = list(s_glob.chunk(2)) + (list(s_loc.chunk(n_local)) if locs else [])
        loss, n = 0.0, 0
        for ti, tp in enumerate(t_prob):
            for si, sv in enumerate(s_views):
                if si == ti:
                    continue
                loss = loss + torch.sum(-tp * F.log_softmax(sv / t_s, -1), -1).mean()
                n += 1
        loss = loss / n
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
        opt.step()
        with torch.no_grad():
            for ps, pt in zip(student.parameters(), teacher.parameters()):
                pt.mul_(m).add_(ps.detach(), alpha=1 - m)
            center.mul_(0.9).add_(t_out.mean(0, keepdim=True), alpha=0.1)
        if step % 25 == 0 or step == steps - 1:
            ent = float(-(t_prob[0] * torch.log(t_prob[0] + 1e-9)).sum(-1).mean())
            log.append({"step": step + 1, "loss": float(loss), "teacher_entropy": ent, "momentum": m})
        if probe is not None and (step + 1) % cfg.get("probe_every", 500) == 0:
            log.append({"step": step + 1, **probe(teacher.backbone)})
    sd = {k: v for k, v in teacher.backbone.state_dict().items() if not k.startswith("head")}
    torch.save(sd, out_dir / "backbone.pt")
    info = {"gpu_hours": (time.time() - t0) / 3600, "peak_mem_gb": peak_gb(),
            "gpu": gpu_name(), "log": log, "config": cfg}
    (out_dir / "dino_log.json").write_text(json.dumps(info, indent=1))
    return info
