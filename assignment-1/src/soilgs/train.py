"""Cross-validated training of one experiment: tile model -> photo -> sample curves, OOF + test predictions,
controls (shuffled patches, grey input), scale robustness and a compute log, all written to results/exps/<name>/."""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm.optim
import torch
import torch.nn.functional as F

from .metric import COLS, d_errors, emd
from .models import (add_lora, build_model, emd_loss, features, freeze_bn, gpu_augment, head_bias_init,
                     logits_to_curve)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def gpu_name() -> str:
    return torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"


def reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_gb() -> float:
    return round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class TileStore:
    """All tiles of one cache resident on the GPU as uint8 (a few GB at most)."""

    def __init__(self, cache_dir: Path, device: str, photos: set[str] | None = None):
        idx = pd.read_csv(cache_dir / "tiles.csv")
        arr = np.load(cache_dir / "tiles.npy", mmap_mode="r")
        keep = np.ones(len(idx), bool) if photos is None else idx.photo.isin(photos).to_numpy()
        self.index = idx[keep].reset_index(drop=True)
        self.x = torch.from_numpy(np.ascontiguousarray(arr[keep])).to(device)  # [N,H,W,3] uint8

    def batch(self, ids, img: int | None = None) -> torch.Tensor:
        x = self.x[ids].permute(0, 3, 1, 2).float().div_(255)
        if img and img != x.shape[-1]:
            x = F.interpolate(x, size=(img, img), mode="bilinear", antialias=True, align_corners=False)
        return x


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def shuffle_patches(x: torch.Tensor, grid: int = 4) -> torch.Tensor:
    """Control: permute a grid x grid of blocks inside each tile (keeps texture, destroys layout)."""
    b, c, h, w = x.shape
    p = h // grid
    blocks = x[:, :, :p * grid, :p * grid].unfold(2, p, p).unfold(3, p, p)  # b c g g p p
    blocks = blocks.reshape(b, c, grid * grid, p, p)[:, :, torch.randperm(grid * grid, device=x.device)]
    return blocks.reshape(b, c, grid, grid, p, p).permute(0, 1, 2, 4, 3, 5).reshape(b, c, p * grid, p * grid)


def grey(x: torch.Tensor) -> torch.Tensor:
    return x.mean(1, keepdim=True).expand_as(x).contiguous()


@torch.no_grad()
def predict_tiles(model, store: TileStore, ids: np.ndarray, img: int | None, transform=None,
                  bs: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(ids), bs):
        x = store.batch(ids[i:i + bs], img)
        if transform is not None:
            x = transform(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(logits_to_curve(model(normalize(x))).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, len(COLS)))


@torch.no_grad()
def tile_features(model, store: TileStore, ids: np.ndarray, img: int | None, bs: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(ids), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(features(model, normalize(store.batch(ids[i:i + bs], img))).float().cpu())
    return torch.cat(out).numpy()


def aggregate(index: pd.DataFrame, curves: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tile curves -> photo curve (mean) -> sample curve (mean over photos). Means of valid curves are valid."""
    df = pd.concat([index[["photo", "sample_id"]].reset_index(drop=True), pd.DataFrame(curves, columns=COLS)], axis=1)
    photo = df.groupby(["sample_id", "photo"], as_index=False)[COLS].mean()
    sample = photo.groupby("sample_id", as_index=False)[COLS].mean()
    return photo, sample


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(0)
    y = y - y.mean(0)
    hsic = np.linalg.norm(y.T @ x) ** 2
    return float(hsic / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))


def make_optimizer(model, cfg: dict, params_only_trainable: bool = True):
    return timm.optim.create_optimizer_v2(
        model, opt=cfg["opt"], lr=cfg["lr"], weight_decay=cfg["wd"], momentum=0.9,
        layer_decay=cfg.get("layer_decay"), filter_bias_and_bn=cfg.get("no_wd_norm_bias", True))


def lr_factor(step: int, total: int, warmup: float, sched: str) -> float:
    w = int(total * warmup)
    if step < w:
        return (step + 1) / w
    if sched == "constant":
        return 1.0
    t = (step - w) / max(1, total - w)
    return 0.5 * (1 + math.cos(math.pi * t))


class BatchSampler:
    """by_sample: pick a physical sample uniformly, then one of its tiles (every soil weighs the same).
    grouped: every batch holds tiles of only `groups` samples (stresses BatchNorm statistics)."""

    def __init__(self, index: pd.DataFrame, ids: np.ndarray, mode: str, groups: int, seed: int):
        self.rng = np.random.default_rng(seed)
        sids = index.sample_id.to_numpy()[ids]
        self.by_sample = {s: ids[sids == s] for s in np.unique(sids)}
        self.samples = list(self.by_sample)
        self.mode, self.groups, self.ids = mode, groups, ids

    def __call__(self, bs: int) -> np.ndarray:
        if self.mode == "uniform":
            return self.rng.choice(self.ids, bs)
        pool = self.samples if self.mode == "by_sample" else list(
            self.rng.choice(self.samples, self.groups, replace=False))
        chosen = self.rng.choice(len(pool), bs)
        return np.array([self.rng.choice(self.by_sample[pool[c]]) for c in chosen])


def train_fold(cfg: dict, store: TileStore, tr_ids: np.ndarray, targets: dict[str, np.ndarray],
               prior: np.ndarray, seed: int, log: list) -> torch.nn.Module:
    torch.manual_seed(seed)
    dev = store.x.device
    model = build_model(cfg).to(dev)
    head_bias_init(model, prior)
    mode = cfg.get("mode", "ft")
    if mode == "lora":
        cfg["_lora_params"] = add_lora(model, cfg.get("lora_r", 8), cfg.get("lora_alpha", 16))
        model.to(dev)
    stages = [("lp", cfg.get("lp_steps", 300)), ("ft", cfg["steps"])] if mode == "lpft" else [(mode, cfg["steps"])]
    sampler = BatchSampler(store.index, tr_ids, cfg.get("sampler", "by_sample"), cfg.get("groups", 2), seed)
    tgt = torch.tensor(np.stack([targets.get(s, np.full(len(COLS), np.nan)) for s in store.index.sample_id]),
                       dtype=torch.float32, device=dev)  # test tiles get NaN and are never sampled
    for stage, steps in stages:
        scfg = dict(cfg)
        if stage == "lp":
            for n, p in model.named_parameters():
                p.requires_grad = n.startswith(("head", "fc.", "classifier"))
            scfg.update(lr=cfg.get("lp_lr", 1e-3), layer_decay=None, opt="adamw")
        elif mode == "lpft" and stage == "ft":
            for p in model.parameters():
                p.requires_grad = True
            scfg["lr"] = cfg.get("ft_lr", cfg["lr"])
        if cfg.get("norm") == "frozen_bn":
            freeze_bn(model)
        opt = make_optimizer(model, scfg)
        for g in opt.param_groups:
            g["base_lr"] = g["lr"]
        for step in range(steps):
            model.train()
            if stage == "lp":  # frozen backbone in eval mode: features are those of the pretrained encoder
                model.eval()
            if cfg.get("norm") == "frozen_bn":
                freeze_bn(model)
            f = lr_factor(step, steps, scfg.get("warmup", 0.1), scfg.get("sched", "cosine"))
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f * g.get("lr_scale", 1.0)
            ids = sampler(cfg["batch"])
            x = normalize(gpu_augment(store.batch(ids, cfg.get("img")), cfg.get("aug", "basic")))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = emd_loss(logits_to_curve(model(x)), tgt[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.get("clip", 1.0))
            opt.step()
            if step % 20 == 0 or step == steps - 1:
                log.append({"stage": stage, "step": step, "loss": float(loss), "lr": opt.param_groups[0]["lr"]})
            if not math.isfinite(float(loss)):
                log.append({"stage": stage, "step": step, "loss": float("nan"), "diverged": True})
                return model
    return model


def run_experiment(name: str, cfg: dict, stores: dict[str, TileStore], labels: pd.DataFrame,
                   folds: pd.DataFrame, out_root: Path, ckpt_root: Path) -> dict:
    """stores: 'main' (train+test tiles of the experiment's cache) and robustness variants (train photos)."""
    out = out_root / name
    if (out / "metrics.json").exists():
        return json.loads((out / "metrics.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    store = stores["main"]
    idx = store.index
    targets = dict(zip(labels.sample_id, labels[COLS].to_numpy(float)))
    fold_of = dict(zip(folds.sample_id, folds.fold))
    k = int(folds.fold.max()) + 1
    reset_peak()
    t0 = time.time()
    base_feats = None
    if cfg.get("family") == "ssl":  # the encoder before tuning, for CKA against the tuned one
        base = build_model(copy.deepcopy(cfg)).to(store.x.device)
        base_feats = tile_features(base, store, np.arange(len(idx)), cfg.get("img"))
        del base
    oof, oof_photo, test_parts, logs, extra = [], [], [], {}, {"cka": []}
    variants = {v: [] for v in stores if v != "main"}
    controls = {"shuffle": [], "grey": []}
    test_ids = np.where(idx.split == "test")[0]
    for fold in range(k):
        val_samples = {s for s, f in fold_of.items() if f == fold}
        is_tr = (idx.split == "train") & ~idx.sample_id.isin(val_samples)
        tr_ids, va_ids = np.where(is_tr)[0], np.where(idx.sample_id.isin(val_samples))[0]
        prior = labels[~labels.sample_id.isin(val_samples)][COLS].to_numpy(float).mean(0)
        log: list = []
        model = train_fold(copy.deepcopy(cfg), store, tr_ids, targets, prior, cfg.get("seed", 0) + fold, log)
        logs[fold] = log
        img = cfg.get("img")
        photo, sample = aggregate(idx.iloc[va_ids], predict_tiles(model, store, va_ids, img))
        sample["fold"], photo["fold"] = fold, fold
        oof.append(sample)
        oof_photo.append(photo)
        for cname, fn in (("shuffle", shuffle_patches), ("grey", grey)):
            controls[cname].append(aggregate(idx.iloc[va_ids], predict_tiles(model, store, va_ids, img, fn))[1]
                                   .assign(fold=fold))
        for v in variants:
            vs = stores[v]
            v_ids = np.where(vs.index.sample_id.isin(val_samples))[0]
            variants[v].append(aggregate(vs.index.iloc[v_ids], predict_tiles(model, vs, v_ids, img))[1]
                               .assign(fold=fold))
        if len(test_ids):
            test_parts.append(aggregate(idx.iloc[test_ids], predict_tiles(model, store, test_ids, img))[1])
        if base_feats is not None:
            tuned = tile_features(model, store, va_ids[:2000], img)
            extra["cka"].append(linear_cka(base_feats[va_ids[:2000]], tuned))
        n_params = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if cfg.get("save_ckpt", True):  # every fold, fp16 (fold 0 feeds the interpretation maps)
            ckpt_root.mkdir(parents=True, exist_ok=True)
            sd = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
            torch.save(sd, ckpt_root / f"{name}_fold{fold}.pt")
        del model
        torch.cuda.empty_cache()
    hours = (time.time() - t0) / 3600
    oof = pd.concat(oof).sort_values("sample_id").reset_index(drop=True)
    pd.concat(oof_photo).to_csv(out / "oof_photo.csv", index=False)
    oof.to_csv(out / "oof.csv", index=False)
    metrics = score(oof, labels)
    metrics.update(name=name, family=cfg.get("family"), gpu=gpu_name(),
                   gpu_hours=round(hours, 4), peak_mem_gb=peak_gb(),
                   params_m=round(n_params / 1e6, 2), trainable_m=round(n_train / 1e6, 3))
    if extra["cka"]:
        metrics["cka_base_vs_tuned"] = float(np.mean(extra["cka"]))
    metrics["controls"] = {c: score(pd.concat(v), labels)["cv_mean"] for c, v in controls.items()}
    metrics["robustness"] = {v: score(pd.concat(p), labels)["cv_mean"] for v, p in variants.items()}
    if test_parts:
        test = pd.concat(test_parts).groupby("sample_id", as_index=False)[COLS].mean()
        test[COLS[-1]] = 100.0
        test.to_csv(out / "test.csv", index=False)
    (out / "train_log.json").write_text(json.dumps(logs))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


def score(pred: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """CV mean +- std over folds of the per-sample EMD, plus D-value errors on the pooled OOF."""
    lab = labels.set_index("sample_id").loc[pred.sample_id, COLS].to_numpy(float)
    p = pred[COLS].to_numpy(float)
    e = emd(lab, p)
    res = {"cv_pooled": float(e.mean())}
    if "fold" in pred:
        per_fold = pd.Series(e).groupby(pred.fold.to_numpy()).mean()
        res.update(cv_mean=float(per_fold.mean()), cv_std=float(per_fold.std(ddof=0)),
                   per_fold=per_fold.round(3).tolist())
    else:
        res.update(cv_mean=float(e.mean()), cv_std=float("nan"))
    res.update(d_errors(lab, p))
    res["per_sample"] = dict(zip(pred.sample_id, np.round(e, 3)))
    return res
