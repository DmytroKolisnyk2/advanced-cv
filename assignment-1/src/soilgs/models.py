"""Backbones (timm CNN / ViT / SSL encoders, a from-scratch ViT), the curve head and LoRA."""

from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metric import DLOG

N_BINS = 11  # mass in (<0.002), (0.002-0.0063), ..., (63-200) mm
_DLOG = torch.tensor(DLOG, dtype=torch.float32)


def logits_to_curve(logits: torch.Tensor) -> torch.Tensor:
    """11 bin logits -> cumulative percent passing at the 11 sieves (valid by construction, last = 100)."""
    return 100 * torch.cumsum(logits.float().softmax(-1), -1)


def emd_loss(curve: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((curve[:, :-1] - target[:, :-1]).abs() * _DLOG.to(curve.device)).sum(-1).mean()


class TinyViT(nn.Module):
    """Plain pre-norm ViT trained from scratch (patch embedding, [CLS], learned positions, MHSA + MLP)."""

    def __init__(self, img=224, patch=16, dim=192, depth=6, heads=3, mlp=4, drop_path=0.1, num_classes=N_BINS):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        n = (img // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n + 1, dim) * 0.02)
        dpr = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList([timm.models.vision_transformer.Block(dim, heads, mlp, qkv_bias=True,
                                                                          drop_path=d) for d in dpr])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.num_features = dim
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.norm(x)

    def forward_head(self, x, pre_logits=False):
        x = x[:, 0]
        return x if pre_logits else self.head(x)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) B A x, with W frozen, A ~ Kaiming, B = 0 (the update starts at zero)."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def add_lora(model: nn.Module, r: int = 8, alpha: float = 16) -> int:
    """LoRA on the fused qkv and the output projection of every attention block; freeze everything else
    except the head. Returns the number of LoRA parameters."""
    for p in model.parameters():
        p.requires_grad = False
    n = 0
    for blk in model.blocks:
        blk.attn.qkv = LoRALinear(blk.attn.qkv, r, alpha)
        blk.attn.proj = LoRALinear(blk.attn.proj, r, alpha)
        n += sum(p.numel() for m in (blk.attn.qkv, blk.attn.proj) for p in (m.A, m.B))
    for p in model.get_classifier().parameters():
        p.requires_grad = True
    return n


def bn_to_gn(module: nn.Module, groups: int = 32) -> nn.Module:
    """Replace every BatchNorm2d with GroupNorm, keeping the pretrained affine parameters."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            g = math.gcd(groups, child.num_features)
            gn = nn.GroupNorm(g, child.num_features, affine=True)
            gn.weight.data.copy_(child.weight.data)
            gn.bias.data.copy_(child.bias.data)
            setattr(module, name, gn)
        else:
            bn_to_gn(child, groups)
    return module


def freeze_bn(model: nn.Module) -> None:
    """Keep BatchNorm in eval mode (pretrained running statistics) with frozen affine parameters."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def build_model(cfg: dict) -> nn.Module:
    arch = cfg["arch"]
    if arch == "tiny_vit":
        return TinyViT(**cfg.get("arch_kwargs", {}))
    kwargs = dict(pretrained=cfg.get("pretrained", True), num_classes=N_BINS,
                  drop_path_rate=cfg.get("drop_path", 0.0))
    if any(k in arch for k in ("vit_", "deit_", "eva")):
        kwargs["img_size"] = cfg.get("img") or 224
        if cfg.get("dynamic_img_size"):
            kwargs["dynamic_img_size"] = True
    model = timm.create_model(arch, **kwargs)
    if cfg.get("init_ckpt"):
        sd = torch.load(cfg["init_ckpt"], map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert all(k.startswith("head") for k in missing), missing
    if cfg.get("norm") == "groupnorm":
        bn_to_gn(model)
    return model


def head_bias_init(model: nn.Module, prior_curve) -> None:
    """Start the head at the training-set mean distribution (bin log-masses), so early steps are sane."""
    import numpy as np
    bins = np.diff(np.concatenate([[0.0], np.asarray(prior_curve, float)])) / 100
    head = model.get_classifier() if hasattr(model, "get_classifier") else model.head
    with torch.no_grad():
        head.bias.copy_(torch.tensor(np.log(np.clip(bins, 1e-4, None)), dtype=head.bias.dtype))
        head.weight.mul_(0.1)


def features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Pre-logit embedding (what the head sees)."""
    return model.forward_head(model.forward_features(x), pre_logits=True)


def is_vit(model: nn.Module) -> bool:
    return hasattr(model, "blocks") and hasattr(model, "patch_embed")


def gpu_augment(x: torch.Tensor, strength: str = "basic") -> torch.Tensor:
    """Scale-safe augmentation on a uint8->float batch [B,3,H,W] in [0,1]: dihedral group + mild photometric.
    No random-resized crops and no zoom: they change mm-per-pixel and so the label."""
    if strength == "none":
        return x
    b = len(x)
    k = int(torch.randint(0, 4, (1,)))
    x = torch.rot90(x, k, (2, 3))
    flip = torch.rand(b, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    if strength in ("basic", "strong"):
        amp = 0.15 if strength == "basic" else 0.3
        bri = 1 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * amp
        con = 1 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * amp
        sat = 1 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * amp
        mean = x.mean((1, 2, 3), keepdim=True)
        gray = x.mean(1, keepdim=True)
        x = ((x - mean) * con + mean) * bri
        x = gray + (x - gray) * sat
    if strength == "strong":
        x = x + torch.randn_like(x) * 0.02
        blur = torch.rand(b, device=x.device) < 0.2
        xb = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, 1)
        x = torch.where(blur[:, None, None, None], xb, x)
    return x.clamp(0, 1)
