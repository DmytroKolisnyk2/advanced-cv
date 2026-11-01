"""Frozen self-supervised features: weighted k-NN, ridge and a linear probe, on the same folds."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV

from .metric import COLS, make_valid
from .models import build_model, emd_loss, logits_to_curve
from .train import TileStore, gpu_name, grey, normalize, peak_gb, reset_peak, score, shuffle_patches

FEATS = ("cls", "patch", "cls+patch", "block4", "block8")


@torch.no_grad()
def extract(model, store: TileStore, transform=None, bs: int = 256) -> dict[str, np.ndarray]:
    """Per-tile CLS, mean patch token (last layer) and mean patch tokens of intermediate blocks 4 and 8."""
    model.eval()
    npt = getattr(model, "num_prefix_tokens", 1)
    out = {k: [] for k in FEATS}
    for i in range(0, len(store.index), bs):
        x = store.batch(np.arange(i, min(i + bs, len(store.index))))
        if transform is not None:
            x = transform(x)
        x = normalize(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tok = model.forward_features(x).float()
            inter = model.forward_intermediates(x, indices=[3, 7], intermediates_only=True)
        cls, patch = tok[:, 0], tok[:, npt:].mean(1)
        out["cls"].append(cls.cpu())
        out["patch"].append(patch.cpu())
        out["cls+patch"].append(torch.cat([cls, patch], 1).cpu())
        out["block4"].append(inter[0].float().mean((2, 3)).cpu())
        out["block8"].append(inter[1].float().mean((2, 3)).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def sample_means(index: pd.DataFrame, f: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(f)
    df["photo"], df["sample_id"] = index.photo.to_numpy(), index.sample_id.to_numpy()
    photo = df.groupby(["sample_id", "photo"]).mean()
    return photo.groupby("sample_id").mean()


def ridge_fit_predict(xtr, ytr, xte_list):
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    r = RidgeCV(alphas=np.logspace(-1, 5, 25)).fit((xtr - mu) / sd, ytr[:, :-1])
    out = []
    for x in xte_list:
        p = np.concatenate([r.predict((x - mu) / sd), np.full((len(x), 1), 100.0)], 1)
        out.append(make_valid(p))
    return out, float(r.alpha_)


def knn_fit_predict(xtr, ytr, xte_list, k: int = 5, tau: float = 0.1):
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    a = (xtr - mu) / sd
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    out = []
    for x in xte_list:
        b = (x - mu) / sd
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        s = b @ a.T
        top = np.argsort(-s, 1)[:, :k]
        st = np.take_along_axis(s, top, 1)
        w = np.exp((st - st.max(1, keepdims=True)) / tau)
        w /= w.sum(1, keepdims=True)
        out.append((w[:, :, None] * ytr[top]).sum(1))
    return out, None


def lp_fit_predict(ftr, ytr_tiles, fte_list, prior, steps: int = 500, lr: float = 1e-3, wd: float = 1e-3):
    """Linear probe on tile features, trained with the EMD loss through the softmax curve head."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    x = torch.tensor((ftr - mu) / sd, dtype=torch.float32, device=dev)
    y = torch.tensor(ytr_tiles, dtype=torch.float32, device=dev)
    lin = torch.nn.Linear(x.shape[1], 11).to(dev)
    bins = np.diff(np.concatenate([[0.0], prior])) / 100
    with torch.no_grad():
        lin.weight.zero_()
        lin.bias.copy_(torch.tensor(np.log(np.clip(bins, 1e-4, None))))
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(steps):
        ids = torch.randint(0, len(x), (min(512, len(x)),), generator=g).to(dev)
        loss = emd_loss(logits_to_curve(lin(x[ids])), y[ids])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return [logits_to_curve(lin(torch.tensor((f - mu) / sd, dtype=torch.float32, device=dev))).cpu().numpy()
                for f in fte_list], None


def run_frozen(enc_name: str, cfg: dict, stores: dict[str, TileStore], labels: pd.DataFrame,
               folds: pd.DataFrame, out_root: Path, feats_root: Path) -> list[dict]:
    """One encoder: extract once, then ridge on every feature set + k-NN + linear probe on cls+patch."""
    heads = [("ridge", f) for f in FEATS] + [("knn", "cls+patch"), ("lp", "cls+patch")]
    names = {(h, f): f"{enc_name}_{h}_{f.replace('+', '')}" for h, f in heads}
    if all((out_root / n / "metrics.json").exists() for n in names.values()):
        return [json.loads((out_root / n / "metrics.json").read_text()) for n in names.values()]
    t0 = time.time()
    reset_peak()
    model = build_model({**cfg, "pretrained": True}).to(stores["main"].x.device)
    feats = {"main": extract(model, stores["main"])}
    for v in stores:
        if v != "main":
            feats[v] = extract(model, stores[v])
    feats["shuffle"] = extract(model, stores["main"], shuffle_patches)
    feats["grey"] = extract(model, stores["main"], grey)
    extract_hours = (time.time() - t0) / 3600
    feats_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feats_root / f"{enc_name}.npz", **{f"main_{k}": v for k, v in feats["main"].items()})
    del model
    torch.cuda.empty_cache()

    idx = stores["main"].index
    targets = dict(zip(labels.sample_id, labels[COLS].to_numpy(float)))
    fold_of = dict(zip(folds.sample_id, folds.fold))
    results = []
    for (head, fset), name in names.items():
        t1 = time.time()
        out = out_root / name
        out.mkdir(parents=True, exist_ok=True)
        groups = {g: (stores["main" if g in ("shuffle", "grey") else g].index, feats[g][fset]) for g in feats}
        smean = {g: sample_means(ix, f) for g, (ix, f) in groups.items()}
        oof, ctrl, var, test_parts = [], {"shuffle": [], "grey": []}, {v: [] for v in stores if v != "main"}, []
        alphas = []
        for fold in range(int(folds.fold.max()) + 1):
            val = sorted(s for s, f in fold_of.items() if f == fold)
            tr = sorted(s for s, f in fold_of.items() if f != fold)
            ytr = np.stack([targets[s] for s in tr])
            evals = {"val": smean["main"].loc[val].to_numpy()}
            for g in ctrl:
                evals[g] = smean[g].loc[val].to_numpy()
            for v in var:
                evals[v] = smean[v].loc[[s for s in val if s in smean[v].index]].to_numpy()
            test_sids = sorted(set(idx.sample_id[idx.split == "test"]))
            if test_sids:
                evals["test"] = smean["main"].loc[test_sids].to_numpy()
            keys = list(evals)
            if head == "lp":
                tr_mask = idx.sample_id.isin(tr).to_numpy() & (idx.split == "train").to_numpy()
                ftile = groups["main"][1]
                preds, alpha = lp_fit_predict(ftile[tr_mask], np.stack([targets[s] for s in idx.sample_id[tr_mask]]),
                                              [evals[k] for k in keys], ytr.mean(0))
            else:
                fn = ridge_fit_predict if head == "ridge" else knn_fit_predict
                preds, alpha = fn(smean["main"].loc[tr].to_numpy(), ytr, [evals[k] for k in keys])
            alphas.append(alpha)
            P = dict(zip(keys, preds))
            mk = lambda sids, p: pd.DataFrame(p, columns=COLS).assign(sample_id=list(sids), fold=fold)
            oof.append(mk(val, P["val"]))
            for g in ctrl:
                ctrl[g].append(mk(val, P[g]))
            for v in var:
                var[v].append(mk([s for s in val if s in smean[v].index], P[v]))
            if test_sids:
                test_parts.append(mk(test_sids, P["test"]))
        oof = pd.concat(oof).sort_values("sample_id").reset_index(drop=True)
        oof[["sample_id", "fold"] + COLS].to_csv(out / "oof.csv", index=False)
        m = score(oof, labels)
        m.update(name=name, family="ssl", encoder=enc_name, head=head, features=fset,
                 gpu=gpu_name(),
                 gpu_hours=round(extract_hours / len(names) + (time.time() - t1) / 3600, 4),
                 peak_mem_gb=peak_gb(),
                 ridge_alpha=alphas if head == "ridge" else None,
                 controls={g: score(pd.concat(v), labels)["cv_mean"] for g, v in ctrl.items()},
                 robustness={v: score(pd.concat(p), labels)["cv_mean"] for v, p in var.items()})
        if test_parts:
            t = pd.concat(test_parts).groupby("sample_id", as_index=False)[COLS].mean()
            t[COLS[-1]] = 100.0
            t.to_csv(out / "test.csv", index=False)
        (out / "config.json").write_text(json.dumps({**cfg, "head": head, "features": fset}, indent=1))
        (out / "metrics.json").write_text(json.dumps(m, indent=1))
        results.append(m)
    return results
