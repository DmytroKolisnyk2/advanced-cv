"""Competition metric (log-weighted EMD), complementary D-value errors and curve repair."""

from __future__ import annotations

import numpy as np

SIEVES = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0])
COLS = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]
LOG_D = np.log10(SIEVES)
DLOG = np.diff(LOG_D)  # widths of the ten intervals, sum = 5 decades


def emd(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-sample EMD: sum_i |F_i - F^_i| * (log10 x_{i+1} - log10 x_i), i = 1..10. Range [0, 500]."""
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    return (np.abs(true[..., :-1] - pred[..., :-1]) * DLOG).sum(-1)


def mean_emd(true: np.ndarray, pred: np.ndarray) -> float:
    return float(emd(true, pred).mean())


def shift_emd(decades: float) -> float:
    """EMD caused only by a scale error that shifts a curve along log d (curve fully inside the grid)."""
    return 100.0 * abs(decades)


def d_value(curve: np.ndarray, q: float) -> np.ndarray:
    """log10 diameter at which the curve passes q percent (linear in log10 d, clamped to the sieve range)."""
    curve = np.atleast_2d(np.asarray(curve, float))
    out = np.empty(len(curve))
    for n, c in enumerate(curve):
        c = np.maximum.accumulate(c)
        if q <= c[0]:
            out[n] = LOG_D[0]
            continue
        i = int(np.searchsorted(c, q))  # first index with c >= q
        i = min(i, len(c) - 1)
        c0, c1 = c[i - 1], c[i]
        t = 0.0 if c1 == c0 else (q - c0) / (c1 - c0)
        out[n] = LOG_D[i - 1] + t * (LOG_D[i] - LOG_D[i - 1])
    return out


def d_errors(true: np.ndarray, pred: np.ndarray, qs=(10, 30, 50, 60)) -> dict[str, float]:
    """Mean absolute error of D_q in decades (log10 mm); 0.30 = a factor-2 error in diameter."""
    return {f"D{q}": float(np.abs(d_value(true, q) - d_value(pred, q)).mean()) for q in qs}


def pav(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: least-squares non-decreasing fit (unit weights)."""
    blocks: list[list[float]] = []  # [mean, size]
    for v in np.asarray(y, float):
        blocks.append([v, 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, n2 = blocks.pop()
            m1, n1 = blocks.pop()
            blocks.append([(m1 * n1 + m2 * n2) / (n1 + n2), n1 + n2])
    return np.concatenate([np.full(int(n), m) for m, n in blocks])


def make_valid(pred: np.ndarray, method: str = "isotonic") -> np.ndarray:
    """Clip to [0, 100], enforce monotonicity and a final 100."""
    pred = np.atleast_2d(np.clip(np.asarray(pred, float), 0, 100)).copy()
    for n in range(len(pred)):
        pred[n] = pav(pred[n]) if method == "isotonic" else np.maximum.accumulate(pred[n])
    pred[:, -1] = 100.0
    return np.clip(pred, 0, 100)


def is_valid(pred: np.ndarray) -> bool:
    pred = np.atleast_2d(pred)
    return bool((pred >= 0).all() and (pred <= 100).all() and (np.diff(pred, axis=1) >= -1e-9).all()
                and np.allclose(pred[:, -1], 100))


UNIFORM_BASELINE = np.round(np.arange(1, 12) * 100 / 11, 2)
