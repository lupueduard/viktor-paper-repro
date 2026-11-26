"""Paired bootstrap, the exact visibility split of H1 and the preregistered verdict rules.

Everything here is pure numpy and is unit-tested in tests/test_stats.py.
"""
from __future__ import annotations

import math

import numpy as np

M = 0.05       # minimal difference (H3, H4)
DELTA = 0.05   # non-inferiority margin (H2)
B = 20_000     # bootstrap resamples


def split(uP, cP, uR, cR) -> dict:
    """Exact split of Δ = acc(R) − acc(P) into rows only R parses (V), both parse (B), only P
    parses (L). Each part is divided by n, so ΔV + ΔB + ΔL = Δ exactly."""
    uP, cP, uR, cR = (np.asarray(x, dtype=float) for x in (uP, cP, uR, cR))
    n = len(uP)
    V = (1 - uP) * uR
    Bm = uP * uR
    L = uP * (1 - uR)
    dV = (V * cR).sum() / n
    dB = (Bm * (cR - cP)).sum() / n
    dL = -(L * cP).sum() / n
    d = (cR - cP).sum() / n
    return {"delta": d, "dV": dV, "dB": dB, "dL": dL, "T": dV - 0.8 * (dV + dB),
            "nV": int(V.sum()), "nB": int(Bm.sum()), "nL": int(L.sum()), "n": n}


def resample_weights(n: int, seed: int, b: int = B) -> np.ndarray:
    """(b, n) matrix of draw counts: a unit drawn k times counts k times."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(b, n))
    w = np.zeros((b, n), dtype=np.float32)
    np.add.at(w, (np.arange(b)[:, None], idx), 1)
    return w


def percentile_ci(samples: np.ndarray, q: float) -> tuple[float, float]:
    """Paper §3.5: sort ascending; lower = index floor(Bq)−1, upper = index ceil(B(1−q))−1, clamped."""
    s = np.sort(np.asarray(samples))
    b = len(s)
    lo = min(max(math.floor(b * q) - 1, 0), b - 1)
    hi = min(max(math.ceil(b * (1 - q)) - 1, 0), b - 1)
    return float(s[lo]), float(s[hi])


def q_for(family_size: int) -> float:
    return 0.025 / family_size


def boot_mean_diff(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Resampled mean(x) − mean(y) for paired row vectors."""
    d = np.asarray(x, dtype=np.float32) - np.asarray(y, dtype=np.float32)
    return (w @ d) / w.sum(1)


def boot_mean(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (w @ np.asarray(x, dtype=np.float32)) / w.sum(1)


def boot_split(w: np.ndarray, uP, cP, uR, cR) -> dict[str, np.ndarray]:
    uP, cP, uR, cR = (np.asarray(x, dtype=np.float32) for x in (uP, cP, uR, cR))
    n = w.sum(1)
    dV = (w @ ((1 - uP) * uR * cR)) / n
    dB = (w @ (uP * uR * (cR - cP))) / n
    return {"delta": (w @ (cR - cP)) / n, "T": dV - 0.8 * (dV + dB), "dV": dV, "dB": dB}


# ---- verdict rules (paper Table 2 / §3.4) -------------------------------------------------

def h1_verdict(delta_ci, T_ci) -> str:
    if not delta_ci[0] > 0:
        return "no gain: not tested"
    if T_ci[0] > 0:
        return "visibility dominates"
    if T_ci[1] < 0:
        return "quality matters"
    return "inconclusive"


def h2_verdict(theta_ci, delta: float = DELTA) -> str:
    if theta_ci[0] > -delta:
        return "holds"
    if theta_ci[1] < -delta:
        return "fails"
    return "inconclusive"


def h3_verdict(dT_ci, dP_ci, trunc_base_ci, m: float = M) -> str:
    if trunc_base_ci[1] < m:
        return "no room"
    if dT_ci[0] > m and dP_ci[0] > m:
        return "holds"
    if dT_ci[1] < m or dP_ci[1] < m:
        return "fails"
    return "inconclusive"


def _outside(ci, m):
    return ci[0] > m or ci[1] < -m


def _inside(ci, m):
    return ci[0] > -m and ci[1] < m


def h4_claim(dS_ci, dR_ci, g_ci, m: float = M) -> str:
    """Holds if D_S and D_R lie wholly outside ±m with the same sign and Γ wholly inside ±m;
    fails if any of the three lies wholly on the other side of its boundary."""
    same_sign = (dS_ci[0] > m and dR_ci[0] > m) or (dS_ci[1] < -m and dR_ci[1] < -m)
    if same_sign and _inside(g_ci, m):
        return "holds"
    if _inside(dS_ci, m) or _inside(dR_ci, m) or _outside(g_ci, m):
        return "fails"
    return "inconclusive"


def replication_label(test_verdict: str, rep_verdict: str) -> str:
    if test_verdict == rep_verdict:
        return "replicated"
    opposite = {("holds", "fails"), ("fails", "holds"),
                ("visibility dominates", "quality matters"), ("quality matters", "visibility dominates")}
    if (test_verdict, rep_verdict) in opposite:
        return "not replicated"
    return "inconclusive replication"
