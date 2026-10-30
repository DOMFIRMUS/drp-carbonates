"""Histogram-based segmentation: thresholds, end-members and grey-level porosity.

All functions work on an exact 16-bit histogram ``h`` (``h[i]`` = number of
voxels with grey level ``i``), so the statistics of a billion-voxel stack are
computed without holding it in memory. Convention: a voxel is pore when
``I < T``; thresholds are therefore returned as integers (the first grey level
classified as solid).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

LEVELS = np.arange(65536, dtype=np.float64)


# ---------------------------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------------------------
def moments(h: np.ndarray) -> tuple[int, float, float]:
    n = int(h.sum())
    mean = float((h * LEVELS).sum() / n)
    sd = float(np.sqrt((h * (LEVELS - mean) ** 2).sum() / n))
    return n, mean, sd


def cdf(h: np.ndarray) -> np.ndarray:
    return np.cumsum(h) / h.sum()


def porosity_at(h: np.ndarray, T: float) -> float:
    """Fraction of voxels with grey level < T."""
    t = int(np.ceil(T))
    if t <= 0:
        return 0.0
    return float(h[: min(t, 65536)].sum() / h.sum())


def quantile(h: np.ndarray, q: float) -> float:
    """Grey level below which a fraction q of the voxels lie (interpolated)."""
    c = cdf(h)
    i = int(np.searchsorted(c, q))
    if i == 0:
        return 0.0
    c0 = c[i - 1]
    c1 = c[i]
    frac = (q - c0) / (c1 - c0) if c1 > c0 else 0.0
    return float(i - 1 + frac + 0.5)


def occupied_range(h: np.ndarray, q_lo=1e-4, q_hi=1 - 1e-4) -> tuple[int, int]:
    return int(quantile(h, q_lo)), int(np.ceil(quantile(h, q_hi)))


def rebin(h: np.ndarray, nbins: int = 512):
    """Rebin the occupied grey-level range into `nbins` equal bins."""
    lo, hi = occupied_range(h)
    width = max(1, int(np.ceil((hi - lo) / nbins)))
    start = lo - lo % width
    edges = np.arange(start, min(65536, hi + width) + width, width)
    idx = np.clip((LEVELS[start:edges[-1]] - start) // width, 0, len(edges) - 2).astype(int)
    hb = np.bincount(idx, weights=h[start:edges[-1]], minlength=len(edges) - 1)
    centers = edges[:-1] + width / 2
    return hb, centers, width


def solid_mode(h: np.ndarray, nbins: int = 512, sigma_bins: float = 2.0) -> float:
    """Grey level of the solid peak (histogram mode, smoothed and refined)."""
    from .volume import histogram_mode

    hb, centers, _ = rebin(h, nbins)
    return histogram_mode(hb, centers, sigma_bins)


# ---------------------------------------------------------------------------
# Thresholds (pore = I < T)
# ---------------------------------------------------------------------------
def t_mean_k_sigma(h: np.ndarray, k: float) -> float:
    _, m, s = moments(h)
    return m - k * s


def t_quantile(h: np.ndarray, porosity: float) -> float:
    """Threshold that reproduces a target porosity (e.g. the laboratory value)."""
    c = cdf(h)
    return float(np.searchsorted(c, porosity) + 1)


def t_otsu(h: np.ndarray) -> float:
    """Otsu (1979): maximises the between-class variance."""
    w = np.cumsum(h, dtype=np.float64)
    mu = np.cumsum(h * LEVELS)
    wt, mt = w[-1], mu[-1]
    w1 = wt - w
    with np.errstate(divide="ignore", invalid="ignore"):
        m0 = mu / w
        m1 = (mt - mu) / w1
        sb = w * w1 * (m0 - m1) ** 2
    sb[~np.isfinite(sb)] = -1
    return float(np.argmax(sb) + 1)


def t_li(h: np.ndarray, tol: float = 0.5, max_iter: int = 1000) -> float:
    """Li & Lee (1993) minimum cross-entropy threshold (Li & Tam iteration)."""
    p = h.astype(np.float64)
    _, t, _ = moments(h)
    tn = t
    for _ in range(max_iter):
        T = int(np.ceil(t))
        wb, wf = p[:T].sum(), p[T:].sum()
        if wb == 0 or wf == 0:
            break
        mb = (p[:T] * LEVELS[:T]).sum() / wb
        mf = (p[T:] * LEVELS[T:]).sum() / wf
        mb = max(mb, 1e-9)
        tn = (mb - mf) / (np.log(mb) - np.log(mf))
        if abs(tn - t) < tol:
            break
        t = tn
    return float(np.ceil(tn))


def t_kittler_illingworth(h: np.ndarray, nbins: int = 1024) -> float:
    """Kittler & Illingworth (1986) minimum-error threshold (two Gaussians).

    Evaluated on a rebinned histogram to avoid empty-class singularities.
    """
    hb, centers, _ = rebin(h, nbins)
    p = hb / hb.sum()
    P1 = np.cumsum(p)
    M1 = np.cumsum(p * centers)
    S1 = np.cumsum(p * centers**2)
    P2 = 1 - P1
    M2 = M1[-1] - M1
    S2 = S1[-1] - S1
    with np.errstate(divide="ignore", invalid="ignore"):
        m1, m2 = M1 / P1, M2 / P2
        v1, v2 = S1 / P1 - m1**2, S2 / P2 - m2**2
        J = 1 + P1 * np.log(v1) + P2 * np.log(v2) - 2 * (P1 * np.log(P1) + P2 * np.log(P2))
    ok = np.isfinite(J) & (P1 > 1e-4) & (P2 > 1e-4) & (v1 > 0) & (v2 > 0)
    J[~ok] = np.inf
    i = int(np.argmin(J))
    return float(centers[i] + (centers[1] - centers[0]) / 2)


def t_triangle(h: np.ndarray, nbins: int = 512) -> float:
    """Zack et al. (1977) triangle threshold on the low-intensity tail."""
    hb, centers, _ = rebin(h, nbins)
    hs = gaussian_filter1d(hb, 1.0)
    pk = int(np.argmax(hs))
    nz = np.nonzero(hs[:pk] > 0)[0]
    lo = int(nz[0]) if len(nz) else 0
    x = np.arange(lo, pk + 1, dtype=np.float64)
    y = hs[lo : pk + 1]
    x1, y1, x2, y2 = lo, hs[lo], pk, hs[pk]
    d = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    i = int(x[np.argmax(d)])
    return float(centers[i] + (centers[1] - centers[0]) / 2)


# ---------------------------------------------------------------------------
# Partial-volume scale, end-members and grey-level (threshold-free) porosity
# ---------------------------------------------------------------------------
def pore_endmember(h: np.ndarray, q: float) -> float:
    """Pore (air) end-member: grey-level quantile of the darkest voxels."""
    return quantile(h, q)


def t_partial_volume(I_s: float, I_p: float, f: float) -> float:
    """Threshold of the partial-volume criterion f (pore if f(I) > f)."""
    return I_s - f * (I_s - I_p)


def f_of_threshold(I_s: float, I_p: float, T: float) -> float:
    return (I_s - T) / (I_s - I_p)


def mixing_porosity(h: np.ndarray, I_s: float, I_p: float, clip: bool = False) -> float:
    """Porosity from a linear two-end-member mixing model of the grey levels.

    Unclipped, it equals (I_s - mean) / (I_s - I_p): a mass balance of X-ray
    attenuation that is unbiased under zero-mean noise and needs no threshold.
    Clipped to [0, 1] per voxel it follows the depositors' voxel-porosity maps
    (with a background porosity of zero).
    """
    n = h.sum()
    f = (I_s - LEVELS) / (I_s - I_p)
    if clip:
        f = np.clip(f, 0.0, 1.0)
    return float((h * f).sum() / n)


def with_background(phi_mix: float, phi_bg: float) -> float:
    """Total porosity when the 'solid' end-member itself holds phi_bg porosity."""
    return phi_bg + (1.0 - phi_bg) * phi_mix


def implied_background(phi_lab: float, phi_mix: float) -> float:
    """Solid-phase porosity needed to reconcile image and laboratory porosity."""
    return (phi_lab - phi_mix) / (1.0 - phi_mix)


def histogram_excess(h: np.ndarray, I_s: float, nbins: int = 512) -> float:
    """Fraction of voxels in excess of a solid peak mirrored about its mode.

    Everything darker than the mode that is not explained by a symmetric solid
    peak (pore, partial-volume and microporous voxels). Upper bound on the
    fraction of voxels that are not pure solid.
    """
    hb, centers, _ = rebin(h, nbins)
    hs = gaussian_filter1d(hb, 1.0)
    m = int(np.argmin(np.abs(centers - I_s)))
    mirrored = np.zeros_like(hs)
    for i in range(m):
        j = 2 * m - i
        mirrored[i] = hs[j] if j < len(hs) else 0.0
    excess = np.clip(hs[:m] - mirrored[:m], 0, None).sum()
    return float(excess / hs.sum())


def all_thresholds(h: np.ndarray, I_s: float, I_p: float, lab_porosity: float,
                   legacy_k: float, primary_f: float) -> dict[str, float]:
    return {
        f"mean-{legacy_k:g}sd (legacy)": t_mean_k_sigma(h, legacy_k),
        "Otsu": t_otsu(h),
        "Li": t_li(h),
        "Kittler-Illingworth": t_kittler_illingworth(h),
        "Triangle": t_triangle(h),
        f"f = {primary_f:g} (primary)": t_partial_volume(I_s, I_p, primary_f),
        "Lab-matched": t_quantile(h, lab_porosity),
    }
