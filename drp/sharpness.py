"""Effective spatial resolution from edge-spread profiles.

Voxel size is a sampling interval, not a resolution: the reconstruction also
blurs the image with a point-spread function (PSF). Across a pore wall much
larger than the PSF, the grey level follows an error function whose width is
the PSF standard deviation sigma (edge-spread function, ESF):

    I(s) = b + a * Phi((s - s0) / sigma)

Profiles are sampled along the grey-level gradient at points of the
partial-volume iso-surface, fitted individually, and only profiles that reach
both plateaus (contrast >= `min_contrast` of the end-member contrast) and fit
well are kept. The median sigma over those profiles approximates the PSF
width: interface roughness and curvature widen edges, whereas a second
interface inside the fitting window (e.g. the opposite wall of a pore some
14-18 voxels wide) narrows them by a few per cent (tests: flat slab).
FWHM = 2.355 sigma; the 10-90 % edge rise distance = 2.563 sigma.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.special import erf


def _esf(s, b, a, s0, sigma):
    return b + a * 0.5 * (1 + erf((s - s0) / (np.sqrt(2) * sigma)))


def edge_profiles(grey: np.ndarray, level: float, tol: float, n_edges: int, half_width: int,
                  seed: int = 0, step: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Grey-level profiles along the gradient through iso-surface points."""
    g = grey.astype(np.float32)
    sm = ndi.gaussian_filter(g, 1.0)
    grads = [ndi.gaussian_filter(g, 1.5, order=o) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    gmag = np.sqrt(sum(gr**2 for gr in grads))
    m = half_width + 2
    interior = np.zeros(g.shape, bool)
    interior[m:-m, m:-m, m:-m] = True
    near = (np.abs(sm - level) < tol) & interior & (gmag > 0)
    thr = np.percentile(gmag[near], 50) if near.any() else 0
    cand = np.argwhere(near & (gmag >= thr))
    rng = np.random.default_rng(seed)
    if len(cand) > n_edges:
        cand = cand[rng.choice(len(cand), n_edges, replace=False)]
    normals = np.stack([gr[tuple(cand.T)] for gr in grads], 1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    s = np.arange(-half_width, half_width + step / 2, step)
    coords = cand[:, None, :] + s[None, :, None] * normals[:, None, :]
    prof = ndi.map_coordinates(g, coords.reshape(-1, 3).T, order=1, mode="nearest")
    return s, prof.reshape(len(cand), len(s))


def fit_profiles(s: np.ndarray, profiles: np.ndarray, contrast: float,
                 min_contrast: float, min_r2: float) -> dict:
    """Fit the ESF model to every profile; keep full-contrast, well-fitted edges."""
    params, r2s = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        for p in profiles:
            lo, hi = np.percentile(p, [5, 95])
            try:
                popt, _ = curve_fit(_esf, s, p, p0=[lo, hi - lo, 0.0, 2.0], maxfev=2000,
                                    bounds=([-np.inf, 0, -3, 0.2], [np.inf, np.inf, 3, s.max() / 2]))
            except (RuntimeError, ValueError):
                popt = np.full(4, np.nan)
            pred = _esf(s, *popt)
            ss_tot = np.sum((p - p.mean()) ** 2)
            r2s.append(1 - np.sum((p - pred) ** 2) / ss_tot if ss_tot > 0 else 0.0)
            params.append(popt)
    params = np.asarray(params)
    r2s = np.asarray(r2s)
    b, a, s0, sig = params.T
    keep = np.isfinite(sig) & (a >= min_contrast * contrast) & (r2s >= min_r2) & (sig < s.max() / 2 - 0.1)
    out = {"n_profiles": int(len(profiles)), "n_kept": int(keep.sum()),
           "sigma_kept": sig[keep], "params": params, "r2": r2s, "keep": keep}
    if keep.sum() >= 10:
        q = np.percentile(sig[keep], [25, 50, 75])
        out.update({"sigma_q25": float(q[0]), "sigma_median": float(q[1]), "sigma_q75": float(q[2])})
        # average normalised edge, re-centred on each fitted s0
        grid = np.arange(-8, 8.01, 0.25)
        aligned = [np.interp(grid, s - s0[i], (profiles[i] - b[i]) / a[i]) for i in np.nonzero(keep)[0]]
        out["aligned_grid"] = grid
        out["aligned_mean"] = np.mean(aligned, axis=0)
        out["aligned_q25"], out["aligned_q75"] = np.percentile(aligned, [25, 75], axis=0)
    return out
