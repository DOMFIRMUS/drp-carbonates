"""Representative elementary volume by the integral-range method.

For a stationary, ergodic two-phase medium the variance of the porosity
measured in windows of volume V decays as (Matheron 1971; Kanit et al. 2003)

    Var[phi_V] = phi (1 - phi) A3 / V          for V >> A3,

where A3 is the integral range (the volume over which the microstructure is
correlated). More generally Var[phi_V] = K V^-alpha: alpha = 1 means a finite
integral range, alpha < 1 long-range correlation, trends or
non-stationarity, in which case no REV exists at the scale of the sample.

Finite domain. The windows are placed inside one imaged volume S, so their
porosities scatter around the mean of S, not around the mean of the
material: what is observed is (approximately) the dispersion variance of v
within S, which by Krige's additivity relation is D2(v|S) = D2(v) - D2(S)
(Matheron 1971; Lantuejoul 1991). With the power law this suggests
Var_obs(V) = K (V^-alpha - V_S^-alpha). Large windows cannot move freely in a
section only slightly wider than they are, so the relation is approximate;
the classical straight-line fit of log Var against log V (as in Kanit et al.
2003) is therefore kept as the primary estimator and the finite-domain fit is
reported as a sensitivity analysis. For the variance of the whole stack the
larger (more conservative) of the two extrapolations is used.

Uncertainty. The variance points come from nested, overlapping windows in a
single specimen, so the OLS standard error of the slope measures curvature,
not sampling error. Two calibrations replace it:
* a moving-block bootstrap along z (slabs of the block-count grid resampled
  and concatenated; each replicate is re-analysed), whose SD is reported as
  the SE of alpha. Only z is resampled, so it is a lower bound;
* a Monte Carlo test against stationary media with a finite integral range:
  thresholded Gaussian fields with the porosity of the sample and a
  correlation length longer than the measured one, analysed with exactly the
  same windows and fit. The null distribution of alpha also absorbs any bias
  of the estimator in a finite domain.

Porosity in arbitrary cubic windows is obtained from a coarse grid of block
pore counts and a 3-D summed-area table, so thousands of windows cost
nothing and no voxel is read twice.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy import optimize, stats


def block_counts(stack, threshold: float, block: int, slab_blocks: int = 8) -> np.ndarray:
    """Pore-voxel counts in non-overlapping block^3 cells of the whole stack."""
    nz, ny, nx = stack.shape
    kz, ky, kx = nz // block, ny // block, nx // block
    out = np.zeros((kz, ky, kx), np.int32)
    step = block * slab_blocks
    for z0 in range(0, kz * block, step):
        z1 = min(z0 + step, kz * block)
        m = stack.grey(z0, z1)[:, : ky * block, : kx * block] < threshold
        nb = (z1 - z0) // block
        out[z0 // block : z0 // block + nb] = (
            m.reshape(nb, block, ky, block, kx, block).sum(axis=(1, 3, 5), dtype=np.int32)
        )
    return out


def block_grey_sums(stack, block: int, slab_blocks: int = 8) -> np.ndarray:
    """Sum of grey levels in non-overlapping block^3 cells (for the mass balance)."""
    nz, ny, nx = stack.shape
    kz, ky, kx = nz // block, ny // block, nx // block
    out = np.zeros((kz, ky, kx), np.float64)
    step = block * slab_blocks
    for z0 in range(0, kz * block, step):
        z1 = min(z0 + step, kz * block)
        g = stack.grey(z0, z1)[:, : ky * block, : kx * block].astype(np.float64)
        nb = (z1 - z0) // block
        out[z0 // block : z0 // block + nb] = g.reshape(nb, block, ky, block, kx, block).sum(axis=(1, 3, 5))
    return out


def _sat(counts: np.ndarray) -> np.ndarray:
    dt = np.float64 if counts.dtype.kind == "f" else np.int64
    s = counts.astype(dt).cumsum(0).cumsum(1).cumsum(2)
    return np.pad(s, ((1, 0), (1, 0), (1, 0)))


def window_porosity(sat: np.ndarray, starts: np.ndarray, n: int, block: int) -> np.ndarray:
    """Mean per voxel of the counted quantity in cubic windows of n blocks."""
    z, y, x = starts.T
    Z, Y, X = z + n, y + n, x + n
    total = (sat[Z, Y, X] - sat[z, Y, X] - sat[Z, y, X] - sat[Z, Y, x]
             + sat[z, y, X] + sat[z, Y, x] + sat[Z, y, x] - sat[z, y, x])
    return total / float((n * block) ** 3)


def variance_curve(counts: np.ndarray, block: int, voxel_mm: float, edges_mm, n_windows: int,
                   seed: int = 0, transform=None, min_target_mm: float | None = None) -> list[dict]:
    """Mean, variance and CV of window porosity versus window edge.

    `transform` maps the window mean of the counted quantity to porosity
    (e.g. grey level -> mass-balance porosity). Each row keeps the requested
    edge (`target_mm`) and the realised one (`edge_mm`, whole blocks).
    """
    rng = np.random.default_rng(seed)
    sat = _sat(counts)
    kz, ky, kx = counts.shape
    rows = []
    for L in edges_mm:
        if min_target_mm is not None and L < min_target_mm:
            continue
        n = int(round(L / (block * voxel_mm)))
        if n < 1 or n > min(kz, ky, kx):
            continue
        hi = np.array([kz - n, ky - n, kx - n]) + 1
        starts = np.stack([rng.integers(0, h, n_windows) for h in hi], 1)
        phi = window_porosity(sat, starts, n, block)
        if transform is not None:
            phi = transform(phi)
        edge = n * block * voxel_mm
        rows.append({"target_mm": float(L), "edge_mm": edge, "volume_mm3": edge**3, "n_windows": n_windows,
                     "mean": float(phi.mean()), "var": float(phi.var(ddof=1)),
                     "sd": float(phi.std(ddof=1)), "cv": float(phi.std(ddof=1) / phi.mean()),
                     "p05": float(np.percentile(phi, 5)), "p95": float(np.percentile(phi, 95))})
    return rows


def _select(rows, fit_min_mm):
    return [r for r in rows if r.get("target_mm", r["edge_mm"]) >= fit_min_mm - 1e-9 and r["var"] > 0]


def fit_powerlaw(V, var, V_domain=None) -> dict:
    """Straight-line fit log Var = log K - alpha log V (primary) and, if
    V_domain is given, the finite-domain fit Var = K (V^-a - V_domain^-a).

    Keys: alpha/K/r2 (straight line) and alpha_fd/K_fd/r2_fd (finite domain).
    The OLS standard error of the slope is returned only as a descriptive
    goodness-of-fit number (`alpha_se_ols`); it is not a sampling error.
    """
    V, var = np.asarray(V, float), np.asarray(var, float)
    ols = stats.linregress(np.log(V), np.log(var))
    a0, k0 = -ols.slope, np.exp(ols.intercept)
    out = {"alpha": float(a0), "K": float(k0), "r2": float(ols.rvalue**2), "alpha_se_ols": float(ols.stderr)}
    if V_domain is None:
        return out
    lv = np.log(var)

    def resid(p):
        a, lk = p
        d = V ** (-a) - V_domain ** (-a)
        return lk + np.log(np.maximum(d, 1e-300)) - lv

    best = None
    for a_start in (a0, 0.5 * a0, 1.0):
        a_start = float(np.clip(a_start, 0.02, 2.5))
        lk0 = float(np.mean(lv - np.log(V ** (-a_start) - V_domain ** (-a_start))))
        r = optimize.least_squares(resid, [a_start, lk0], bounds=([0.01, -60], [3.0, 60]))
        if best is None or r.cost < best.cost:
            best = r
    a, lk = best.x
    pred = lk + np.log(V ** (-a) - V_domain ** (-a))
    ss_res, ss_tot = np.sum((lv - pred) ** 2), np.sum((lv - lv.mean()) ** 2)
    out.update({"alpha_fd": float(a), "K_fd": float(np.exp(lk)),
                "r2_fd": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan, "V_domain_mm3": float(V_domain)})
    return out


def integral_range(rows: list[dict], phi: float, fit_min_mm: float, target_eps: float,
                   total_volume_mm3: float, confidence: float = 0.95,
                   domain_volume_mm3: float | None = None) -> dict:
    """Fit the variance-volume law on the large-window tail; derive REV quantities.

    Windows are selected by their requested edge (>= fit_min_mm), and the
    realised range is reported. `domain_volume_mm3` is the volume inside which
    the windows were placed (default: the total volume), used by the
    finite-domain sensitivity fit. If alpha < 1 the integral range A3 is not
    defined; the power law itself is used to extrapolate the variance to the
    whole volume and to the target error. Sampling uncertainty of alpha is
    added by `alpha_block_bootstrap` and `alpha_null_test`.
    """
    sel = _select(rows, fit_min_mm)
    if len(sel) < 3:
        return {"alpha": np.nan}
    V = np.array([r["volume_mm3"] for r in sel])
    var = np.array([r["var"] for r in sel])
    dom = float(domain_volume_mm3 or total_volume_mm3)
    fp = fit_powerlaw(V, var, dom)
    alpha, K = fp["alpha"], fp["K"]
    se_total = float(np.sqrt(K * total_volume_mm3 ** (-alpha)))
    se_total_fd = float(np.sqrt(fp["K_fd"] * total_volume_mm3 ** (-fp["alpha_fd"])))
    se_cons = max(se_total, se_total_fd)
    v_target = (4 * K / (target_eps * phi) ** 2) ** (1 / alpha) if alpha > 0 else np.inf
    out = {
        "alpha": float(alpha), "K": float(K), "r2": fp["r2"], "alpha_se_ols_descriptive": fp["alpha_se_ols"],
        "alpha_finite_domain": fp["alpha_fd"], "K_finite_domain": fp["K_fd"], "r2_finite_domain": fp["r2_fd"],
        "domain_volume_mm3": dom, "n_points": int(len(V)),
        "fit_min_mm": float(min(r["edge_mm"] for r in sel)), "fit_min_requested_mm": float(fit_min_mm),
        "fit_max_mm": float(max(r["edge_mm"] for r in sel)),
        "porosity_se_total_powerlaw": se_total, "porosity_se_total_finite_domain": se_total_fd,
        "porosity_se_total_conservative": se_cons,
        "relative_error95_total_powerlaw": float(2 * se_cons / phi),
        "target_relative_error": target_eps,
        "V_target_mm3_powerlaw": float(v_target), "L_target_mm_powerlaw": float(v_target ** (1 / 3)),
        "L_target_is_extrapolated": bool(v_target ** (1 / 3) > max(r["edge_mm"] for r in sel)),
        "cv_at_largest_window": float(sel[-1]["cv"]), "largest_window_mm": float(sel[-1]["edge_mm"]),
    }
    # classical integral range (meaningful only if alpha ~ 1), for reference
    a3_each = var / (1 - V / dom) * V / (phi * (1 - phi))
    a3 = float(np.exp(np.mean(np.log(a3_each))))
    out.update({
        "A3_mm3_if_alpha1": a3, "A3_edge_mm_if_alpha1": a3 ** (1 / 3),
        "porosity_se_total_if_alpha1": float(np.sqrt(phi * (1 - phi) * a3 / total_volume_mm3)),
    })
    return out


def _alphas_of(counts, block, voxel_mm, edges_mm, n_windows, seed, fit_min_mm, transform):
    """(alpha, alpha_finite_domain) of one block grid, with the production windows and fit."""
    rows = variance_curve(counts, block, voxel_mm, edges_mm, n_windows, seed, transform, min_target_mm=fit_min_mm)
    sel = _select(rows, fit_min_mm)
    if len(sel) < 3:
        return np.nan, np.nan
    vol = float(np.prod(counts.shape)) * (block * voxel_mm) ** 3
    V = np.array([r["volume_mm3"] for r in sel])
    var = np.array([r["var"] for r in sel])
    fp = fit_powerlaw(V, var, vol)
    return fp["alpha"], fp["alpha_fd"]


def alpha_block_bootstrap(counts, block, voxel_mm, edges_mm, n_windows, fit_min_mm, slab_mm, n_boot,
                          confidence=0.95, seed=0, transform=None) -> dict:
    """SE of alpha from a moving-block bootstrap of z-slabs of the block grid.

    Slabs of `slab_mm` along z are drawn with wrap-around and concatenated to
    the original length; each replicate is re-analysed (summed-area table,
    windows, fit). The interval is alpha +/- z SD (the percentile interval is
    not used: concatenation breaks long-range structure and pulls replicates
    towards alpha = 1). Only z is resampled, so this is a lower bound.
    """
    rng = np.random.default_rng(seed)
    kz = counts.shape[0]
    L = int(max(1, round(slab_mm / (block * voxel_mm))))
    nb = int(np.ceil(kz / L))
    al = []
    for b in range(n_boot):
        starts = rng.integers(0, kz, nb)
        idx = ((starts[:, None] + np.arange(L)[None, :]) % kz).ravel()[:kz]
        al.append(_alphas_of(counts[idx], block, voxel_mm, edges_mm, n_windows, seed + 1 + b, fit_min_mm, transform))
    al = np.asarray(al, float).reshape(-1, 2)
    al = al[np.isfinite(al).all(axis=1)]
    zc = stats.norm.ppf(0.5 + confidence / 2)
    return {"alpha_se_boot": float(al[:, 0].std(ddof=1)) if len(al) > 1 else np.nan,
            "alpha_finite_domain_se_boot": float(al[:, 1].std(ddof=1)) if len(al) > 1 else np.nan,
            "boot_mean": float(al[:, 0].mean()) if len(al) else np.nan,
            "n_boot": int(len(al)), "slab_mm": L * block * voxel_mm, "z": float(zc),
            "method": "moving-block bootstrap of z-slabs (lower bound: x and y not resampled)"}


def gaussian_null_field(shape, corr_blocks: float, rng) -> np.ndarray:
    """Stationary Gaussian field with a Gaussian covariance (finite integral range)."""
    w = rng.standard_normal(shape).astype(np.float32)
    f = ndi.gaussian_filter(w, corr_blocks, mode="wrap")
    return (f - f.mean()) / f.std()


def alpha_null_test(alpha_obs: dict, shape, block, voxel_mm, edges_mm, n_windows, fit_min_mm, porosity: dict,
                    corr_mm: float, n_sim: int, seed=0) -> dict:
    """Null distribution of alpha for stationary media with a finite integral range.

    One Gaussian field per realisation (same block grid as the sample, Gaussian
    covariance of length `corr_mm`) is analysed (i) thresholded to each
    binary porosity in `porosity` and (ii) as a continuous field (key with
    porosity None, for the grey-level mass balance, which is not binary).
    `alpha_obs[key]` = (alpha, alpha_finite_domain). Returns per key the null
    band and the one-sided p = (#{alpha_null <= alpha_obs} + 1)/(N + 1) for
    both estimators.
    """
    rng = np.random.default_rng(seed)
    corr_blocks = corr_mm / (block * voxel_mm)
    sims = {k: [] for k in alpha_obs}
    nvox = block**3
    for i in range(n_sim):
        g = gaussian_null_field(shape, corr_blocks, rng)
        for key in alpha_obs:
            phi = porosity.get(key)
            if phi is None:           # continuous field
                c = g.astype(np.float64)
            else:
                t = np.quantile(g, phi)
                c = (g < t).astype(np.int32) * nvox
            sims[key].append(_alphas_of(c, block, voxel_mm, edges_mm, n_windows, seed + 7 * i, fit_min_mm, None))
            del c
        del g
    out = {"corr_mm": corr_mm, "n_sim": n_sim, "model": "stationary Gaussian field, Gaussian covariance"}
    for key, (a_obs, a_fd) in alpha_obs.items():
        s = np.asarray(sims[key], float).reshape(-1, 2)
        s = s[np.isfinite(s).all(axis=1)]
        res = {}
        for j, (name, a) in enumerate((("alpha", a_obs), ("alpha_finite_domain", a_fd))):
            x = s[:, j]
            res[name] = {"null_mean": float(x.mean()), "null_sd": float(x.std(ddof=1)), "n_valid": int(len(x)),
                         "null_q025": float(np.quantile(x, 0.025)), "null_q975": float(np.quantile(x, 0.975)),
                         "p_one_sided": float((np.sum(x <= a) + 1) / (len(x) + 1)), "observed": float(a)}
        out[key] = res
    return out
