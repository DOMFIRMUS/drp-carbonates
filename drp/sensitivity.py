"""Sensitivity of the variance-volume analysis and of the porosity budget.

Added after peer-style review of the first manuscript draft. Nothing here
changes the primary results; every number is a sensitivity analysis reported in
the supplementary tables (T15, T16):

* variance-volume exponent (same windows, seed and fit as `step_rev`) on
  (i) the stack without the per-slice drift correction, with the end-members and
  thresholds of the uncorrected histogram, and (ii) the analysed block grids
  after removing a linear trend of the layer means along z. Windows of
  15-20 mm can move only a few millimetres across stacks 22-27 mm wide, so
  their dispersion mostly reflects the z profile;
* null model: stationary Gaussian fields smoothed with kernels of 1-5 mm
  (integral-range edge of the continuous field 2 sigma sqrt(pi) = 3.5-17.7 mm),
  thresholded to the porosity at f = 0.5 and continuous (mass balance), with
  and without the same z detrending. Over windows of 2-20 mm an exponent below 1
  is also what a stationary medium gives when its integral range is not small
  compared with the windows;
* how far the observed exponents lie from the 1-mm null of `step_rev`, in
  null standard deviations, and the standard error of the whole-stack porosity
  under each extrapolation, including a decay as 1/V beyond the largest window;
* solid end-member: lateral non-uniformity of the solid-peak grey level over a
  6 x 6 grid of tiles (beam hardening / cupping are not corrected by the
  per-slice gain), the mass-balance porosity with tile-wise solid end-members,
  and the sensitivity of the mass balance to I_s, including the I_s that would
  close the porosity budget without any sub-resolution porosity.

The parameters are module constants (not in config.toml) so that adding this
analysis does not change the configuration hash of the primary results.
"""
from __future__ import annotations

import logging

import numpy as np

from . import rev as revmod
from .context import SampleContext
from .util import load_json, save_json, timed
from .volume import histogram_mode

log = logging.getLogger("drp")

NULL_KERNELS_MM = (1.0, 2.0, 3.0, 5.0)   # SD of the Gaussian smoothing kernel
NULL_FIELDS = 20                         # fields per kernel (sensitivity only; the primary test uses 100)
TILES = 6                                # tiles per lateral axis
TILE_SLICES = 120                        # evenly spaced slices used for the tile histograms
TILE_BINS = 512


def detrend_z(grid: np.ndarray) -> np.ndarray:
    """Remove a linear trend of the layer means along z (axis 0), keeping the overall mean."""
    g = grid.astype(np.float64)
    m = g.mean(axis=(1, 2))
    z = np.arange(len(m), dtype=np.float64)
    slope, icpt = np.polyfit(z, m, 1)
    trend = slope * z + icpt
    g -= (trend - trend.mean())[:, None, None]
    return g


def _fit(grid, block, voxel_mm, cfg, n_win, seed, transform, phi, vol, conf) -> dict:
    rows = revmod.variance_curve(grid, block, voxel_mm, cfg["edges_mm"], n_win, seed=seed, transform=transform)
    domain = float(np.prod(grid.shape)) * (block * voxel_mm) ** 3
    f = revmod.integral_range(rows, phi, float(cfg["fit_min_mm"]), float(cfg["target_relative_error"]), vol, conf,
                              domain_volume_mm3=domain)
    big = [r for r in rows if r["var"] > 0][-1]
    cv20, v20 = big["cv"], big["volume_mm3"]
    eps = float(cfg["target_relative_error"])
    # if the variance decayed as 1/V beyond the largest window
    l_alpha1 = float((v20 * (2 * cv20 / eps) ** 2) ** (1 / 3))
    return {"alpha": f["alpha"], "alpha_finite_domain": f["alpha_finite_domain"], "cv_largest_window": cv20,
            "largest_window_mm": big["edge_mm"], "se_powerlaw": f["porosity_se_total_powerlaw"],
            "se_finite_domain": f["porosity_se_total_finite_domain"], "se_conservative": f["porosity_se_total_conservative"],
            "L_target_powerlaw_mm": f["L_target_mm_powerlaw"], "L_target_if_1_over_V_beyond_largest_mm": l_alpha1,
            "porosity": phi}


def _null_sensitivity(shape, block, voxel_mm, cfg, n_win, seed, phi_bin) -> dict:
    out = {}
    fit_min = float(cfg["fit_min_mm"])
    for k_mm in NULL_KERNELS_MM:
        corr_blocks = k_mm / (block * voxel_mm)
        rng = np.random.default_rng(seed + int(k_mm * 1000))
        res = {"binary": [], "binary_detrended": [], "continuous": [], "continuous_detrended": []}
        for i in range(NULL_FIELDS):
            g = revmod.gaussian_null_field(shape, corr_blocks, rng)
            t = np.quantile(g, phi_bin)
            b = (g < t).astype(np.float64) * block**3
            c = g.astype(np.float64)
            for key, grid in (("binary", b), ("binary_detrended", detrend_z(b)),
                              ("continuous", c), ("continuous_detrended", detrend_z(c))):
                a, _ = revmod._alphas_of(grid, block, voxel_mm, cfg["edges_mm"], n_win, seed + 11 * i, fit_min, None)
                res[key].append(a)
            del g, b, c
        out[f"{k_mm:g}"] = {
            "kernel_sd_mm": k_mm,
            "integral_range_edge_continuous_mm": float(2 * k_mm * np.sqrt(np.pi)),
            "n_fields": NULL_FIELDS,
            **{key: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                     "min": float(np.min(v)), "max": float(np.max(v))} for key, v in res.items()},
        }
        log.info("null kernel %.0f mm: alpha binary %.2f +/- %.2f (detrended %.2f), continuous %.2f (detrended %.2f)",
                 k_mm, out[f"{k_mm:g}"]["binary"]["mean"], out[f"{k_mm:g}"]["binary"]["sd"],
                 out[f"{k_mm:g}"]["binary_detrended"]["mean"], out[f"{k_mm:g}"]["continuous"]["mean"],
                 out[f"{k_mm:g}"]["continuous_detrended"]["mean"])
    return out


def _tiles(ctx: SampleContext) -> dict:
    """Solid-peak level and mass-balance porosity in a TILES x TILES lateral grid."""
    stack = ctx.stack()
    nz, ny, nx = stack.shape
    zs = np.unique(np.linspace(0, nz - 1, TILE_SLICES).round().astype(int))
    yb = np.linspace(0, ny, TILES + 1).astype(int)
    xb = np.linspace(0, nx, TILES + 1).astype(int)
    I_s, I_p = ctx.I_s, ctx.I_p
    lo, hi = I_p * 0.5, I_s * 1.4
    edges = np.linspace(lo, hi, TILE_BINS + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    hist = np.zeros((TILES, TILES, TILE_BINS))
    sums = np.zeros((TILES, TILES))
    cnt = np.zeros((TILES, TILES))
    for z in zs:
        g = stack.grey(int(z), int(z) + 1)[0].astype(np.float64)
        for i in range(TILES):
            for j in range(TILES):
                t = g[yb[i]:yb[i + 1], xb[j]:xb[j + 1]]
                hist[i, j] += np.histogram(t, edges)[0]
                sums[i, j] += t.sum()
                cnt[i, j] += t.size
    mode = np.array([[histogram_mode(hist[i, j], centers) for j in range(TILES)] for i in range(TILES)])
    mean = sums / cnt
    phi_global = (I_s - mean) / (I_s - I_p)
    phi_local = (mode - mean) / (mode - I_p)
    w = cnt / cnt.sum()
    rel = 100 * (mode - I_s) / I_s
    return {"tiles": TILES, "n_slices": int(len(zs)),
            "solid_mode_rel_percent": rel.tolist(),
            "solid_mode_range_percent": float(rel.max() - rel.min()),
            "solid_mode_rel_min_percent": float(rel.min()), "solid_mode_rel_max_percent": float(rel.max()),
            "mass_balance_tiles_global_endmembers": [float(phi_global.min()), float(phi_global.max())],
            "mass_balance_tiles_local_solid": [float(phi_local.min()), float(phi_local.max())],
            "mass_balance_weighted_global_endmembers": float((w * phi_global).sum()),
            "mass_balance_weighted_local_solid": float((w * phi_local).sum())}


def _endmember(ctx: SampleContext) -> dict:
    seg = ctx.seg
    I_s, I_p, mean = ctx.I_s, ctx.I_p, float(seg["mean"])
    lab = float(ctx.sample.lab_porosity)
    phi = (I_s - mean) / (I_s - I_p)
    dphi_dIs = (mean - I_p) / (I_s - I_p) ** 2
    I_s_close = (mean - lab * I_p) / (1 - lab)
    return {"I_s": I_s, "I_p": I_p, "mean_grey": mean, "mass_balance": phi,
            "dphi_per_1pc_Is": float(dphi_dIs * 0.01 * I_s),
            "I_s_closing_budget": float(I_s_close),
            "I_s_closing_budget_rel_percent": float(100 * (I_s_close - I_s) / I_s)}


def step_sensitivity(ctx: SampleContext):
    cfg = ctx.cfg["rev"]
    block = int(cfg["block_voxels"])
    conf = float(ctx.cfg["stats"]["confidence"])
    seed = ctx.cfg.seed
    n_win = int(ctx.cfg.scaled("rev", "n_windows"))
    vol = float(np.prod(ctx.shape)) * ctx.voxel_mm**3
    rev = load_json(ctx.data("rev"))
    cache = ctx.cfg.cache_dir / f"{ctx.sample.id}_rev_blocks.npz"
    with np.load(cache) as z:
        grids = {k: z[k] for k in ("primary", "legacy", "mixing")}
    I_s, I_p = ctx.I_s, ctx.I_p
    unc = ctx.seg["uncorrected"]
    Is0, Ip0 = float(unc["I_s"]), float(unc["I_p"])
    out = {"as_analysed": {}, "z_detrended": {}, "uncorrected_stack": {}, "distance_from_null": {}, "se_ratios": {}}

    # ---- exponent: detrended along z and without drift correction
    with timed(log, f"sensitivity: detrended and uncorrected exponents {ctx.sample.id}"):
        cache_u = ctx.cfg.cache_dir / f"{ctx.sample.id}_rev_blocks_uncorrected.npz"
        T_u = {"primary": float(unc["thresholds_by_key"]["primary"]["T"]),
               "legacy": float(unc["thresholds_by_key"]["legacy"]["T"])}
        key = {"T_primary": T_u["primary"], "T_legacy": T_u["legacy"], "block": block}
        grids_u = {}
        if cache_u.exists():
            with np.load(cache_u) as z:
                if all(np.isclose(float(z[f"meta_{k}"]), float(v)) for k, v in key.items()):
                    grids_u = {k: z[k] for k in ("primary", "legacy", "mixing")}
        if not grids_u:
            raw = ctx.stack(corrected=False)
            grids_u = {"primary": revmod.block_counts(raw, T_u["primary"], block),
                       "legacy": revmod.block_counts(raw, T_u["legacy"], block),
                       "mixing": revmod.block_grey_sums(raw, block)}
            np.savez_compressed(cache_u, **grids_u, **{f"meta_{k}": float(v) for k, v in key.items()})
        for tag in ("primary", "legacy", "mixing"):
            mix = tag == "mixing"
            tr = (lambda g: (I_s - g) / (I_s - I_p)) if mix else None
            tr_u = (lambda g: (Is0 - g) / (Is0 - Ip0)) if mix else None
            g = grids[tag]
            n = g.size * block**3
            phi = float(tr(g.sum() / n)) if mix else float(g.sum() / n)
            gu = grids_u[tag]
            phi_u = float(tr_u(gu.sum() / n)) if mix else float(gu.sum() / n)
            out["as_analysed"][tag] = _fit(g, block, ctx.voxel_mm, cfg, n_win, seed, tr, phi, vol, conf)
            out["z_detrended"][tag] = _fit(detrend_z(g), block, ctx.voxel_mm, cfg, n_win, seed, tr, phi, vol, conf)
            out["uncorrected_stack"][tag] = _fit(gu, block, ctx.voxel_mm, cfg, n_win, seed, tr_u, phi_u, vol, conf)
            f = rev[tag]["fit"]
            nl = f["null"]
            out["distance_from_null"][tag] = {
                "alpha": f["alpha"], "null_mean": nl["null_mean"], "null_sd": nl["null_sd"],
                "sd_below_null_mean": float((nl["null_mean"] - f["alpha"]) / nl["null_sd"]),
                "finite_domain_null_band": [f["null_finite_domain"]["null_q025"], f["null_finite_domain"]["null_q975"]],
                "alpha_finite_domain": f["alpha_finite_domain"]}
            se1 = f["porosity_se_total_if_alpha1"]
            out["se_ratios"][tag] = {"se_if_alpha1": se1,
                                     "ratio_powerlaw_to_alpha1": float(f["porosity_se_total_powerlaw"] / se1),
                                     "ratio_conservative_to_alpha1": float(f["porosity_se_total_conservative"] / se1)}
            log.info("%s %s: alpha %.2f | detrended %.2f | uncorrected stack %.2f; CV20 %.3f / %.3f / %.3f",
                     ctx.sample.id, tag, out["as_analysed"][tag]["alpha"], out["z_detrended"][tag]["alpha"],
                     out["uncorrected_stack"][tag]["alpha"], out["as_analysed"][tag]["cv_largest_window"],
                     out["z_detrended"][tag]["cv_largest_window"], out["uncorrected_stack"][tag]["cv_largest_window"])

    # ---- null model with longer correlation scales (and detrended)
    with timed(log, f"sensitivity: null kernels {ctx.sample.id}"):
        out["null_kernels"] = _null_sensitivity(grids["primary"].shape, block, ctx.voxel_mm, cfg, n_win, seed,
                                                out["as_analysed"]["primary"]["porosity"])

    # ---- solid end-member
    with timed(log, f"sensitivity: solid end-member {ctx.sample.id}"):
        out["solid_tiles"] = _tiles(ctx)
        out["endmember"] = _endmember(ctx)
    save_json(ctx.data("sensitivity"), out)
