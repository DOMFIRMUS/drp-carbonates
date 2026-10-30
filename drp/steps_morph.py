"""Pipeline steps 7-11: morphology on equal physical volumes.

7. cube        - central cube of the same physical edge in both samples:
                 S_V (all estimators), box-counting D, two-point correlation,
                 connectivity, and a sweep over segmentation criteria.
8. psd         - continuous pore-size distribution (smaller central cube).
9. esf         - effective resolution from edge-spread profiles.
10. replicates - non-overlapping sub-volumes tiling each whole stack.
11. resolution - block-averaging experiment (coarser voxels, same criterion).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import connectivity as conn
from . import correlation, fractal, poresize, sharpness, surface
from .context import SampleContext
from .util import save_json, timed
from .volume import ArraySource, GreySource, central_cube, downsample, materialize, tile_cubes

log = logging.getLogger("drp")


class ThresholdView:
    """Pore mask (grey < T) of an in-memory/memmapped grey array, per slab."""

    def __init__(self, grey: np.ndarray, T: float):
        self.grey, self.T = grey, T
        self.shape = grey.shape

    def slab(self, z0, z1):
        return np.asarray(self.grey[z0:z1]) < self.T


def _box_sizes(ctx_voxel_um, fcfg, edge_vox, s_min_mm=None, s_max_mm=None, n=None):
    return fractal.box_sizes(ctx_voxel_um, s_min_mm or fcfg["box_min_mm"], s_max_mm or fcfg["box_max_mm"],
                             n or int(fcfg["n_box_sizes"]), int(fcfg["min_box_voxels"]), max(2, edge_vox // 4))


def _fractal_block(src, voxel_um, fcfg, fit_min_mm, fit_max_mm, n_offsets, seed, sizes=None,
                   mode="boundary"):
    edge = min(src.shape)
    sizes = sizes if sizes is not None else _box_sizes(voxel_um, fcfg, edge)
    bc = fractal.box_count(src, sizes, n_offsets=n_offsets, seed=seed, mode=mode)
    size_mm = bc["size_vox"] * voxel_um / 1000
    fit = fractal.fit_dimension(size_mm, bc["count"], fit_min_mm, fit_max_mm)
    return {"size_vox": bc["size_vox"], "size_mm": size_mm, "count": bc["count"], "count_sd": bc["count_sd"],
            "boxes": bc["boxes"], "local_slope": fractal.local_slope(size_mm, bc["count"]), "fit": fit}


def _fit_fields(fit: dict, prefix: str = "D") -> dict:
    """Flatten a fit_dimension result into row fields (D, SE, t-based CI, realised range)."""
    return {prefix: fit.get("D"), f"{prefix}_se": fit.get("se"), f"{prefix}_ci_low": fit.get("ci_low"),
            f"{prefix}_ci_high": fit.get("ci_high"), f"{prefix}_r2": fit.get("r2"),
            f"{prefix}_fit_min_mm": fit.get("s_min"), f"{prefix}_fit_max_mm": fit.get("s_max"),
            f"{prefix}_n_points": fit.get("n_points")}


# ---------------------------------------------------------------------------
# 7. Central cube
# ---------------------------------------------------------------------------
def step_cube(ctx: SampleContext):
    cfg = ctx.cfg
    fcfg = cfg["fractal"]
    edge_mm = float(cfg.scaled("volumes", "central_cube_mm"))
    stack = ctx.stack()
    box = central_cube(stack.shape, ctx.vox(edge_mm))
    v_um, v_mm = ctx.voxel_um, ctx.voxel_mm
    with timed(log, f"materialize central cube {ctx.sample.id} {box.shape}"):
        grey = materialize(GreySource(stack, box), cfg.cache_dir / f"{ctx.sample.id}_cube_grey.npy").arr
    out = {"edge_mm": box.shape[0] * v_mm, "box": box.as_dict(), "shape": list(box.shape)}
    arrays = {}
    T = ctx.T_primary
    mask_src = ThresholdView(grey, T)
    grey_src = ArraySource(grey)

    with timed(log, f"S_V estimators {ctx.sample.id}"):
        out["surface"] = surface.specific_surface(mask_src, v_mm, grey_src, T)
    n_off = int(cfg.scaled("fractal", "n_offsets"))
    # all box sizes, and the subset inside the common fitting range whose end
    # points are whole-voxel sizes inside [fit_min, fit_max] for every voxel size
    sizes_all, sizes_fit = fractal.grid_with_fit_range(
        v_um, float(fcfg["box_min_mm"]), float(fcfg["box_max_mm"]), float(fcfg["fit_min_mm"]),
        float(fcfg["fit_max_mm"]), int(fcfg["n_fit_sizes"]), int(fcfg["n_box_sizes"]), int(fcfg["min_box_voxels"]),
        max(2, min(box.shape) // 4))
    s_lo, s_hi = sizes_fit[0] * v_mm, sizes_fit[-1] * v_mm
    with timed(log, f"box counting {ctx.sample.id}"):
        bnd = materialize(fractal.BoundarySource(mask_src), cfg.cache_dir / f"{ctx.sample.id}_cube_boundary.npy")
        fr = _fractal_block(bnd, v_um, fcfg, s_lo, s_hi, n_off, cfg.seed, sizes=sizes_all, mode="occupied")
        # porespy-style definition on the same cube, for comparison
        fr_mixed = _fractal_block(mask_src, v_um, fcfg, s_lo, s_hi, 2, cfg.seed,
                                  sizes=fr["size_vox"], mode="mixed")
        arrays["box_count_mixed"] = fr_mixed["count"]
        del bnd
        fr["windows"] = fractal.window_sensitivity(fr["size_mm"], fr["count"], float(fcfg["min_window_decades"]))
        out["fractal"] = {k: fr[k] for k in ("fit", "windows")}
        out["fractal"]["fit_mixed_definition"] = fr_mixed["fit"]
        for k in ("size_vox", "size_mm", "count", "count_sd", "boxes", "local_slope"):
            arrays[f"box_{k}"] = fr[k]
    with timed(log, f"two-point correlation {ctx.sample.id}"):
        s2 = correlation.two_point(mask_src, max_lag=ctx.vox(4.0))
        out["correlation"] = {}
        for a in "zyx":
            arrays[f"s2_{a}_lag_mm"] = s2[a]["lag_vox"] * v_mm
            arrays[f"s2_{a}"] = s2[a]["s2"]
            arrays[f"c_{a}"] = s2[a]["c"]
            out["correlation"][a] = correlation.correlation_lengths(s2[a]["lag_vox"] * v_mm, s2[a]["c"])
    out["connectivity"] = {}
    for c in (6, 26):
        with timed(log, f"connectivity-{c} {ctx.sample.id}"):
            lab = conn.label_clusters(mask_src, c, slab=48)
        out["connectivity"][str(c)] = conn.summarize(lab, v_mm)
        arrays[f"cluster_volumes_{c}"] = lab["cluster_volumes"]

    # Robustness sweep over the segmentation criterion (plus the legacy rule)
    sweep = []
    crit = [("f", float(f), ctx.T_f(float(f))) for f in cfg["segmentation"]["f_sweep"]]
    crit.append(("legacy", np.nan, ctx.T_legacy))
    # same fitting sizes and grid offsets as the central-cube fit, so the
    # sweep at f = 0.5 reproduces T09 and both samples share the fitting range
    for kind, f, Tf in crit:
        with timed(log, f"sweep {ctx.sample.id} {kind} f={f} T={Tf:.0f}"):
            src = ThresholdView(grey, Tf)
            counts = surface.transition_counts(src)
            phi = counts["n_pore"] / box.size
            fr_s = _fractal_block(src, v_um, fcfg, s_lo, s_hi, n_off, cfg.seed, sizes_fit)
            cs = conn.summarize(conn.label_clusters(src, 6, slab=48), v_mm)
            sweep.append({"criterion": kind, "f": f, "T": Tf, "porosity": phi,
                          "sv_crofton13": surface.sv_crofton(counts, v_mm), **_fit_fields(fr_s["fit"]),
                          "largest_cluster_fraction": cs.get("largest_cluster_fraction"),
                          "percolating_porosity_any": cs.get("percolating_porosity_any"),
                          "largest_spans_z": cs.get("largest_spans_z")})
    out["sweep"] = sweep
    np.savez_compressed(ctx.data("cube", "npz"), **arrays)
    save_json(ctx.data("cube"), out)
    s = out["surface"]
    log.info("%s cube: phi=%.4f Sv[crofton13]=%.3f Sv[mc grey]=%.3f /mm, D=%.3f (R2=%.4f)", ctx.sample.id,
             s["porosity"], s["sv_crofton13"], s.get("sv_mc_grey", np.nan), out["fractal"]["fit"]["D"],
             out["fractal"]["fit"]["r2"])


# ---------------------------------------------------------------------------
# 8. Pore-size distribution
# ---------------------------------------------------------------------------
def step_psd(ctx: SampleContext):
    cfg = ctx.cfg
    stack = ctx.stack()
    box = central_cube(stack.shape, ctx.vox(float(cfg.scaled("volumes", "psd_cube_mm"))))
    radii_um = np.asarray(cfg["psd"]["radii_um"], float)
    radii_vox = radii_um / ctx.voxel_um
    out = {"box": box.as_dict(), "edge_mm": box.shape[0] * ctx.voxel_mm, "curves": {}}
    grey = materialize(GreySource(stack, box)).arr
    for tag, T in (("primary", ctx.T_primary), ("legacy", ctx.T_legacy)):
        with timed(log, f"PSD {ctx.sample.id} {tag}"):
            res = poresize.continuous_psd(grey < T, radii_vox)
        out["curves"][tag] = {
            "radius_um": radii_um, "radius_vox": radii_vox, "fraction_ge": res["fraction_ge"],
            "median_radius_um": poresize.median_radius(radii_um, res["fraction_ge"]),
            "fraction_radius_lt_2vox": float(1 - np.interp(2.0, radii_vox, res["fraction_ge"])),
            "porosity": float((grey < T).mean()),
        }
    save_json(ctx.data("psd"), out)


# ---------------------------------------------------------------------------
# 9. Effective resolution
# ---------------------------------------------------------------------------
def step_esf(ctx: SampleContext):
    cfg = ctx.cfg
    e = cfg["esf"]
    stack = ctx.stack()
    box = central_cube(stack.shape, ctx.vox(float(cfg.scaled("volumes", "esf_cube_mm"))))
    grey = materialize(GreySource(stack, box)).arr
    level = 0.5 * (ctx.I_s + ctx.I_p)
    contrast = ctx.I_s - ctx.I_p
    with timed(log, f"edge profiles {ctx.sample.id}"):
        s, prof = sharpness.edge_profiles(grey, level, 0.05 * contrast, int(cfg.scaled("esf", "n_edges")),
                                          int(e["half_width_voxels"]), seed=cfg.seed)
        res = sharpness.fit_profiles(s, prof, contrast, float(e["min_contrast"]), float(e["min_r2"]))
    out = {"level": level, "contrast": contrast, "n_profiles": res["n_profiles"], "n_kept": res["n_kept"],
           "voxel_um": ctx.voxel_um}
    if "sigma_median" in res:
        for k in ("sigma_q25", "sigma_median", "sigma_q75"):
            out[k + "_vox"] = res[k]
            out[k + "_um"] = res[k] * ctx.voxel_um
        out["fwhm_um"] = 2.3548 * out["sigma_median_um"]
        out["edge_10_90_um"] = 2.5631 * out["sigma_median_um"]
        np.savez_compressed(ctx.data("esf", "npz"), grid_vox=res["aligned_grid"], mean=res["aligned_mean"],
                            q25=res["aligned_q25"], q75=res["aligned_q75"], sigma_kept=res["sigma_kept"])
        log.info("%s ESF: sigma = %.2f vox = %.1f um (n=%d)", ctx.sample.id, res["sigma_median"],
                 out["sigma_median_um"], res["n_kept"])
    else:
        log.warning("%s ESF: too few full-contrast edges (%d)", ctx.sample.id, res["n_kept"])
    save_json(ctx.data("esf"), out)


# ---------------------------------------------------------------------------
# 10. Replicate sub-volumes
# ---------------------------------------------------------------------------
def step_replicates(ctx: SampleContext):
    cfg = ctx.cfg
    fcfg = cfg["fractal"]
    edge = ctx.vox(float(cfg.scaled("volumes", "replicate_cube_mm")))
    stack = ctx.stack()
    boxes = tile_cubes(stack.shape, edge)
    v_um, v_mm = ctx.voxel_um, ctx.voxel_mm
    sizes = fractal.matched_box_sizes(v_um, float(fcfg["replicate_fit_min_mm"]), float(fcfg["replicate_fit_max_mm"]),
                                      8, max(2, edge // 4), int(fcfg["min_box_voxels"]))
    z_levels = sorted({b.z0 for b in boxes})
    rows = []
    T = ctx.T_primary
    with timed(log, f"replicates {ctx.sample.id}: {len(boxes)} cubes of {edge} vox"):
        for i, b in enumerate(boxes):
            g = stack.grey(b.z0, b.z1, b)
            m = g < T
            src = ArraySource(m)
            counts = surface.transition_counts(src, slab=edge)
            phi = counts["n_pore"] / m.size
            row = {"sample": ctx.sample.id, "cube": i, "layer": z_levels.index(b.z0),
                   "zc_mm": (b.z0 + b.z1) / 2 * v_mm, "yc_mm": (b.y0 + b.y1) / 2 * v_mm,
                   "xc_mm": (b.x0 + b.x1) / 2 * v_mm, "edge_mm": edge * v_mm,
                   "porosity": phi,
                   "porosity_mixing": float((ctx.I_s - g.mean()) / (ctx.I_s - ctx.I_p)),
                   "porosity_legacyT": float((g < ctx.T_legacy).mean()),
                   "sv_crofton13": surface.sv_crofton(counts, v_mm),
                   "sv_mc_grey": surface.mc_area(ArraySource(g), T - 0.5, v_mm, slab=edge) / (m.size * v_mm**3)}
            if phi > 0:
                fr = _fractal_block(src, v_um, fcfg, fcfg["replicate_fit_min_mm"], fcfg["replicate_fit_max_mm"],
                                    2, cfg.seed + i, sizes)
                row.update({"D": fr["fit"].get("D"), "D_r2": fr["fit"].get("r2")})
                for k_s, (smm, cnt) in enumerate(zip(fr["size_mm"], fr["count"])):
                    row[f"box_mm_{k_s}"] = float(smm)
                    row[f"box_n_{k_s}"] = float(cnt)
                cs = conn.summarize(conn.label_clusters(src, 6, slab=edge), v_mm)
                row.update({"largest_cluster_fraction": cs["largest_cluster_fraction"],
                            "percolates_z": cs["largest_spans_z"],
                            "percolating_porosity_any": cs["percolating_porosity_any"]})
            rows.append(row)
    pd.DataFrame(rows).to_csv(ctx.data("replicates", "csv"), index=False)


# ---------------------------------------------------------------------------
# 11. Resolution experiment
# ---------------------------------------------------------------------------
def step_resolution(ctx: SampleContext):
    cfg = ctx.cfg
    fcfg = cfg["fractal"]
    cube_path = cfg.cache_dir / f"{ctx.sample.id}_cube_grey.npy"
    grey = np.load(cube_path, mmap_mode="r")
    src = ArraySource(grey)
    T = ctx.T_primary
    rows = []
    targets = sorted({ctx.voxel_um, *[float(t) for t in cfg["resolution"]["target_voxel_um"] if t >= ctx.voxel_um - 1e-6]})
    # a truly fixed range for every resolution: from 2 voxels of the coarsest
    # target (of either sample) to fit_max
    coarsest = max(float(t) for t in cfg["resolution"]["target_voxel_um"]) / 1000
    fixed_min = max(float(fcfg["fit_min_mm"]), 2 * coarsest)
    for target in targets:
        factor = target / ctx.voxel_um
        with timed(log, f"resolution {ctx.sample.id}: {target:.2f} um (x{factor:.3f})"):
            if abs(factor - 1) < 1e-6:
                g = grey
                level = T - 0.5
            else:
                g = downsample(src, factor)
                level = T
            vmm = target / 1000
            msrc = ThresholdView(g, T)
            counts = surface.transition_counts(msrc)
            n = int(np.prod(g.shape))
            phi = counts["n_pore"] / n
            mean_g = float(np.mean(g, dtype=np.float64))
            s_min = max(float(fcfg["fit_min_mm"]), 2 * vmm)
            max_vox = max(2, min(g.shape) // 4)
            sz = fractal.matched_box_sizes(target, s_min, float(fcfg["fit_max_mm"]), 8, max_vox)
            sz_fixed = fractal.matched_box_sizes(target, fixed_min, float(fcfg["fit_max_mm"]), 6, max_vox)
            sizes = np.unique(np.concatenate([sz, sz_fixed])).astype(int)
            fr = _fractal_block(msrc, target, fcfg, sz[0] * vmm, sz[-1] * vmm, 2, cfg.seed, sizes)
            sel_m = np.isin(fr["size_vox"], sz)
            fr["fit"] = fractal.fit_dimension(fr["size_mm"][sel_m], fr["count"][sel_m])
            sel_f = np.isin(fr["size_vox"], sz_fixed)
            fit_fixed = fractal.fit_dimension(fr["size_mm"][sel_f], fr["count"][sel_f])
            cs = conn.summarize(conn.label_clusters(msrc, 6, slab=48), vmm)
            rows.append({"sample": ctx.sample.id, "voxel_um": target, "factor": factor, "shape": list(g.shape),
                         "porosity": phi, "porosity_mixing": (ctx.I_s - mean_g) / (ctx.I_s - ctx.I_p),
                         "sv_crofton13": surface.sv_crofton(counts, vmm),
                         "sv_mc_grey": surface.mc_area(ArraySource(g), level, vmm) / (n * vmm**3),
                         **_fit_fields(fr["fit"]), **_fit_fields(fit_fixed, "D_fixed"),
                         "largest_cluster_fraction": cs.get("largest_cluster_fraction"),
                         "percolating_porosity_any": cs.get("percolating_porosity_any"),
                         "largest_spans_z": cs.get("largest_spans_z"),
                         "volume_mm3": n * vmm**3,
                         "box_size_mm": fr["size_mm"].tolist(), "box_count": fr["count"].tolist()})
            if g is not grey:
                del g
    # apparent surface dimension from the scaling of S_V with voxel size:
    # S_V(dx) ~ dx^(2 - D)  =>  D = 2 - slope of log S_V vs log dx
    from scipy import stats as sps

    df = pd.DataFrame(rows)
    scaling = {}
    for col in ("sv_crofton13", "sv_mc_grey"):
        ok = df[col] > 0
        if ok.sum() >= 3:
            r = sps.linregress(np.log(df.loc[ok, "voxel_um"]), np.log(df.loc[ok, col]))
            t = sps.t.ppf(0.975, ok.sum() - 2)
            scaling[col] = {"slope": r.slope, "se": r.stderr, "D_res": 2 - r.slope,
                            "D_res_ci": [2 - r.slope - t * r.stderr, 2 - r.slope + t * r.stderr],
                            "r2": r.rvalue**2, "n": int(ok.sum())}
    save_json(ctx.data("resolution"), {"rows": rows, "scaling": scaling})
