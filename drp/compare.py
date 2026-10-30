"""Cross-sample synthesis: every number reported in the article, with its
uncertainty, collected into outputs/data/results.json.

Reads the per-sample outputs of steps 1-11; computes nothing expensive.
Contrasts between specimens are always first sample minus second sample
(config order: Limestone - Dolomite), and ratios are first over second.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import fractal, stats as st
from .config import Config
from .context import SampleContext
from .util import load_json, save_json

log = logging.getLogger("drp")

MD_PER_MM2 = 1.0 / 9.869233e-10        # 1 mD = 9.869233e-16 m2 = 9.869233e-10 mm2


def _npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def kozeny_carman_md(phi, sv_per_mm, c: float = 5.0):
    """Kozeny-Carman permeability proxy k = phi^3 / (c S_V^2), S_V per bulk volume, in mD."""
    phi = np.asarray(phi, float)
    sv = np.asarray(sv_per_mm, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sv > 0, phi**3 / (c * sv**2) * MD_PER_MM2, np.nan)


# ---------------------------------------------------------------------------
# Porosity budget
# ---------------------------------------------------------------------------
def porosity_budget(ctx: SampleContext, rev: dict, conf: float) -> dict:
    """Image porosity against the laboratory value.

    The spatial SE of each whole-stack image porosity is the variance-volume
    extrapolation to the stack volume; the larger of the straight-line and the
    finite-domain extrapolations is used (see rev.py). The mass-balance
    porosity has its own variance-volume curve (block sums of grey levels).
    """
    s = ctx.sample
    seg = ctx.seg
    thr = seg["thresholds_by_key"]
    mix = seg["mixing"]
    zc = sps.norm.ppf(0.5 + conf / 2)
    lab, lab_sd = s.lab_porosity, s.lab_porosity_sd
    vol_total = float(np.prod(ctx.shape)) * ctx.voxel_mm**3

    def se_of(tag):
        f = rev.get(tag, {}).get("fit", {})
        return float(f.get("porosity_se_total_conservative", f.get("porosity_se_total_powerlaw", np.nan)))

    def deviation(phi, se_spatial):
        se = float(np.hypot(se_spatial, lab_sd))
        d = phi - lab
        return {"estimate": phi, "se_spatial": se_spatial, "ci_spatial": [phi - zc * se_spatial, phi + zc * se_spatial],
                "deviation": d, "relative_deviation": d / lab, "se_combined": se,
                "deviation_ci": [d - zc * se, d + zc * se], "z": d / se if se > 0 else np.nan,
                "p_two_sided": float(2 * sps.norm.sf(abs(d / se))) if se > 0 else np.nan}

    out = {"lab": lab, "lab_sd_assumed": lab_sd, "volume_mm3": vol_total,
           "se_method": "variance-volume extrapolation to the stack volume (larger of straight-line and "
                        "finite-domain fits)"}
    out["legacy_threshold"] = deviation(thr["legacy"]["porosity"], se_of("legacy"))
    out["primary_threshold"] = deviation(thr["primary"]["porosity"], se_of("primary"))
    se_mix = se_of("mixing")
    m = deviation(mix["unclipped"], se_mix)
    m["endmember_range"] = mix["unclipped_range"]
    m["deviation_range_endmembers"] = [v - lab for v in mix["unclipped_range"]]
    out["mixing"] = m
    out["mixing_clipped"] = {"estimate": mix["clipped"], "endmember_range": mix["clipped_range"]}
    # sub-resolution (background) porosity needed to close the budget
    pm = mix["unclipped"]
    bg = (lab - pm) / (1 - pm)
    d_lab = 1 / (1 - pm)
    d_mix = -(1 - lab) / (1 - pm) ** 2
    se_bg = float(np.sqrt((d_lab * lab_sd) ** 2 + (d_mix * se_mix) ** 2)) if np.isfinite(se_mix) else np.nan
    out["implied_background"] = {"estimate": bg, "se": se_bg, "ci": [bg - zc * se_bg, bg + zc * se_bg],
                                 "se_components": "assumed lab sd and spatial SE of the mass balance (delta method)",
                                 "endmember_range": mix["implied_background_range"]}
    phi_bg = mix["background_scenario"]
    out["background_scenario"] = {
        "phi_bg": phi_bg,
        "legacy_binary_plus_bg": mix["total_with_background_legacy_binary"],
        "legacy_binary_plus_bg_rel_error": (mix["total_with_background_legacy_binary"] - lab) / lab,
        "mixing_plus_bg": mix["total_with_background_mixing"],
        "mixing_plus_bg_rel_error": (mix["total_with_background_mixing"] - lab) / lab,
    }
    return out


# ---------------------------------------------------------------------------
# Profiles: matched-thickness, matched-extent heterogeneity comparison
# ---------------------------------------------------------------------------
def profile_comparison(ctxs, slab_mm: float, n_boot: int, conf: float, seed: int) -> dict:
    """Heterogeneity of the two specimens at equal slab thickness and extent.

    z profiles are cut to the common height; y and x profiles are averaged
    over a central z window of that same height in both specimens, so every
    slab average covers the same physical extent. Slab-porosity series are
    strongly autocorrelated (few independent slabs). Intervals use effective
    degrees of freedom of the variance, nu = n (1 - rho^2)/(1 + rho^2) - 1
    with the bias-corrected lag-1 correlation (chi-square for each CV, F for
    the ratio), but only when nu >= 5 and n_eff >= 5; otherwise the CV is
    reported without an interval ("not estimable"). Even when given, the
    intervals are anticonservative if the two series differ in persistence.
    """
    prof = {c.sample.id: _npz(c.data("profiles", "npz")) for c in ctxs}
    a_id, b_id = ctxs[0].sample.id, ctxs[1].sample.id
    alpha = 1 - conf
    out = {}
    for axis in "zyx":
        key = axis if axis == "z" else f"{axis}_zwin"
        L = min(len(prof[c.sample.id][f"{key}_binary"]) * c.voxel_mm for c in ctxs)
        out[axis] = {"common_length_mm": L, "matched_z_extent": axis != "z"}
        for kind in ("mixing", "binary", "legacy"):
            res = {}
            for c in ctxs:
                y = prof[c.sample.id][f"{key}_{kind}"]
                n_keep = int(round(L / c.voxel_mm))
                start = (len(y) - n_keep) // 2
                y = y[start : start + n_keep]
                ns = max(1, int(round(slab_mm / c.voxel_mm)))
                ya = y[: (len(y) // ns) * ns].reshape(-1, ns).mean(axis=1)
                ne = st.effective_n(ya)
                rc = ne.get("rho1_corrected", 0.0)
                rc = 0.0 if not np.isfinite(rc) or rc < 0 else rc
                nu_raw = len(ya) * (1 - rc**2) / (1 + rc**2) - 1
                est = bool(nu_raw >= 5 and ne["n_eff"] >= 5 and ne["neff_estimable"])
                nu = max(1.0, nu_raw)
                cv = st.cv(ya)
                lo = cv * np.sqrt(nu / sps.chi2.ppf(1 - alpha / 2, nu)) if est else np.nan
                hi = cv * np.sqrt(nu / sps.chi2.ppf(alpha / 2, nu)) if est else np.nan
                res[c.sample.id] = {"n_slabs": len(ya), "slab_mm": ns * c.voxel_mm, "mean": float(ya.mean()),
                                    "cv": cv, "cv_ci": [float(lo), float(hi)], "n_eff": ne["n_eff"], "dof": nu,
                                    "dof_raw": float(nu_raw), "rho1_corrected": rc, "estimable": est}
            ra, rb = res[a_id], res[b_id]
            ratio = ra["cv"] / rb["cv"]
            both = ra["estimable"] and rb["estimable"]
            fa, fb = ra["dof"], rb["dof"]
            lo = ratio / np.sqrt(sps.f.ppf(1 - alpha / 2, fa, fb)) if both else np.nan
            hi = ratio / np.sqrt(sps.f.ppf(alpha / 2, fa, fb)) if both else np.nan
            fstat = ratio**2
            p = min(1.0, 2 * min(sps.f.cdf(fstat, fa, fb), sps.f.sf(fstat, fa, fb))) if both else np.nan
            res["cv_ratio"] = {"estimate": float(ratio), "ci": [float(lo), float(hi)], "p_equal": float(p),
                               "numerator": a_id, "estimable": both, "approximate": True,
                               "method": "F interval with effective degrees of freedom of the variance "
                                         "(only when nu >= 5 and n_eff >= 5 in both samples)"}
            out[axis][kind] = res
    return out


# ---------------------------------------------------------------------------
# Box counting: saturation and plateau diagnostics
# ---------------------------------------------------------------------------
def fractal_diagnostics(ctx: SampleContext, cfg, esf: dict | None) -> dict:
    cube = load_json(ctx.data("cube"))
    arr = _npz(ctx.data("cube", "npz"))
    edge = cube["edge_mm"]
    vol = edge**3
    size, count = arr["box_size_mm"], arr["box_count"]
    occ = fractal.occupancy(size, count, vol)
    fcfg = cfg["fractal"]
    max_occ = float(fcfg.get("max_occupancy", 0.5))
    sig_mm = (esf or {}).get("sigma_median_um", 0.0) / 1000.0
    s_floor = max(float(fcfg["fit_min_mm"]), sig_mm)
    pl = fractal.plateau(size, count, vol, s_floor, max_occ, float(fcfg.get("plateau_tol", 0.05)))
    pl90 = fractal.plateau(size, count, vol, s_floor, 0.9, float(fcfg.get("plateau_tol", 0.05)))
    out = {
        "common_range_fit": cube["fractal"]["fit"],
        "unsaturated_fit": fractal.unsaturated_fit(size, count, vol, float(fcfg["fit_min_mm"]), max_occ),
        "crossover_mm_occ50": fractal.crossover_size(size, occ, 0.5),
        "crossover_mm_occ90": fractal.crossover_size(size, occ, 0.9),
        "local_slope_min_all": float(np.nanmin(arr["box_local_slope"])),
        "local_slope_max_all": float(np.nanmax(arr["box_local_slope"])),
        "plateau": pl, "plateau_floor_mm": s_floor, "plateau_reportable": bool(pl["npts"] >= fractal.MIN_PLATEAU_PTS),
        "plateau_occupancy_le_0.9": pl90,
        "slope_trend": fractal.slope_trend(size, count, vol, 0.9),
        "min_window_decades": float(fcfg["min_window_decades"]),
        "has_scaling_regime": bool(pl["npts"] >= fractal.MIN_PLATEAU_PTS
                                   and pl["decades"] >= float(fcfg["min_window_decades"])),
        "window_D_range": [float(min(w["D"] for w in cube["fractal"]["windows"])),
                           float(max(w["D"] for w in cube["fractal"]["windows"]))],
        "occupancy": occ.tolist(), "size_mm": size.tolist(),
    }
    leg = load_json(ctx.data("legacy"))
    bc = leg["boxcount"]
    fitted = np.asarray(bc["occupancy"])[1:]
    v_leg = leg["metadata"]["voxel_um"] / 1000
    out["legacy"] = {
        "D_published_algorithm": bc["D_legacy"],
        "fit_r2": bc["fit"]["r2"],
        "sizes_mm_metadata_voxel": (np.asarray(bc["size_vox"]) * v_leg).tolist(),
        "occupancy": bc["occupancy"],
        "fraction_of_fitted_points_occupancy_gt_0.5": float(np.mean(fitted > 0.5)),
        "fraction_of_fitted_points_occupancy_gt_0.9": float(np.mean(fitted > 0.9)),
    }
    return out


# ---------------------------------------------------------------------------
# Replicate sub-volumes
# ---------------------------------------------------------------------------
REPLICATE_METRICS = ["porosity_mixing", "porosity", "porosity_legacyT", "sv_crofton13", "sv_mc_grey",
                     "s_per_vp", "mean_chord_mm", "k_kc_md", "D", "D_common_grid", "D_unsat", "crossover_mm",
                     "largest_cluster_fraction"]


def replicate_frame(ctxs, cfg) -> pd.DataFrame:
    fcfg = cfg["fractal"]
    w_lo, w_hi = (float(v) for v in fcfg["replicate_common_window_mm"])
    max_occ = float(fcfg.get("max_occupancy", 0.5))
    frames = []
    for c in ctxs:
        p = c.data("replicates", "csv")
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["s_per_vp"] = df["sv_crofton13"] / df["porosity"].where(df["porosity"] > 0)
        df["mean_chord_mm"] = 4 * df["porosity"] / df["sv_crofton13"].where(df["sv_crofton13"] > 0)
        df["k_kc_md"] = kozeny_carman_md(df["porosity"], df["sv_crofton13"])
        size_cols = sorted([k for k in df if k.startswith("box_mm_")], key=lambda k: int(k.split("_")[-1]))
        n_cols = sorted([k for k in df if k.startswith("box_n_")], key=lambda k: int(k.split("_")[-1]))
        d_un, npts, d2, d_win, xo = [], [], [], [], []
        for _, r in df.iterrows():
            if not size_cols or not np.isfinite(r[size_cols[0]]):
                d_un.append(np.nan), npts.append(0), d2.append(np.nan), d_win.append(np.nan), xo.append(np.nan)
                continue
            size = r[size_cols].to_numpy(float)
            cnt = r[n_cols].to_numpy(float)
            vol = r["edge_mm"] ** 3
            occ = fractal.occupancy(size, cnt, vol)
            un = fractal.unsaturated_fit(size, cnt, vol, float(fcfg["replicate_fit_min_mm"]), max_occ)
            d_un.append(un["D"])
            npts.append(un.get("n_points", 0))
            # sensitivity variants: two-point slope where only two unsaturated
            # sizes exist, and one fixed window for every cube of both samples
            sel = (occ <= max_occ) & (size >= float(fcfg["replicate_fit_min_mm"]) * (1 - 1e-9)) & (cnt > 0)
            if sel.sum() == 2:
                xs, ys = np.log(size[sel]), np.log(cnt[sel])
                d2.append(float(-(ys[1] - ys[0]) / (xs[1] - xs[0])))
            else:
                d2.append(un["D"])
            win = (size >= w_lo * (1 - 1e-9)) & (size <= w_hi * (1 + 1e-9)) & (cnt > 0)
            d_win.append(fractal.fit_dimension(size[win], cnt[win])["D"] if win.sum() >= 3 else np.nan)
            xo.append(fractal.crossover_size(size, occ))
        df["D_unsat"] = d_un
        df["D_unsat_npts"] = npts
        df["D_unsat_2pt_fill"] = d2
        df["D_window"] = d_win
        df["crossover_mm"] = xo
        df["sample_label"] = c.sample.label
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # D on one common interpolated grid of sizes for every cube of both samples
    size_cols = sorted([k for k in out if k.startswith("box_mm_")], key=lambda k: int(k.split("_")[-1]))
    n_cols = sorted([k for k in out if k.startswith("box_n_")], key=lambda k: int(k.split("_")[-1]))
    if size_cols:
        sz = out[size_cols].to_numpy(float)
        lo = np.nanmax([np.nanmin(sz[out["sample"] == c.sample.id]) for c in ctxs])
        hi = np.nanmin([np.nanmax(sz[out["sample"] == c.sample.id]) for c in ctxs])
        out["D_common_grid"] = [fractal.fit_on_common_sizes(r[size_cols].to_numpy(float), r[n_cols].to_numpy(float),
                                                            lo, hi, 8)["D"] if np.isfinite(r[size_cols[0]]) else np.nan
                                for _, r in out.iterrows()]
        out.attrs["common_grid_mm"] = [float(lo), float(hi)]
    return out


def replicate_statistics(df: pd.DataFrame, ctxs, n_boot: int, conf: float, seed: int, cfg) -> dict:
    a_id, b_id = ctxs[0].sample.id, ctxs[1].sample.id
    out = {"n_cubes": {c.sample.id: int((df["sample"] == c.sample.id).sum()) for c in ctxs},
           "edge_mm": {c.sample.id: float(df.loc[df["sample"] == c.sample.id, "edge_mm"].iloc[0]) for c in ctxs},
           "n_layers": {c.sample.id: int(df.loc[df["sample"] == c.sample.id, "layer"].nunique()) for c in ctxs},
           "contrast": f"{a_id} − {b_id}",
           "ci_method": "z-layer means with AR(1) effective sizes (Kendall-corrected rho capped at 0.95); "
                        "Welch-Satterthwaite contrast; approximate",
           "common_grid_mm": df.attrs.get("common_grid_mm"), "metrics": {}}
    for m in REPLICATE_METRICS:
        if m not in df:
            continue
        a = df.loc[df["sample"] == a_id]
        b = df.loc[df["sample"] == b_id]
        ok_a, ok_b = a[m].notna(), b[m].notna()
        if ok_a.sum() < 5 or ok_b.sum() < 5:
            continue
        res = {}
        lm = st.layer_mean_contrast(a.loc[ok_a, m], b.loc[ok_b, m], a.loc[ok_a, "layer"], b.loc[ok_b, "layer"], conf)
        for sid, d, ok, sd, sm in ((a_id, a, ok_a, seed, lm["a"]), (b_id, b, ok_b, seed + 1, lm["b"])):
            desc = st.describe(d.loc[ok, m], 200, conf, sd)
            desc.update({"mean_ci_low": sm["ci_low"], "mean_ci_high": sm["ci_high"], "n_layers": sm["n_layers"],
                         "n_eff_layers": sm["n_eff_layers"], "layer_mean_range": [sm["layer_mean_min"],
                                                                                   sm["layer_mean_max"]]})
            res[sid] = desc
        cmp_ = {k: v for k, v in lm.items() if k not in ("a", "b")}
        cmp_.update({"hedges_g": st.hedges_g(a.loc[ok_a, m], b.loc[ok_b, m]),
                     "cliffs_delta": st.cliffs_delta(a.loc[ok_a, m], b.loc[ok_b, m]),
                     "n_a": int(ok_a.sum()), "n_b": int(ok_b.sum())})
        res["comparison"] = cmp_
        out["metrics"][m] = res
    # D_unsat: which cubes have no resolvable unsaturated range, and how much
    # the contrast depends on that exclusion
    if "D_unsat" in df:
        ex = {}
        for c in ctxs:
            d = df[df["sample"] == c.sample.id]
            miss = d["D_unsat"].isna() & d["porosity"].gt(0)
            ex[c.sample.id] = {"n_excluded": int(miss.sum()), "n_total": int(len(d)),
                               "porosity_excluded_mean": float(d.loc[miss, "porosity"].mean()) if miss.any() else None,
                               "porosity_kept_mean": float(d.loc[~miss, "porosity"].mean()),
                               "sv_excluded_mean": float(d.loc[miss, "sv_crofton13"].mean()) if miss.any() else None,
                               "sv_kept_mean": float(d.loc[~miss, "sv_crofton13"].mean())}
        sens = {}
        for var in ("D_unsat_2pt_fill", "D_window"):
            if var not in df:
                continue
            a = df[(df["sample"] == a_id) & df[var].notna()]
            b = df[(df["sample"] == b_id) & df[var].notna()]
            if len(a) >= 5 and len(b) >= 5:
                cmp_ = st.layer_mean_contrast(a[var], b[var], a["layer"], b["layer"], conf)
                sens[var] = {"mean_a": float(a[var].mean()), "mean_b": float(b[var].mean()),
                             "n_a": int(len(a)), "n_b": int(len(b)), "difference": cmp_["difference"],
                             "diff_ci": [cmp_["diff_ci_low"], cmp_["diff_ci_high"]]}
        for e in ex.values():
            pe, pk = e["porosity_excluded_mean"], e["porosity_kept_mean"]
            e["excluded_are"] = None if pe is None else ("more porous" if pe > pk else "less porous")
        out["D_unsat_exclusion"] = {"rule": "fewer than 3 box sizes >= replicate_fit_min_mm with occupancy <= "
                                            "max_occupancy", "min_points": 3,
                                    "fit_min_mm": float(cfg["fractal"]["replicate_fit_min_mm"]),
                                    "max_occupancy": float(cfg["fractal"].get("max_occupancy", 0.5)),
                                    "per_sample": ex, "sensitivity": sens,
                                    "window_mm": cfg["fractal"]["replicate_common_window_mm"]}
    # Do the samples differ once porosity is accounted for?
    out["ancova"] = {}
    for resp, cov in (("sv_crofton13", "porosity"), ("D_unsat", "porosity"), ("D_common_grid", "porosity"),
                      ("crossover_mm", "porosity"), ("largest_cluster_fraction", "porosity")):
        if resp in df and df[resp].notna().sum() > 20:
            try:
                out["ancova"][resp] = st.ancova_descriptive(df, resp, cov, (a_id, b_id))
            except Exception as exc:  # noqa: BLE001 - report and continue
                out["ancova"][resp] = {"error": str(exc)}
    out["spearman_vs_porosity"] = {}
    for c in ctxs:
        d = df[df["sample"] == c.sample.id]
        out["spearman_vs_porosity"][c.sample.id] = {
            m: {**st.spearman_ci(d["porosity"], d[m], 1000, conf, seed, groups=d["layer"]),
                "note": "CI from layer-block resampling: a lower bound on the uncertainty"}
            for m in ("sv_crofton13", "D", "D_unsat", "porosity_mixing") if m in d and d[m].notna().sum() > 5}
    return out


# ---------------------------------------------------------------------------
# Resolution experiment
# ---------------------------------------------------------------------------
def resolution_comparison(ctxs, cfg) -> dict:
    res = {c.sample.id: load_json(c.data("resolution")) for c in ctxs}
    out = {"per_sample": {}, "matched": []}
    for c in ctxs:
        rows = res[c.sample.id]["rows"]
        for r in rows:
            vol = r.get("volume_mm3") or (np.prod(r["shape"]) * (r["voxel_um"] / 1000) ** 3)
            if "box_count" in r:
                size, cnt = np.asarray(r["box_size_mm"]), np.asarray(r["box_count"])
                occ = fractal.occupancy(size, cnt, vol)
                r["crossover_mm"] = fractal.crossover_size(size, occ)
                # lower bound: 2 voxels (the smallest box used), so native and
                # coarsened grids are fitted from their own resolution limit
                un = fractal.unsaturated_fit(size, cnt, vol, 2 * r["voxel_um"] / 1000 * 0.999)
                r["D_unsat"] = un["D"]
                r["D_unsat_se"] = un.get("se")
                r["D_unsat_fit_min_mm"] = un.get("s_min")
                r["D_unsat_fit_max_mm"] = un.get("s_max")
            r["k_kc_md"] = float(kozeny_carman_md(r["porosity"], r["sv_crofton13"]))
        native = rows[0]
        out["per_sample"][c.sample.id] = {"rows": rows, "scaling": res[c.sample.id]["scaling"],
                                          "native_voxel_um": native["voxel_um"]}
    # D over one nominal fixed range, refitted on a common interpolated grid of
    # sizes shared by every resolution of both samples (realised integer
    # sizes differ by up to one voxel at each end)
    allrows = [r for c in ctxs for r in out["per_sample"][c.sample.id]["rows"] if "box_count" in r]
    if allrows:
        lo = max(r.get("D_fixed_fit_min_mm") or np.nan for r in allrows)
        hi = min(r.get("D_fixed_fit_max_mm") or np.nan for r in allrows)
        for r in allrows:
            f = fractal.fit_on_common_sizes(r["box_size_mm"], r["box_count"], lo, hi, 6)
            r["D_fixed_common"] = f["D"]
            r["D_fixed_common_se"] = f.get("se")
        out["fixed_common_range_mm"] = [float(lo), float(hi)]
    a, b = ctxs[0].sample.id, ctxs[1].sample.id
    keys = ("porosity", "porosity_mixing", "sv_crofton13", "D", "D_fixed", "D_fixed_common", "D_unsat",
            "crossover_mm", "largest_cluster_fraction", "k_kc_md")
    for ra in out["per_sample"][a]["rows"]:
        for rb in out["per_sample"][b]["rows"]:
            if abs(ra["voxel_um"] - rb["voxel_um"]) < 0.05:
                out["matched"].append({"voxel_um": ra["voxel_um"], **{f"{k}_{a}": ra.get(k) for k in keys},
                                       **{f"{k}_{b}": rb.get(k) for k in keys}})
    # effect of bringing the finer scan to the coarser scan's voxel size
    fine = min(ctxs, key=lambda c: c.voxel_um).sample.id
    rf = out["per_sample"][fine]["rows"]
    if len(rf) > 1:
        coarse_um = max(c.voxel_um for c in ctxs)
        n0 = rf[0]
        n1 = min(rf[1:], key=lambda r: abs(r["voxel_um"] - coarse_um))

        def d(k):
            return (n1.get(k) - n0.get(k)) if n0.get(k) is not None and n1.get(k) is not None else None

        out["finer_scan_coarsened"] = {
            "sample": fine, "from_um": n0["voxel_um"], "to_um": n1["voxel_um"],
            **{f"rel_change_{k}": (n1[k] - n0[k]) / n0[k] for k in ("porosity", "sv_crofton13", "sv_mc_grey",
                                                                    "crossover_mm", "k_kc_md") if n0.get(k)},
            "abs_change_D_unsat": d("D_unsat"), "abs_change_D_matched": d("D"),
            "abs_change_D_fixed": d("D_fixed_common"),
            "D_se": [n0.get("D_se"), n1.get("D_se")], "D_unsat_se": [n0.get("D_unsat_se"), n1.get("D_unsat_se")],
        }
    return out


# ---------------------------------------------------------------------------
def step_compare(cfg: Config):
    ctxs = [SampleContext(cfg, s) for s in cfg.samples]
    conf = float(cfg["stats"]["confidence"])
    n_boot = int(cfg.scaled("stats", "n_boot"))
    seed = cfg.seed
    results = {"samples": {}, "config_primary_f": float(cfg["segmentation"]["primary_f"]),
               "sample_order": [c.sample.id for c in ctxs],
               "contrast_convention": f"{ctxs[0].sample.label} − {ctxs[1].sample.label}"}

    rep = replicate_frame(ctxs, cfg)
    if len(rep):
        rep.to_csv(cfg.data_dir / "replicates_all.csv", index=False)

    for c in ctxs:
        sid = c.sample.id
        r = {"label": c.sample.label, "long_label": c.sample.long_label,
             "acquisition": load_json(c.data("acquisition")), "qc": load_json(c.data("qc")),
             "segmentation": {k: v for k, v in c.seg.items() if k not in ("k_curve", "f_curve", "f_curve_fine",
                                                                         "uncorrected")},
             "segmentation_uncorrected_thresholds": c.seg["uncorrected"]["thresholds_by_key"],
             "segmentation_uncorrected_mixing": c.seg["uncorrected"]["mixing"]["unclipped"],
             "legacy": load_json(c.data("legacy")), "profiles": load_json(c.data("profiles")),
             "rev": load_json(c.data("rev"))}
        r["porosity_budget"] = porosity_budget(c, r["rev"], conf)
        for step in ("cube", "psd", "esf", "sensitivity"):
            p = c.data(step)
            r[step] = load_json(p) if p.exists() else None
        if r["cube"] is not None:
            r["fractal_diagnostics"] = fractal_diagnostics(c, cfg, r["esf"])
            cub = r["cube"]["surface"]
            r["cube"]["mean_chord_mm"] = 4 * cub["porosity"] / cub["sv_crofton13"]
            r["cube"]["k_kc_md"] = float(kozeny_carman_md(cub["porosity"], cub["sv_crofton13"]))
            for w in r["cube"]["sweep"]:
                w["k_kc_md"] = float(kozeny_carman_md(w["porosity"], w["sv_crofton13"]))
                w["s_per_vp"] = w["sv_crofton13"] / w["porosity"] if w["porosity"] > 0 else None
            perc = [w for w in r["cube"]["sweep"] if w["criterion"] == "f"]
            r["cube"]["percolation_by_criterion"] = {
                "f_percolating": [w["f"] for w in perc if (w.get("percolating_porosity_any") or 0) > 0],
                "legacy_percolates": any((w.get("percolating_porosity_any") or 0) > 0
                                         for w in r["cube"]["sweep"] if w["criterion"] == "legacy")}
        if r["esf"] and r["esf"].get("sigma_median_vox"):
            r["esf"]["sigma_over_voxel"] = r["esf"]["sigma_median_vox"]
        results["samples"][sid] = r

    # central-cube D of both samples on one interpolated grid of sizes inside
    # the intersection of their realised fitting ranges
    if all(results["samples"][c.sample.id].get("fractal_diagnostics") for c in ctxs):
        fits = {c.sample.id: results["samples"][c.sample.id]["fractal_diagnostics"]["common_range_fit"] for c in ctxs}
        lo, hi = max(f["s_min"] for f in fits.values()), min(f["s_max"] for f in fits.values())
        for c in ctxs:
            arr = _npz(c.data("cube", "npz"))
            fd = results["samples"][c.sample.id]["fractal_diagnostics"]
            fd["common_grid_fit"] = fractal.fit_on_common_sizes(arr["box_size_mm"], arr["box_count"], lo, hi, 10)
            fd["common_grid_mm"] = [float(lo), float(hi)]
    results["profiles_comparison"] = profile_comparison(ctxs, float(cfg["profiles"]["comparison_slab_mm"]),
                                                        n_boot, conf, seed)
    if len(rep):
        results["replicates"] = replicate_statistics(rep, ctxs, n_boot, conf, seed, cfg)
    if all(c.data("resolution").exists() for c in ctxs):
        results["resolution"] = resolution_comparison(ctxs, cfg)

    # ordering robustness of the sample contrast across segmentation criteria;
    # only matched criteria (same f in both samples) enter the robustness
    # verdicts; the mean - k sd rule is reported separately (it is a different
    # f in each sample)
    if all(results["samples"][c.sample.id]["cube"] for c in ctxs):
        a, b = ctxs[0].sample.id, ctxs[1].sample.id
        sw = {sid: {(r["criterion"], r["f"] if r["criterion"] == "f" else None): r
                    for r in results["samples"][sid]["cube"]["sweep"]} for sid in (a, b)}
        order = []
        for key in sw[a]:
            ra, rb = sw[a][key], sw[b].get(key)
            if rb is None:
                continue
            order.append({"criterion": key[0], "f": key[1], "matched": key[0] == "f",
                          **{f"{m}_ratio_{a}_over_{b}": (ra[m] / rb[m] if ra.get(m) and rb.get(m) else None)
                             for m in ("porosity", "sv_crofton13", "s_per_vp", "D", "largest_cluster_fraction",
                                       "k_kc_md")}})
        results["sweep_ratios"] = order
        robust = {}
        for m in ("sv_crofton13", "s_per_vp", "k_kc_md", "D", "porosity"):
            vals = [r[f"{m}_ratio_{a}_over_{b}"] for r in order if r["matched"] and r[f"{m}_ratio_{a}_over_{b}"]]
            leg = [r[f"{m}_ratio_{a}_over_{b}"] for r in order if not r["matched"]]
            if vals:
                robust[m] = {"min": float(min(vals)), "max": float(max(vals)),
                             "robust_order": bool(all(v > 1 for v in vals) or all(v < 1 for v in vals)),
                             "legacy_unmatched": leg[0] if leg else None, "n_criteria": len(vals)}
        results["sweep_robustness"] = robust
    save_json(cfg.data_dir / "results.json", results)
    log.info("results.json written")
