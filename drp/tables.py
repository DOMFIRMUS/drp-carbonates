"""Manuscript tables, built from outputs/data/results.json.

Every table is written as CSV (machine-readable, full precision), as a sheet of
outputs/tables/tables.xlsx, as LaTeX (pdfLaTeX-safe: Unicode symbols are
converted to LaTeX math) and into one Markdown file for reading. Displayed
values are rounded; the CSV keeps full precision. Bootstrap p-values at the
resolution floor 2/(B + 1) are displayed as "≤ floor".
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from .config import Config
from .util import load_json

log = logging.getLogger("drp")


def _f(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    return f"{x:.{nd}f}"


def _ci(lo, hi, nd=4):
    return f"[{_f(lo, nd)}, {_f(hi, nd)}]"


def _pct(x, nd=1):
    return "–" if x is None else f"{100 * x:+.{nd}f} %"


def _p(p, floor=None):
    """p-value for display; values at the bootstrap floor are shown as ≤ floor."""
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return "–"
    if floor is not None and p <= floor * (1 + 1e-9):
        return f"≤ {floor:.4f}"
    if p < 1e-4:
        return "< 0.0001"
    return f"{p:.4f}"


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float, np.floating)) and bool(np.isfinite(x))


def _labels(R):
    ids = list(R["samples"])
    return ids, {s: R["samples"][s]["label"] for s in ids}


# ---------------------------------------------------------------------------
def t_acquisition(R, cfg):
    rows = []
    for sid, r in R["samples"].items():
        a = r["acquisition"]
        nz, ny, nx = a["shape"]
        ez, ey, ex = a["extent_mm_zyx"]
        leg = r["legacy"]
        q = r["qc"]
        rows.append({
            "Sample": r["long_label"],
            "Voxel size, metadata (µm)": a["voxel_size_um_metadata"],
            "Voxel size, previous workflow (µm)": cfg["legacy"]["assumed_voxel_um"],
            "Original reconstruction (x × y × z)": " × ".join(str(v) for v in a["original_dims_xyz"]),
            "Cropped stack (x × y × z, voxels)": f"{nx} × {ny} × {nz}",
            "Cropped stack (x × y × z, mm)": f"{ex:.2f} × {ey:.2f} × {ez:.2f}",
            "Stack volume (mm³)": round(a["volume_mm3"], 1),
            "Previous subvolume (slices)": f"{leg['box']['z'][0]}–{leg['box']['z'][1] - 1}",
            "Previous subvolume, true volume (mm³)": round(leg["metadata"]["volume_mm3"], 1),
            "Solid-peak drift along z (%)": round(q["mode_range_percent"], 2),
            "ρ(solid peak, slice porosity) [95 % CI]": f"{q['spearman_mode_vs_porosity']:+.2f} "
                                                        f"{_ci(*q.get('spearman_mode_vs_porosity_ci', [None, None]), 2)}",
            "Association (surrogate p)": f"{q.get('association_class', '–')} ({_p(q.get('spearman_p_surrogate'))})",
            "Drift correction": "yes" if q["drift_correction_applied"] else "no",
            "Laboratory porosity": a["lab_porosity"],
        })
    return pd.DataFrame(rows)


def t_legacy(R, cfg):
    rows = []
    for sid, r in R["samples"].items():
        L = r["legacy"]
        pub = L["published"]
        for name, pubv, asm, meta, unit in [
            ("Porosity", pub.get("porosity"), L["porosity"], L["porosity"], "–"),
            ("Total volume", None, L["assumed"]["volume_mm3"], L["metadata"]["volume_mm3"], "mm³"),
            ("Pore volume", None, L["assumed"]["pore_volume_mm3"], L["metadata"]["pore_volume_mm3"], "mm³"),
            ("Interface area (MC, binary)", pub.get("area_mm2"), L["assumed"]["area_mm2"], L["metadata"]["area_mm2"], "mm²"),
            ("S_V (MC, binary)", None, L["assumed"]["sv_mc_binary_per_mm"], L["metadata"]["sv_mc_binary_per_mm"], "mm⁻¹"),
            ("S_V (Crofton, 13 directions)", None, L["assumed"]["sv_crofton13_per_mm"], L["metadata"]["sv_crofton13_per_mm"], "mm⁻¹"),
            ("Box-counting D (porespy definition)", pub.get("fractal_dimension"), L["boxcount"]["D_legacy"], L["boxcount"]["D_legacy"], "–"),
        ]:
            rows.append({"Sample": r["label"], "Quantity": name, "Unit": unit, "Earlier analysis": pubv,
                         f"Reproduced ({cfg['legacy']['assumed_voxel_um']} µm assumed)": asm,
                         "With metadata voxel size": meta, "Metadata voxel (µm)": L["metadata"]["voxel_um"]})
    return pd.DataFrame(rows)


def t_segmentation(R, cfg):
    rows = []
    for sid, r in R["samples"].items():
        seg = r["segmentation"]
        lab = r["acquisition"]["lab_porosity"]
        for key, t in seg["thresholds_by_key"].items():
            rows.append({"Sample": r["label"], "Method": t["name"], "Threshold (grey)": round(t["T"]),
                         "k (mean − k·sd)": t["k_equiv"], "f (partial-volume)": t["f_equiv"],
                         "Porosity": t["porosity"], "Deviation from lab": t["porosity"] - lab,
                         "Relative deviation": t["rel_error_lab"]})
        m = seg["mixing"]
        rows.append({"Sample": r["label"], "Method": "Grey-level mass balance (no threshold)",
                     "Threshold (grey)": None, "k (mean − k·sd)": None, "f (partial-volume)": None,
                     "Porosity": m["unclipped"], "Deviation from lab": m["unclipped"] - lab,
                     "Relative deviation": (m["unclipped"] - lab) / lab})
    return pd.DataFrame(rows)


def t_endmembers(R, cfg):
    rows = []
    for sid, r in R["samples"].items():
        seg = r["segmentation"]
        m = seg["mixing"]
        rows.append({"Sample": r["label"], "Solid end-member I_s (mode)": seg["I_s"],
                     "Pore end-member I_p (q = 0.1 %)": seg["I_p"],
                     "I_p sensitivity range": f"{seg['I_p_range'][0]:.0f}–{seg['I_p_range'][1]:.0f}",
                     "Contrast I_s − I_p": seg["contrast"],
                     "Histogram excess over mirrored solid peak": seg["histogram_excess_fraction"],
                     "Mass-balance porosity": m["unclipped"],
                     "Mass-balance range (I_p)": _ci(*m["unclipped_range"]),
                     "Clipped (depositors' style)": m["clipped"],
                     "k reproducing lab porosity": seg["k_lab"], "f reproducing lab porosity": seg["f_lab"]})
    return pd.DataFrame(rows)


def t_budget(R, cfg):
    rows = []
    for sid, r in R["samples"].items():
        b = r["porosity_budget"]
        for key, name in (("legacy_threshold", "Binary, mean − 1.1 sd"), ("primary_threshold", "Binary, f = 0.5"),
                          ("mixing", "Grey-level mass balance")):
            d = b[key]
            rows.append({"Sample": r["label"], "Quantity": f"Image porosity: {name}", "Estimate": d["estimate"],
                         "SE": d["se_spatial"], "SE source": "spatial (variance–volume)",
                         "Laboratory": b["lab"], "Deviation from lab": d["deviation"],
                         "Relative deviation": d["relative_deviation"],
                         "95 % CI of deviation": _ci(*d["deviation_ci"]), "z": d["z"], "p": d["p_two_sided"],
                         "Pore end-member range": _ci(*d["endmember_range"]) if "endmember_range" in d else "–"})
        ib = b["implied_background"]
        rows.append({"Sample": r["label"], "Quantity": "Implied sub-resolution porosity of the solid phase",
                     "Estimate": ib["estimate"], "SE": ib["se"], "SE source": "lab (assumed) + spatial",
                     "Laboratory": None, "Deviation from lab": None, "Relative deviation": None,
                     "95 % CI of deviation": f"estimate CI {_ci(*ib['ci'])}", "z": None, "p": None,
                     "Pore end-member range": _ci(*ib["endmember_range"])})
    return pd.DataFrame(rows)


def t_profiles(R, cfg):
    rows = []
    names = {"z": "z (slices)", "y": "y (rows)", "x": "x (columns)"}
    for sid, r in R["samples"].items():
        for a, ax in r["profiles"]["axes"].items():
            for kind in ("mixing", "binary", "legacy"):
                d = ax[kind]
                tr = d["trend_per_10mm"]
                rows.append({"Sample": r["label"], "Axis": names[a], "Porosity": {"mixing": "mass balance",
                             "binary": "f = 0.5", "legacy": "mean − 1.1 sd"}[kind],
                             "Length (mm)": ax["length_mm"], "Mean": d["mean"], "SD": d["sd"], "CV": d["cv"],
                             "n_eff (1-mm slabs)": d["neff_n_eff"] if d.get("neff_neff_estimable", True) else "n.e.",
                             "95 % CI of mean (n_eff)": _ci(d["neff_ci_low"], d["neff_ci_high"])
                             if d.get("neff_ci_low") is not None and np.isfinite(d["neff_ci_low"]) else "n.e.",
                             "Trend per 10 mm": tr["slope"], "Trend 95 % CI (AR(1) bootstrap)": _ci(tr["ci_low"], tr["ci_high"]),
                             "Trend p (AR(1) bootstrap)": _p(tr["p_value"], 1 / (tr.get("n_surrogates", 2000) + 1)),
                             "Trend p (conservative, surrogates)": _p(tr.get("p_conservative"),
                                                                       1 / (tr.get("n_surrogates", 2000) + 1))})
    return pd.DataFrame(rows)


def t_profile_comparison(R, cfg):
    rows = []
    pc = R["profiles_comparison"]
    ids, lab = _labels(R)
    for a, d in pc.items():
        for kind in ("mixing", "binary", "legacy"):
            k = d[kind]
            rows.append({"Axis": a, "Porosity": kind, "Common length (mm)": d["common_length_mm"],
                         **{f"CV {lab[s]}": k[s]["cv"] for s in ids},
                         **{f"CV 95 % CI {lab[s]}": _ci(*k[s]["cv_ci"], 3) if k[s].get("estimable") else "n.e."
                            for s in ids},
                         **{f"slabs (n_eff) {lab[s]}": f"{k[s]['n_slabs']} ({k[s]['n_eff']:.1f})" for s in ids},
                         f"CV ratio {lab[ids[0]]}/{lab[ids[1]]}": k["cv_ratio"]["estimate"],
                         "Ratio 95 % CI (F, effective dof; approximate)": _ci(*k["cv_ratio"]["ci"], 2)
                         if k["cv_ratio"].get("estimable") else "n.e.",
                         "p (equal CV; approximate)": k["cv_ratio"]["p_equal"]
                         if k["cv_ratio"].get("estimable") else "n.e."})
    return pd.DataFrame(rows)


def t_rev(R, cfg):
    rows = []
    for sid, r in R["samples"].items():
        for tag, name in (("legacy", "mean − 1.1 sd"), ("primary", "f = 0.5"), ("mixing", "mass balance")):
            if tag not in r["rev"]:
                continue
            f = r["rev"][tag]["fit"]
            nl = f.get("null", {})
            nf = f.get("null_finite_domain", {})
            rows.append({"Sample": r["label"], "Porosity": name, "Mean porosity": r["rev"][tag]["porosity_blocks"],
                         "Fit range (mm)": f"{f['fit_min_mm']:.2f}–{f['fit_max_mm']:.2f} ({f['n_points']} sizes)",
                         "α (Var ∝ V^−α)": f["alpha"], "α SE (z-block bootstrap)": f.get("alpha_se_boot"),
                         "α 95 % CI (± 1.96 SE; lower bound)": _ci(f.get("alpha_ci_low"), f.get("alpha_ci_high"), 2),
                         "α null band (stationary, finite range)": _ci(nl.get("null_q025"), nl.get("null_q975"), 2),
                         "p (α ≤ observed under null)": _p(nl.get("p_one_sided"), 1 / (nl.get("n_valid", 100) + 1)),
                         "α relative to null band": f.get("alpha_vs_null", "–"),
                         "α, finite-domain fit": f.get("alpha_finite_domain"),
                         "p, finite-domain fit": _p(nf.get("p_one_sided"), 1 / (nf.get("n_valid", 100) + 1)),
                         "CV at largest window": f["cv_at_largest_window"],
                         "SE of whole-stack porosity (straight-line / finite-domain)":
                             f"{_f(f['porosity_se_total_powerlaw'])} / {_f(f.get('porosity_se_total_finite_domain'))}",
                         "Edge for 5 % error (mm)": f["L_target_mm_powerlaw"],
                         "Extrapolated": f["L_target_is_extrapolated"]})
    return pd.DataFrame(rows)


def t_morphology(R, cfg):
    rows = []
    for sid, r in R["samples"].items():
        c = r["cube"]
        if c is None:
            continue
        s = c["surface"]
        fd = r["fractal_diagnostics"]
        pl = fd["plateau"]
        cn6, cn26 = c["connectivity"]["6"], c["connectivity"]["26"]
        cor = c["correlation"]
        psd = (r.get("psd") or {}).get("curves", {}).get("primary", {})
        esf = r.get("esf") or {}
        cf = fd["common_range_fit"]
        un = fd["unsaturated_fit"]
        rows += [{"Sample": r["label"], "Group": g, "Quantity": q, "Value": v} for g, q, v in [
            ("Volume", "Cube edge (mm)", c["edge_mm"]),
            ("Volume", "Porosity (f = 0.5)", s["porosity"]),
            ("Specific surface (mm⁻¹)", "Crofton, 13 directions", s["sv_crofton13"]),
            ("Specific surface (mm⁻¹)", "Crofton, 3 axes", s["sv_crofton3"]),
            ("Specific surface (mm⁻¹)", "Marching cubes, grey iso-surface", s.get("sv_mc_grey")),
            ("Specific surface (mm⁻¹)", "Marching cubes, binary", s.get("sv_mc_binary")),
            ("Specific surface (mm⁻¹)", "Voxel faces (upper bound)", s["sv_faces"]),
            ("Specific surface (mm⁻¹)", "S/V_p, Crofton (per pore volume)", s["s_per_vp_crofton13"]),
            ("Specific surface (mm⁻¹)", "Mean pore chord 4φ/S_V (mm)", c["mean_chord_mm"]),
            ("Specific surface (mm⁻¹)", "Kozeny–Carman proxy φ³/(5 S_V²) (mD)", c.get("k_kc_md")),
            ("Box counting", "D, common range", cf["D"]),
            ("Box counting", "D fit interval (OLS, descriptive)", _ci(cf["ci_low"], cf["ci_high"], 3)),
            ("Box counting", "R²", cf["r2"]),
            ("Box counting", "Fit range (mm)", f"{cf['s_min']:.3f}–{cf['s_max']:.3f} ({cf['n_points']} sizes)"),
            ("Box counting", "D, common interpolated grid (both samples)", (fd.get("common_grid_fit") or {}).get("D")),
            ("Box counting", "Common grid (mm)", "–" if not fd.get("common_grid_mm") else
             f"{fd['common_grid_mm'][0]:.3f}–{fd['common_grid_mm'][1]:.3f}"),
            ("Box counting", "D, unsaturated range (occupancy ≤ 0.5)", un.get("D")),
            ("Box counting", "Unsaturated range (mm)", f"{un.get('s_min', np.nan):.3f}–{un.get('s_max', np.nan):.3f}"),
            ("Box counting", "Crossover, occupancy 50 % (mm)", fd["crossover_mm_occ50"]),
            ("Box counting", "Local slope, min–max (unsaturated)", f"{pl['slope_min']:.2f}–{pl['slope_max']:.2f}"),
            ("Box counting", "Longest constant-slope run (mm; decades; slope)",
             f"{pl['s_min_mm']:.3f}–{pl['s_max_mm']:.3f}; {pl['decades']:.2f}; {pl['mean_slope']:.2f}"
             if fd.get("plateau_reportable", pl["npts"] >= 3) else "none (fewer than 3 sizes)"),
            ("Box counting", "Spearman ρ(local slope, log s), occupancy ≤ 0.9", (fd.get("slope_trend") or {}).get("rho")),
            ("Box counting", "D over all windows ≥ 0.8 decade", f"{fd['window_D_range'][0]:.2f}–{fd['window_D_range'][1]:.2f}"),
            ("Two-point correlation", "1/e length x, y, z (mm)", ", ".join(f"{cor[a]['l_1e']:.3f}" for a in "xyz")),
            ("Connectivity (6)", "Clusters per mm³", cn6["clusters_per_mm3"]),
            ("Connectivity (6)", "Largest cluster / pore volume", cn6["largest_cluster_fraction"]),
            ("Connectivity (6)", "Percolating porosity (any axis)", cn6["percolating_porosity_any"]),
            ("Connectivity (6)", "Pore volume in clusters ≤ 8 voxels", cn6["pore_fraction_in_clusters_le_8vox"]),
            ("Connectivity (26)", "Largest cluster / pore volume", cn26["largest_cluster_fraction"]),
            ("Connectivity (26)", "Percolating porosity (any axis)", cn26["percolating_porosity_any"]),
            ("Pore size", "Median pore radius (µm)", psd.get("median_radius_um")),
            ("Pore size", "Pore volume with r < 2 voxels", psd.get("fraction_radius_lt_2vox")),
            ("Effective resolution", "Edge-spread σ (µm)", esf.get("sigma_median_um")),
            ("Effective resolution", "Edge-spread σ (voxels)", esf.get("sigma_median_vox")),
            ("Effective resolution", "PSF FWHM (µm)", esf.get("fwhm_um")),
            ("Effective resolution", "Edges used", esf.get("n_kept")),
        ]]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    wide = df.pivot_table(index=["Group", "Quantity"], columns="Sample", values="Value", aggfunc="first", sort=False)
    return wide.reset_index()


def t_sweep(R, cfg):
    rows = []
    for sid, r in R["samples"].items():
        if not r["cube"]:
            continue
        for w in r["cube"]["sweep"]:
            rows.append({"Sample": r["label"], "Criterion": "mean − 1.1 sd" if w["criterion"] == "legacy" else f"f = {w['f']:g}",
                         "Porosity": w["porosity"], "S_V Crofton (mm⁻¹)": w["sv_crofton13"],
                         "S/V_p (mm⁻¹)": w.get("s_per_vp"), "Kozeny–Carman proxy (mD)": w.get("k_kc_md"),
                         "D (per-sample range, end points within one voxel)": w["D"],
                         "D 95 % fit interval (t)": _ci(w.get("D_ci_low"), w.get("D_ci_high"), 3),
                         "D fit range (mm)": f"{_f(w.get('D_fit_min_mm'), 3)}–{_f(w.get('D_fit_max_mm'), 3)}",
                         "Largest cluster fraction": w["largest_cluster_fraction"],
                         "Percolating porosity": w["percolating_porosity_any"]})
    return pd.DataFrame(rows)


METRIC_NAMES = {"porosity_mixing": "Porosity, mass balance", "porosity": "Porosity, f = 0.5",
                "porosity_legacyT": "Porosity, mean − 1.1 sd", "sv_crofton13": "S_V Crofton (mm⁻¹)",
                "sv_mc_grey": "S_V MC grey (mm⁻¹)", "s_per_vp": "S/V_p (mm⁻¹)", "mean_chord_mm": "Mean pore chord (mm)",
                "k_kc_md": "Kozeny–Carman proxy (mD)", "D": "D, matched range (0.1–1.25 mm)",
                "D_unsat": "D, unsaturated range", "crossover_mm": "Crossover length (mm)",
                "largest_cluster_fraction": "Largest-cluster fraction"}


def t_replicates(R, cfg):
    rp = R.get("replicates")
    if not rp:
        return pd.DataFrame()
    ids, lab = _labels(R)
    rows = []
    for m, d in rp["metrics"].items():
        c = d["comparison"]
        rows.append({"Metric": METRIC_NAMES.get(m, m),
                     **{f"{lab[s]} mean [95 % CI, layer means]":
                        f"{d[s]['mean']:.4g} " + (_ci(d[s]['mean_ci_low'], d[s]['mean_ci_high'], 4)
                                                  if _finite(d[s]['mean_ci_low']) else "(n.e.)") for s in ids},
                     **{f"{lab[s]} median (IQR)": f"{d[s]['median']:.4g} ({d[s]['q1']:.4g}–{d[s]['q3']:.4g})" for s in ids},
                     **{f"n cubes / layers (n_eff) {lab[s]}":
                        f"{d[s]['n']} / {d[s]['n_layers']} ({d[s]['n_eff_layers']:.1f})" for s in ids},
                     **{f"layer-mean range {lab[s]}":
                        f"{d[s]['layer_mean_range'][0]:.4g}–{d[s]['layer_mean_range'][1]:.4g}" for s in ids},
                     "Layer-mean ranges disjoint": c["ranges_disjoint"],
                     f"Difference {lab[ids[0]]} − {lab[ids[1]]} [95 % CI, approximate]":
                         f"{c['difference']:.4g} " + (_ci(c['diff_ci_low'], c['diff_ci_high'], 4)
                                                       if c.get("estimable") else "(n.e.)"),
                     f"Ratio {lab[ids[0]]}/{lab[ids[1]]}": c["ratio"],
                     "Hedges g (descriptive)": c["hedges_g"], "Cliff δ (descriptive)": c["cliffs_delta"],
                     "p (approximate, Welch on layer means)": c["p_approx"] if c.get("estimable") else "n.e."})
    return pd.DataFrame(rows)


def t_ancova(R, cfg):
    rp = R.get("replicates")
    if not rp:
        return pd.DataFrame()
    ids, lab = _labels(R)
    rows = []
    for resp, d in rp["ancova"].items():
        if "error" in d:
            continue
        name = f"{lab[d['levels'][0]]} − {lab[d['levels'][1]]}"
        base = {"Response": METRIC_NAMES.get(resp, resp), "Contrast": name, "n": ", ".join(
                    f"{lab[k]}: {v}" for k, v in d["n"].items())}
        sup = d.get("contrast_over_common_support", [])
        rows.append({**base, "Adjusted difference at pooled mean porosity": d["adjusted_difference"],
                     "Pooled mean porosity": d["at_covariate"],
                     **{f"Difference at porosity {lbl}": (f"{sup[i]['difference']:.4g} (φ = {sup[i]['covariate']:.4f})"
                                                         if len(sup) > i else "–")
                        for i, lbl in enumerate(("p05", "mid", "p95"))},
                     f"Slope {lab[d['levels'][0]]}": d["slope_a"], f"Slope {lab[d['levels'][1]]}": d["slope_b"],
                     "R²": d["r2"]})
    return pd.DataFrame(rows)


def t_resolution(R, cfg):
    res = R.get("resolution")
    if not res:
        return pd.DataFrame()
    rows = []
    for sid, d in res["per_sample"].items():
        for r in d["rows"]:
            rows.append({"Sample": R["samples"][sid]["label"], "Voxel (µm)": r["voxel_um"],
                         "Factor": r["factor"], "Porosity f = 0.5": r["porosity"],
                         "Porosity, mass balance": r["porosity_mixing"], "S_V Crofton (mm⁻¹)": r["sv_crofton13"],
                         "S_V MC grey (mm⁻¹)": r["sv_mc_grey"],
                         "D, max(0.1 mm, 2 vox)–2.5 mm": r.get("D"), "D SE": r.get("D_se"),
                         "D fit range (mm)": f"{_f(r.get('D_fit_min_mm'), 3)}–{_f(r.get('D_fit_max_mm'), 3)}",
                         "D, nominal fixed range, common grid": r.get("D_fixed_common"),
                         "Common grid (mm)": "–" if not res.get("fixed_common_range_mm") else
                         f"{res['fixed_common_range_mm'][0]:.3f}–{res['fixed_common_range_mm'][1]:.3f}",
                         "D (unsaturated)": r.get("D_unsat"), "Crossover (mm)": r.get("crossover_mm"),
                         "Kozeny–Carman proxy (mD)": r.get("k_kc_md"),
                         "Largest cluster fraction": r["largest_cluster_fraction"],
                         "Percolating porosity": r["percolating_porosity_any"]})
    return pd.DataFrame(rows)


def t_drift(R, cfg):
    rows = []
    names = {"binary": "f = 0.5", "mixing": "mass balance", "legacy": "mean − 1.1 sd"}
    for sid, r in R["samples"].items():
        sens = r["profiles"].get("drift_sensitivity_z")
        if not sens:
            continue
        if not r["qc"]["drift_correction_applied"]:
            continue                      # nothing to compare: the analysed stack is the uncorrected one
        seg, unc = r["segmentation"]["thresholds_by_key"], r["segmentation_uncorrected_thresholds"]
        rows.append({"Sample": r["label"], "Quantity": "Whole-stack porosity, f = 0.5",
                     "Drift-corrected (as analysed)": seg["primary"]["porosity"],
                     "Uncorrected": unc["primary"]["porosity"]})
        rows.append({"Sample": r["label"], "Quantity": "Whole-stack porosity, mass balance",
                     "Drift-corrected (as analysed)": r["segmentation"]["mixing"]["unclipped"],
                     "Uncorrected": r.get("segmentation_uncorrected_mixing")})
        for kind, d in sens.items():
            for q, key in (("z-profile CV", "cv"), ("z trend per 10 mm", "trend_per_10mm"), ("z trend p (AR(1))", "trend_p")):
                rows.append({"Sample": r["label"], "Quantity": f"{q}, {names[kind]}",
                             "Drift-corrected (as analysed)": d["as_analysed"][key], "Uncorrected": d["uncorrected"][key]})
    return pd.DataFrame(rows)


def t_rev_sensitivity(R, cfg):
    names = {"primary": "f = 0.5", "legacy": "mean − 1.1 sd", "mixing": "mass balance"}
    rows = []
    for sid, r in R["samples"].items():
        s = r.get("sensitivity")
        if not s:
            continue
        nk = s["null_kernels"]
        ks = sorted(nk, key=float)
        for tag in ("primary", "legacy", "mixing"):
            a, d, u = s["as_analysed"][tag], s["z_detrended"][tag], s["uncorrected_stack"][tag]
            dn, se = s["distance_from_null"][tag], s["se_ratios"][tag]
            kind = "continuous" if tag == "mixing" else "binary"
            null_k = " / ".join(f"{nk[k][kind]['mean']:.2f}" for k in ks) if tag != "legacy" else "–"
            null_kd = " / ".join(f"{nk[k][kind + '_detrended']['mean']:.2f}" for k in ks) if tag != "legacy" else "–"
            rows.append({
                "Sample": r["label"], "Porosity": names[tag],
                "α as analysed": a["alpha"], "α, z trend removed": d["alpha"], "α, no drift correction": u["alpha"],
                "CV at largest window (as analysed / z trend removed / no drift correction)":
                    f"{a['cv_largest_window']:.3f} / {d['cv_largest_window']:.3f} / {u['cv_largest_window']:.3f}",
                "Null α (1-mm kernel): mean ± sd": f"{dn['null_mean']:.2f} ± {dn['null_sd']:.2f}",
                "Observed α, null sd below null mean": dn["sd_below_null_mean"],
                "α finite-domain fit [its null band]":
                    f"{dn['alpha_finite_domain']:.2f} [{dn['finite_domain_null_band'][0]:.2f}, {dn['finite_domain_null_band'][1]:.2f}]",
                f"Null α, kernels {'/'.join(f'{float(k):g}' for k in ks)} mm (mean)": null_k,
                "Same, z trend removed": null_kd,
                "SE of whole-stack porosity / SE if α = 1 (straight-line; conservative)":
                    f"{se['ratio_powerlaw_to_alpha1']:.1f}; {se['ratio_conservative_to_alpha1']:.1f}",
                "Edge for 5 % error (mm): power law; 1/V beyond largest window":
                    f"{a['L_target_powerlaw_mm']:.0f}; {a['L_target_if_1_over_V_beyond_largest_mm']:.0f}",
            })
    if rows:
        k0 = R["samples"][next(iter(R["samples"]))]["sensitivity"]["null_kernels"]
        edges = ", ".join(f"{float(k):g} mm → {v['integral_range_edge_continuous_mm']:.1f} mm" for k, v in
                          sorted(k0.items(), key=lambda kv: float(kv[0])))
        rows.append({"Sample": "note", "Porosity": f"integral-range edge of the continuous null field: {edges}; "
                     f"{k0[next(iter(k0))]['n_fields']} fields per kernel"})
    return pd.DataFrame(rows)


def t_endmember_sensitivity(R, cfg):
    rows = []
    for sid, r in R["samples"].items():
        s = r.get("sensitivity")
        if not s:
            continue
        e, t = s["endmember"], s["solid_tiles"]
        lab = r["acquisition"].get("lab_porosity")
        rows += [{"Sample": r["label"], "Quantity": q, "Value": v} for q, v in [
            ("Solid end-member I_s (grey level)", e["I_s"]),
            ("Pore end-member I_p (grey level)", e["I_p"]),
            ("Mean grey level of the stack", e["mean_grey"]),
            ("Mass-balance porosity", e["mass_balance"]),
            ("Laboratory porosity", lab),
            ("Change of mass-balance porosity per 1 % change of I_s", e["dphi_per_1pc_Is"]),
            ("I_s that would close the budget without sub-resolution porosity (grey level)", e["I_s_closing_budget"]),
            ("Same, relative to I_s (%)", e["I_s_closing_budget_rel_percent"]),
            (f"Solid-peak level in {t['tiles']} × {t['tiles']} lateral tiles, relative to I_s: min / max (%)",
             f"{t['solid_mode_rel_min_percent']:+.1f} / {t['solid_mode_rel_max_percent']:+.1f}"),
            ("Same, range (% of I_s)", t["solid_mode_range_percent"]),
            ("Tile mass-balance porosity, global end-members: min–max",
             f"{t['mass_balance_tiles_global_endmembers'][0]:.3f}–{t['mass_balance_tiles_global_endmembers'][1]:.3f}"),
            ("Tile mass-balance porosity, tile solid end-member: min–max",
             f"{t['mass_balance_tiles_local_solid'][0]:.3f}–{t['mass_balance_tiles_local_solid'][1]:.3f}"),
            ("Mass-balance porosity with tile solid end-members (weighted mean)", t["mass_balance_weighted_local_solid"]),
            ("Slices used for the tiles", t["n_slices"]),
        ]]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.pivot_table(index="Quantity", columns="Sample", values="Value", aggfunc="first", sort=False).reset_index()


TABLES = [
    ("T01_acquisition", "Samples, acquisition metadata and analysed volumes", t_acquisition),
    ("T02_legacy_replication", "Replication of the earlier analysis and effect of the metadata voxel size", t_legacy),
    ("T03_segmentation", "Porosity by segmentation method (whole stack)", t_segmentation),
    ("T04_endmembers", "Grey-level end-members and threshold-free porosity", t_endmembers),
    ("T05_porosity_budget", "Image versus laboratory porosity with uncertainty budget", t_budget),
    ("T06_profiles", "Porosity profiles: statistics, effective sample size and trends", t_profiles),
    ("T07_profile_comparison", "Heterogeneity at matched slab thickness and extent", t_profile_comparison),
    ("T08_rev", "Variance-volume scaling of porosity (REV analysis)", t_rev),
    ("T09_morphology", "Morphology of the central cube of equal physical size", t_morphology),
    ("T10_segmentation_sweep", "Robustness of morphology to the segmentation criterion", t_sweep),
    ("T11_replicates", "Replicate sub-volumes: distributions and between-specimen contrasts", t_replicates),
    ("T12_ancova", "Specimen contrasts adjusted for porosity (ANCOVA)", t_ancova),
    ("T13_resolution", "Resolution (coarsening) experiment", t_resolution),
    ("T14_drift_sensitivity", "Sensitivity of the results to the per-slice drift correction", t_drift),
    ("T15_rev_sensitivity", "Variance-volume exponent: z trend, drift correction and correlation scale of the null model",
     t_rev_sensitivity),
    ("T16_endmember_sensitivity", "Solid end-member: lateral non-uniformity and sensitivity of the porosity budget",
     t_endmember_sensitivity),
]


def _is_p_column(name: str) -> bool:
    n = str(name).lower()
    return n == "p" or n.startswith("p (") or n.startswith("p ") or " p (" in n or n.endswith(" p")


def _display(df: pd.DataFrame) -> pd.DataFrame:
    """Rounded copy for display: floats to 4 decimals (also inside mixed-type
    columns), p-values below 1e-4 as '< 0.0001'."""
    out = df.copy()

    def fmt(v, pcol):
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return "–"
            if pcol and v < 1e-4:
                return "< 0.0001"
            if abs(v) >= 1000:
                return f"{v:,.1f}"
            return f"{v:.4f}"
        if v is None:
            return "–"
        return v

    for c in out.columns:
        pcol = _is_p_column(c)
        if out[c].dtype.kind in "fO":
            out[c] = out[c].map(lambda v, pc=pcol: fmt(v, pc))
    return out


# Unicode -> LaTeX (text mode) for pdfLaTeX; applied after escaping specials
_TEX_MAP = [
    ("mm⁻¹", r"mm$^{-1}$"), ("⁻¹", r"$^{-1}$"), ("mm³", r"mm$^{3}$"), ("mm²", r"mm$^{2}$"), ("³", r"$^{3}$"),
    ("²", r"$^{2}$"), ("µm", r"$\mu$m"), ("µ", r"$\mu$"), ("α", r"$\alpha$"), ("φ", r"$\phi$"), ("σ", r"$\sigma$"),
    ("ρ", r"$\rho$"), ("δ", r"$\delta$"), ("Δ", r"$\Delta$"), ("ν", r"$\nu$"), ("≤", r"$\le$"), ("≥", r"$\ge$"),
    ("±", r"$\pm$"), ("×", r"$\times$"), ("·", r"$\cdot$"), ("∝", r"$\propto$"), ("≈", r"$\approx$"),
    ("−", "--"), ("–", "--"), ("—", "---"),
    ("V^--α", r"V$^{-\alpha}$"),
]


def _tex(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}"),
                 ("<", r"\textless{}"), (">", r"\textgreater{}")):
        s = s.replace(a, b)
    for a, b in _TEX_MAP:
        s = s.replace(a, b)
    s = s.replace(r"V\^{}--$\alpha$", r"V$^{-\alpha}$")
    return re.sub(r"\$\$", "", s)


def to_latex(disp: pd.DataFrame, caption: str, label: str) -> str:
    cols = [_tex(c) for c in disp.columns]
    # wide tables are scaled to the text width (needs \usepackage{graphicx});
    # for the main text, select columns or transpose (the CSV keeps everything)
    lines = [r"\begin{table}", r"\centering", r"\small", rf"\caption{{{_tex(caption)}}}", rf"\label{{{label}}}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{" + "l" * len(cols) + "}", r"\hline",
             " & ".join(cols) + r" \\", r"\hline"]
    for _, row in disp.iterrows():
        lines.append(" & ".join(_tex(v) for v in row.tolist()) + r" \\")
    lines += [r"\hline", r"\end{tabular}}", r"\end{table}", ""]
    return "\n".join(lines)


def step_tables(cfg: Config):
    R = load_json(cfg.data_dir / "results.json")
    md = ["# Tables\n", "Generated by `run_all.py`; full-precision values in the CSV files.\n"]
    with pd.ExcelWriter(cfg.table_dir / "tables.xlsx", engine="openpyxl") as xw:
        for name, title, fn in TABLES:
            try:
                df = fn(R, cfg)
            except Exception as exc:  # noqa: BLE001 - one failing table must not stop the rest
                log.warning("table %s skipped: %s", name, exc)
                continue
            if df is None or df.empty:
                continue
            df.to_csv(cfg.table_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
            disp = _display(df)
            disp.to_excel(xw, sheet_name=name[:31], index=False)
            (cfg.table_dir / f"{name}.tex").write_text(to_latex(disp, title, f"tab:{name}"), encoding="utf-8")
            md += [f"\n## {name.replace('_', ' ')} — {title}\n", disp.to_markdown(index=False), ""]
    (cfg.table_dir / "tables.md").write_text("\n".join(md), encoding="utf-8")
    log.info("tables written to %s", cfg.table_dir)
