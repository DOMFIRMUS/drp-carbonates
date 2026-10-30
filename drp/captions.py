"""Figure captions (English) with numbers taken from results.json.

Writes outputs/FIGURE_CAPTIONS.md. Captions carry the information that
journals forbid inside the figure (titles, sample sizes, definitions).
"""
from __future__ import annotations

import logging

from .config import Config
from .util import load_json

log = logging.getLogger("drp")

# suggested placement: main text (M) or electronic supplementary material (S)
ORDER = [("F01", "M"), ("F02", "M"), ("F03", "M"), ("F06", "M"), ("F07", "M"), ("F08", "M"), ("F11", "M"),
         ("F13", "M"), ("F04", "S"), ("F10", "S"), ("F09", "S"), ("F05", "S"), ("F12", "S")]


def captions(R: dict, cfg: Config) -> dict[str, str]:
    S = R["samples"]
    a, b = list(S)
    la, lb = S[a]["label"], S[b]["label"]
    va, vb = (S[s]["acquisition"]["voxel_size_um"] for s in (a, b))
    cube = S[a]["cube"]["edge_mm"] if S[a].get("cube") else 20
    f = R["config_primary_f"]
    rp = R.get("replicates", {})
    n_a = rp.get("n_cubes", {}).get(a, "–")
    n_b = rp.get("n_cubes", {}).get(b, "–")
    edge_rep = rp.get("edge_mm", {}).get(a, 5)
    ea = S[a].get("esf") or {}
    eb = S[b].get("esf") or {}
    rv = S[a]["rev"]["legacy"]["fit"]
    rng_rev = f"{rv['fit_min_mm']:.1f}–{rv['fit_max_mm']:.0f} mm"
    fda = (S[a].get("fractal_diagnostics") or {}).get("common_range_fit", {})
    fdb = (S[b].get("fractal_diagnostics") or {}).get("common_range_fit", {})
    rng_d = (f"{fda.get('s_min', 0.1):.2f}–{fda.get('s_max', 2.5):.2f} mm ({la}) and "
             f"{fdb.get('s_min', 0.1):.2f}–{fdb.get('s_max', 2.5):.2f} mm ({lb})")
    nd = {sid: rp.get("metrics", {}).get("D_unsat", {}).get(sid, {}).get("n", "–") for sid in (a, b)}
    ex = (rp.get("D_unsat_exclusion") or {}).get("per_sample", {})
    n_ex = sum((ex.get(sid) or {}).get("n_excluded", 0) for sid in (a, b))
    more = all((ex.get(sid) or {}).get("excluded_are") in (None, "more porous") for sid in (a, b))
    d_unsat_note = (f"n = {nd[a]} and {nd[b]}: cubes with fewer than three unsaturated box sizes"
                    + (", the more porous ones," if more else "") + " are excluded" if n_ex else
                    "defined for every cube")
    corrected = [S[s]["label"] for s in (a, b) if S[s]["qc"]["drift_correction_applied"]]
    corr_txt = (" and ".join(corrected) if corrected else "no sample")
    all_inconc = all(S[s]["qc"].get("association_class") == "inconclusive" for s in (a, b))
    why = ("an a-priori choice: the association between solid-peak level and slice porosity is not significant "
           "against autocorrelated surrogates in either sample" if all_inconc else
           "association test in Table 1")
    return {
        "F01": (f"Representative XY slices through the centre of the analysed {cube:.0f}-mm cubes of the {la} "
                f"(Calcite-1; voxel {va} µm; a–c) and the {lb} (Dolomite-3; voxel {vb} µm; d–f). (a, d) Whole "
                "cropped slice; the box marks the 4 × 4 mm area enlarged in the other panels. (b, e) Enlarged "
                f"grey levels. (c, f) Segmentation overlay: red, voxels classified as pore at the partial-volume "
                f"criterion f = {f:g} (majority-pore voxels); orange, voxels added by the mean − 1.1 sd threshold "
                "of the earlier analysis."
                + (f" Grey levels of the {corr_txt} are drift-corrected." if corrected else "")),
        "F02": ("(a) Grey-level distributions of the two stacks on the partial-volume scale "
                "f = (I_s − I)/(I_s − I_p), where I_s is the solid-peak mode and I_p the pore end-member (0.1 % "
                "grey-level quantile); the solid line marks f = 0.5 and the dotted lines the f that reproduces "
                "each laboratory porosity. (b) Porosity of the whole stack as a function of the segmentation "
                "criterion f; shaded bands span the pore end-member range (0.02–0.5 % quantiles). Markers give "
                "the thresholds of unsupervised methods and of the mean − 1.1 sd rule of the earlier analysis; dashed lines, "
                "laboratory porosity; dotted lines, threshold-free grey-level mass-balance porosity."),
        "F03": ("(a) Image porosity of the whole stacks by three estimators against the laboratory value. Error "
                "bars: 95 % spatial-sampling intervals from the variance–volume scaling extrapolated to the stack "
                "volume (the larger of two extrapolations; image) and ± 1.96 × the assumed laboratory standard "
                "deviation (0.005); shaded bars: range over the pore end-member. (b) Porosity that the solid phase "
                "must hold below the resolution to close the budget, φ_bg = (φ_lab − φ_mb)/(1 − φ_mb); error bars "
                "combine the spatial SE of the mass balance and the assumed laboratory SD; the dashed line is the "
                "5 % background porosity assumed by the data depositors."),
        "F06": ("Variance–volume scaling of porosity. (a) Coefficient of variation of the porosity of 2000 "
                "random cubic windows per size against window edge, for the mean − 1.1 sd threshold (filled) "
                f"and f = {f:g} (open), with power-law fits over windows of {rng_rev}; the dotted line is the "
                "slope −3/2 expected for a medium with a finite integral range (representative elementary volume). "
                "(b) Mean and 5th–95th percentile range of window porosity (mean − 1.1 sd); dashed lines, "
                "laboratory porosity."),
        "F07": (f"Specific surface S_V of the central {cube:.0f}-mm cubes. (a) Five estimators at f = {f:g}: "
                "Cauchy–Crofton intercept counting over 13 and 3 lattice directions, marching cubes on the grey "
                "levels and on the binary image, and exposed voxel faces (upper bound). (b) S_V (Crofton, 13 "
                "directions) against the segmentation criterion; stars, mean − 1.1 sd. (c) S_V against voxel "
                "size after area-weighted block averaging (large symbols: native resolution), with the slope "
                "of a log–log fit."),
        "F08": (f"Box counting of the pore–solid interface in the central {cube:.0f}-mm cubes (boxes counted when "
                "they contain boundary voxels; four grid offsets). (a) Number of occupied boxes with least-squares "
                f"fits over the common nominal range 0.1–2.5 mm (shaded; realised end points, within one voxel: {rng_d}). "
                "(b) Local slope −d ln N/d ln s; dotted lines mark the box size at which half of all boxes are "
                "occupied. (c) Occupancy N s³/V; open symbols: box sizes used in the fit of the earlier analysis "
                "(porespy definition, metadata voxel size). (d) Box-counting slope over the same range and box "
                "sizes against the segmentation criterion (bars: 95 % t-based fit intervals, descriptive); dashed "
                "lines, values of the earlier analysis."),
        "F11": (f"Effective spatial resolution. (a) Mean normalised grey-level profile across pore walls "
                f"(band: interquartile range) of full-contrast, well-fitted edges ({ea.get('n_kept', '–')} "
                f"edges in the {la}, {eb.get('n_kept', '–')} in the {lb}). (b) Distribution of the fitted "
                "edge-spread σ; dashed lines, voxel size. σ approximates the point-spread function: wall "
                "curvature and roughness widen edges, whereas a second interface within the fitting window can "
                "narrow them by a few per cent."),
        "F13": (f"Resolution experiment: the grey levels of the central cubes are block-averaged (area-weighted) "
                f"to coarser voxels and re-segmented with the same physical criterion f = {f:g}. (a) Porosity "
                "(dotted: grey-level mass balance, invariant by construction), (b) specific surface (Crofton), "
                "(c) box-counting slope over the unsaturated range, (d) fraction of the pore volume in the "
                "largest 6-connected cluster. Dashed vertical lines: native voxel sizes."),
        "F04": ("Gain drift along the stacks. (a) Solid-peak grey level of every slice relative to its median "
                "(thin) and running median over 31 slices (thick). (b) Slice porosity at the mean − 1.1 sd "
                f"threshold before (thin) and after (thick) drift correction; the correction is applied to the "
                f"{corr_txt} ({why})."),
        "F05": ("Porosity profiles along x, y and z for the grey-level mass balance (a–c) and the binary "
                f"segmentation at f = {f:g} (d–f): per-slice values (thin) and 1-mm slab averages (thick)."),
        "F09": (f"(a) Complementary cumulative distribution of the volumes of 6-connected pore clusters in the "
                f"central cubes (f = {f:g}). (b) Continuous pore-size distribution (opening-based) of the central "
                "8-mm cubes; dashed lines, radius of two voxels. (c) Fraction of the pore volume in the largest "
                "cluster (filled) and in clusters spanning the cube (open) against the segmentation criterion."),
        "F10": (f"Normalised two-point autocovariance C(r) of the pore phase (f = {f:g}) along x, y and z in the "
                "central cubes; the dashed line marks 1/e."),
        "F12": (f"Distributions over non-overlapping {edge_rep:.0f}-mm sub-volumes tiling the whole stacks "
                f"(n = {n_a} in the {la}, {n_b} in the {lb}): porosity by mass balance and at f = {f:g}, S_V, "
                "mean pore chord 4φ/S_V, box-counting slope over the unsaturated range "
                f"({d_unsat_note}), crossover length and largest-cluster fraction; (h) S_V against porosity. Boxes: "
                "quartiles and 1.5 × IQR whiskers."),
    }


def step_captions(cfg: Config):
    R = load_json(cfg.data_dir / "results.json")
    caps = captions(R, cfg)
    lines = ["# Figure captions (generated)\n",
             "Suggested numbering: main-text figures (M) in order, then supplementary figures (S).\n"]
    m = s = 0
    for key, where in ORDER:
        if where == "M":
            m += 1
            num = f"Fig. {m}"
        else:
            s += 1
            num = f"Fig. S{s}"
        lines.append(f"**{num}** (`{key}`). {caps[key]}\n")
    (cfg.output_dir / "FIGURE_CAPTIONS.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("FIGURE_CAPTIONS.md written")
