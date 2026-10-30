"""Pipeline steps 1-6: ingest, drift QC, segmentation, legacy replication,
porosity profiles and REV. Each step writes its results to outputs/data and
can be re-run independently once its inputs exist."""
from __future__ import annotations

import logging

import numpy as np

from . import fractal, histogram as H, io, rev as revmod, stats as st, surface
from .context import SampleContext
from .util import load_json, save_json, timed
from .volume import ArraySource, Box, drift_scale

log = logging.getLogger("drp")


# ---------------------------------------------------------------------------
# 1. Ingest
# ---------------------------------------------------------------------------
def step_ingest(ctx: SampleContext):
    s = ctx.sample
    with timed(log, f"ingest {s.id}"):
        meta = io.ingest(s, ctx.cfg.cache_dir, ctx.cfg["preprocessing"]["mode_bins"])
    nz, ny, nx = meta["shape"]
    v = meta["voxel_size_um"] / 1000
    meta["extent_mm_zyx"] = [nz * v, ny * v, nx * v]
    meta["volume_mm3"] = nz * ny * nx * v**3
    meta["lab_porosity"] = s.lab_porosity
    save_json(ctx.data("acquisition"), meta)
    if meta.get("voxel_size_um_metadata") and abs(meta["voxel_size_um_metadata"] - ctx.cfg["legacy"]["assumed_voxel_um"]) > 0.05:
        log.warning("%s: metadata voxel size %.2f um differs from the %.1f um assumed in the legacy workflow",
                    s.id, meta["voxel_size_um_metadata"], ctx.cfg["legacy"]["assumed_voxel_um"])


# ---------------------------------------------------------------------------
# 2. Drift QC
# ---------------------------------------------------------------------------
def step_qc(ctx: SampleContext):
    """Solid-peak drift along z and the evidence for/against correcting it."""
    from scipy import stats as sps

    pre = ctx.cfg["preprocessing"]
    sh = io.load_slice_histograms(ctx.cfg.cache_dir, ctx.sample.id)
    width = int(sh["bin_width"])
    d = drift_scale(sh["slice_hists"], width, int(pre["drift_window"]))
    np.savez(ctx.data("qc", "npz"), scale=d["scale"], mode_raw=d["mode_raw"], mode_smooth=d["mode_smooth"],
             slice_mean=sh["slice_mean"], slice_sd=sh["slice_sd"])
    # apparent porosity of each slice at the legacy threshold of the raw stack
    T = H.t_mean_k_sigma(sh["total_hist"].astype(np.int64), float(ctx.cfg["legacy"]["k_sigma"]))
    hs = sh["slice_hists"].astype(np.float64)
    cut = T / width
    k = int(np.floor(cut))
    phi = (hs[:, :k].sum(1) + (cut - k) * hs[:, k]) / hs.sum(1)
    modes = d["mode_smooth"]
    # both series are almost perfectly autocorrelated along z, so rho gets a
    # circular-block bootstrap interval (blocks of ~10 % of the stack)
    rho_ci = st.spearman_ci(modes, phi, 1000, float(ctx.cfg["stats"]["confidence"]), ctx.cfg.seed,
                            block=int(pre.get("drift_rho_block", max(10, len(phi) // 10))))
    rho = rho_ci["rho"]
    rho_diff = sps.spearmanr(np.diff(d["mode_raw"]), np.diff(phi)).statistic
    # Slow gain drift (e.g. at the ends of a cone-beam stack) shows up in the
    # levels, not in slice-to-slice differences, which are dominated by noise;
    # the difference-based coefficient is reported only as a supplement. Both
    # series are almost perfectly autocorrelated, so the association is tested
    # against phase-randomised surrogates of the solid-peak series; whether
    # to correct a sample is set in config.toml (an a-priori choice) and the
    # profiles step reports the results with and without the correction.
    assoc = st.association_test(modes, phi, int(ctx.cfg.scaled("stats", "n_surrogates")), ctx.cfg.seed)
    out = {
        "reference_mode": d["reference"],
        "mode_min": float(modes.min()), "mode_max": float(modes.max()),
        "mode_range_percent": float(100 * (modes.max() - modes.min()) / d["reference"]),
        "scale_min": float(d["scale"].min()), "scale_max": float(d["scale"].max()),
        "slice_mean_range_percent": float(100 * np.ptp(sh["slice_mean"]) / np.mean(sh["slice_mean"])),
        "spearman_mode_vs_porosity": float(rho),
        "spearman_mode_vs_porosity_ci": [rho_ci["ci_low"], rho_ci["ci_high"]],
        "spearman_ci_resampling": rho_ci["resampling"],
        "spearman_p_surrogate": assoc["p_surrogate"],
        "association_class": assoc["association_class"],
        "mode_series": "running median of the per-slice solid-peak mode",
        "spearman_dmode_vs_dporosity": float(rho_diff),
        "drift_correction_applied": ctx.drift_corrected,
    }
    np.savez(ctx.data("qc", "npz"), scale=d["scale"], mode_raw=d["mode_raw"], mode_smooth=d["mode_smooth"],
             slice_mean=sh["slice_mean"], slice_sd=sh["slice_sd"], slice_porosity_raw_legacyT=phi)
    save_json(ctx.data("qc"), out)
    log.info("%s: solid-peak drift %.2f %%; rho(mode, phi)=%+.2f, rho(dmode, dphi)=%+.2f -> correction %s",
             ctx.sample.id, out["mode_range_percent"], rho, rho_diff, "ON" if ctx.drift_corrected else "OFF")
    cls = assoc["association_class"]
    if (cls == "negative" and not ctx.drift_corrected) or (cls == "positive" and ctx.drift_corrected):
        log.warning("%s: drift_correction=%s contradicts a significant %s association (surrogate p = %.3f)",
                    ctx.sample.id, ctx.drift_corrected, cls, assoc["p_surrogate"])
    elif cls == "inconclusive":
        log.info("%s: association between solid-peak level and slice porosity inconclusive (p = %.3f); "
                 "drift_correction=%s is an a-priori choice", ctx.sample.id, assoc["p_surrogate"], ctx.drift_corrected)


# ---------------------------------------------------------------------------
# 3. Segmentation
# ---------------------------------------------------------------------------
def _segmentation_block(h: np.ndarray, ctx: SampleContext, I_s: float, I_p: float, I_p_range) -> dict:
    cfg = ctx.cfg["segmentation"]
    lab = ctx.sample.lab_porosity
    k_leg = float(ctx.cfg["legacy"]["k_sigma"])
    n, mean, sd = H.moments(h)
    named = H.all_thresholds(h, I_s, I_p, lab, k_leg, float(cfg["primary_f"]))
    keys = ["legacy", "otsu", "li", "kittler", "triangle", "primary", "lab"]
    thr = {}
    for key, (name, T) in zip(keys, named.items()):
        phi = H.porosity_at(h, T)
        thr[key] = {"name": name, "T": float(T), "porosity": phi,
                    "k_equiv": (mean - T) / sd, "f_equiv": H.f_of_threshold(I_s, I_p, T),
                    "abs_error_lab": phi - lab, "rel_error_lab": (phi - lab) / lab}
    k = np.linspace(cfg["k_grid_min"], cfg["k_grid_max"], int(cfg["k_grid_n"]))
    f = np.asarray(cfg["f_grid"], float)
    f_fine = np.linspace(0.02, 0.98, 97)
    mix_u = H.mixing_porosity(h, I_s, I_p, clip=False)
    mix_c = H.mixing_porosity(h, I_s, I_p, clip=True)
    mix_u_rng = sorted(H.mixing_porosity(h, I_s, ip, clip=False) for ip in I_p_range)
    mix_c_rng = sorted(H.mixing_porosity(h, I_s, ip, clip=True) for ip in I_p_range)
    bg = float(cfg["background_porosity_scenario"])
    return {
        "n_voxels": n, "mean": mean, "sd": sd, "I_s": I_s, "I_p": I_p,
        "I_p_range": list(I_p_range), "contrast": I_s - I_p,
        "thresholds_by_key": thr,
        "k_curve": {"k": k, "T": mean - k * sd, "porosity": [H.porosity_at(h, mean - kk * sd) for kk in k]},
        "f_curve": {"f": f, "T": [H.t_partial_volume(I_s, I_p, ff) for ff in f],
                    "porosity": [H.porosity_at(h, H.t_partial_volume(I_s, I_p, ff)) for ff in f]},
        "f_curve_fine": {"f": f_fine, "porosity": [H.porosity_at(h, H.t_partial_volume(I_s, I_p, ff)) for ff in f_fine],
                         "porosity_Ip_lo": [H.porosity_at(h, H.t_partial_volume(I_s, I_p_range[0], ff)) for ff in f_fine],
                         "porosity_Ip_hi": [H.porosity_at(h, H.t_partial_volume(I_s, I_p_range[1], ff)) for ff in f_fine]},
        "k_lab": (mean - thr["lab"]["T"]) / sd,
        "f_lab": H.f_of_threshold(I_s, I_p, thr["lab"]["T"]),
        "mixing": {
            "unclipped": mix_u, "unclipped_range": mix_u_rng,
            "clipped": mix_c, "clipped_range": mix_c_rng,
            "implied_background": H.implied_background(lab, mix_u),
            "implied_background_range": sorted(H.implied_background(lab, m) for m in mix_u_rng),
            "implied_background_clipped": H.implied_background(lab, mix_c),
            "background_scenario": bg,
            "total_with_background_mixing": H.with_background(mix_u, bg),
            "total_with_background_legacy_binary": H.with_background(thr["legacy"]["porosity"], bg),
        },
        "histogram_excess_fraction": H.histogram_excess(h, I_s),
    }


def step_segmentation(ctx: SampleContext):
    cfg = ctx.cfg["segmentation"]
    pre = ctx.cfg["preprocessing"]
    stack = ctx.stack()
    with timed(log, f"corrected histogram {ctx.sample.id}"):
        h_corr = stack.histogram(slab=int(pre["slab"])) if stack.corrected else None
    sh = io.load_slice_histograms(ctx.cfg.cache_dir, ctx.sample.id)
    h_raw = sh["total_hist"].astype(np.int64)
    h = h_corr if h_corr is not None else h_raw

    def endmembers(hh):
        I_s = H.solid_mode(hh, int(pre["mode_bins"]))
        q = float(cfg["pore_endmember_quantile"])
        q_lo, q_hi = cfg["pore_endmember_quantile_range"]
        return I_s, H.pore_endmember(hh, q), (H.pore_endmember(hh, q_lo), H.pore_endmember(hh, q_hi))

    I_s, I_p, I_p_rng = endmembers(h)
    out = _segmentation_block(h, ctx, I_s, I_p, I_p_rng)
    out["drift_corrected"] = bool(stack.corrected)
    out["primary_f"] = float(cfg["primary_f"])
    out["T_primary"] = float(np.ceil(I_s - out["primary_f"] * (I_s - I_p)))
    Is0, Ip0, rng0 = endmembers(h_raw)
    out["uncorrected"] = _segmentation_block(h_raw, ctx, Is0, Ip0, rng0)
    for blk in (out, out["uncorrected"]):
        for key in ("k_curve", "f_curve", "f_curve_fine"):
            blk[key] = {k: np.asarray(v).tolist() for k, v in blk[key].items()}
    save_json(ctx.data("segmentation"), out)
    np.savez_compressed(ctx.data("histograms", "npz"), corrected=h, raw=h_raw)
    t = out["thresholds_by_key"]
    log.info("%s: I_s=%.0f I_p=%.0f | phi legacy=%.4f primary(f=%.2f)=%.4f mixing=%.4f lab=%.3f",
             ctx.sample.id, I_s, I_p, t["legacy"]["porosity"], out["primary_f"], t["primary"]["porosity"],
             out["mixing"]["unclipped"], ctx.sample.lab_porosity)


# ---------------------------------------------------------------------------
# 4. Replication of the published workflow
# ---------------------------------------------------------------------------
def step_legacy(ctx: SampleContext):
    L = ctx.cfg["legacy"]
    z0, z1 = int(L["z_start"]), int(L["z_stop"])
    stack = ctx.stack(corrected=False)
    nz, ny, nx = stack.shape
    z1 = min(z1, nz)
    box = Box(z0, z1, 0, ny, 0, nx)
    h = stack.histogram(box)
    n, mean, sd = H.moments(h)
    T = mean - float(L["k_sigma"]) * sd
    phi = H.porosity_at(h, T)
    with timed(log, f"legacy mask {ctx.sample.id}"):
        mask = np.empty(box.shape, bool)
        for a in range(0, box.shape[0], 32):
            b = min(a + 32, box.shape[0])
            mask[a:b] = stack.grey(z0 + a, z0 + b) < T
    src = ArraySource(mask)
    with timed(log, f"legacy marching cubes {ctx.sample.id}"):
        area_vox2 = surface.mc_area(src, 0.5, 1.0)
    with timed(log, f"legacy box counting {ctx.sample.id}"):
        bc = fractal.legacy_boxcount(mask)
    counts = surface.transition_counts(src)
    sv_crofton_vox = surface.sv_crofton(counts, 1.0)
    del mask, src
    n_vox = box.size
    pub = L["published"].get(ctx.sample.id, {})
    out = {"box": box.as_dict(), "shape_zyx": list(box.shape), "n_voxels": n_vox,
           "mean": mean, "sd": sd, "threshold": T, "porosity": phi,
           "area_vox2": area_vox2, "sv_mc_binary_per_vox": area_vox2 / n_vox,
           "sv_crofton13_per_vox": sv_crofton_vox, "boxcount": bc, "published": pub}
    for tag, vox_um in (("assumed", float(L["assumed_voxel_um"])), ("metadata", ctx.voxel_um)):
        v = vox_um / 1000
        out[tag] = {
            "voxel_um": vox_um,
            "extent_mm_zyx": [s * v for s in box.shape],
            "volume_mm3": n_vox * v**3,
            "pore_volume_mm3": phi * n_vox * v**3,
            "area_mm2": area_vox2 * v**2,
            "sv_mc_binary_per_mm": area_vox2 / n_vox / v,
            "sv_crofton13_per_mm": sv_crofton_vox / v,
        }
    save_json(ctx.data("legacy"), out)
    log.info("%s legacy: phi=%.4f (pub %s), A=%.2f mm2 @28um (pub %s), D=%.4f (pub %s)", ctx.sample.id, phi,
             pub.get("porosity"), out["assumed"]["area_mm2"], pub.get("area_mm2"), bc["D_legacy"],
             pub.get("fractal_dimension"))


# ---------------------------------------------------------------------------
# 5. Porosity profiles
# ---------------------------------------------------------------------------
def _stream_profiles(stack, T: float, I_s: float, I_p: float, slab: int, zwin=None) -> dict:
    """Slice porosity along z, y and x; with `zwin` = (z0, z1) the y and x
    profiles are also computed over that z window only ("y_zwin", "x_zwin"),
    so specimens of different height can be compared over equal extents."""
    nz, ny, nx = stack.shape
    cz = np.zeros(nz)
    cy = np.zeros(ny)
    cx = np.zeros(nx)
    gz = np.zeros(nz)
    gy = np.zeros(ny)
    gx = np.zeros(nx)
    wy, wx, wgy, wgx = np.zeros(ny), np.zeros(nx), np.zeros(ny), np.zeros(nx)
    zw0, zw1 = zwin if zwin is not None else (0, nz)
    for z0 in range(0, nz, slab):
        z1 = min(z0 + slab, nz)
        g = stack.grey(z0, z1)
        m = g < T
        cz[z0:z1] = m.sum(axis=(1, 2))
        cy += m.sum(axis=(0, 2))
        cx += m.sum(axis=(0, 1))
        gf = g.astype(np.float64)
        gz[z0:z1] = gf.sum(axis=(1, 2))
        gy += gf.sum(axis=(0, 2))
        gx += gf.sum(axis=(0, 1))
        a, b = max(z0, zw0), min(z1, zw1)
        if a < b:
            mm, gg = m[a - z0 : b - z0], gf[a - z0 : b - z0]
            wy += mm.sum(axis=(0, 2))
            wx += mm.sum(axis=(0, 1))
            wgy += gg.sum(axis=(0, 2))
            wgx += gg.sum(axis=(0, 1))
    per = {"z": ny * nx, "y": nz * nx, "x": nz * ny}
    out = {}
    for a, c, gsum in (("z", cz, gz), ("y", cy, gy), ("x", cx, gx)):
        out[a] = {"binary": c / per[a], "mixing": (I_s - gsum / per[a]) / (I_s - I_p)}
    nw = zw1 - zw0
    for a, c, gsum, n in (("y_zwin", wy, wgy, nw * nx), ("x_zwin", wx, wgx, nw * ny)):
        out[a] = {"binary": c / n, "mixing": (I_s - gsum / n) / (I_s - I_p)}
    return out


def _slab_average(y: np.ndarray, n: int) -> np.ndarray:
    k = len(y) // n
    return y[: k * n].reshape(k, n).mean(axis=1) if k else np.array([])


def _common_z_window(ctx: SampleContext) -> tuple[int, int, float]:
    """Central z window whose length is the shortest stack height of all samples."""
    lengths = []
    for s in ctx.cfg.samples:
        p = ctx.cfg.data_dir / f"acquisition_{s.id}.json"
        if p.exists():
            lengths.append(load_json(p)["extent_mm_zyx"][0])
    L = min(lengths) if lengths else ctx.shape[0] * ctx.voxel_mm
    n_keep = min(ctx.shape[0], int(round(L / ctx.voxel_mm)))
    z0 = (ctx.shape[0] - n_keep) // 2
    return z0, z0 + n_keep, float(L)


def step_profiles(ctx: SampleContext):
    cfg = ctx.cfg
    slab = int(cfg["preprocessing"]["slab"])
    stack = ctx.stack()
    zw0, zw1, L_common = _common_z_window(ctx)
    n_surr = int(cfg.scaled("stats", "n_surrogates"))
    unc = ctx.seg["uncorrected"]
    with timed(log, f"profiles {ctx.sample.id}"):
        prof = _stream_profiles(stack, ctx.T_primary, ctx.I_s, ctx.I_p, slab, (zw0, zw1))
        # uncorrected stack (drift QC and sensitivity): legacy and primary
        # thresholds and the mass balance, each with the uncorrected end-members
        raw = ctx.stack(corrected=False)
        T_raw = unc["thresholds_by_key"]["legacy"]["T"]
        T_raw_p = unc["thresholds_by_key"]["primary"]["T"]
        Is0, Ip0 = unc["I_s"], unc["I_p"]
        nz = raw.shape[0]
        prof_raw_z, raw_bin, raw_mix = np.empty(nz), np.empty(nz), np.empty(nz)
        for z in range(nz):
            g = raw.grey(z, z + 1)
            prof_raw_z[z] = (g < T_raw).mean()
            raw_bin[z] = (g < T_raw_p).mean()
            raw_mix[z] = (Is0 - g.mean(dtype=np.float64)) / (Is0 - Ip0)
        T_leg = ctx.T_legacy
        prof_leg = _stream_profiles(stack, T_leg, ctx.I_s, ctx.I_p, slab, (zw0, zw1))
    v = ctx.voxel_mm
    arrays = {}
    summary = {"voxel_mm": v, "T_primary": ctx.T_primary, "T_legacy": T_leg, "axes": {},
               "z_window_for_xy": {"z0": zw0, "z1": zw1, "length_mm": (zw1 - zw0) * v, "common_length_mm": L_common}}
    for a in ("y_zwin", "x_zwin"):
        for kind in ("binary", "mixing"):
            arrays[f"{a}_{kind}"] = prof[a][kind]
        arrays[f"{a}_legacy"] = prof_leg[a]["binary"]
    for a in "zyx":
        for kind in ("binary", "mixing"):
            y = prof[a][kind]
            arrays[f"{a}_{kind}"] = y
        arrays[f"{a}_legacy"] = prof_leg[a]["binary"]
        n = len(prof[a]["binary"])
        pos = (np.arange(n) + 0.5) * v
        arrays[f"{a}_pos_mm"] = pos
        ax_sum = {"length_mm": n * v, "n_slices": n}
        for kind, y in (("binary", prof[a]["binary"]), ("mixing", prof[a]["mixing"]),
                        ("legacy", prof_leg[a]["binary"])):
            d = {"mean": float(y.mean()), "sd": float(y.std(ddof=1)), "cv": st.cv(y),
                 "min": float(y.min()), "max": float(y.max())}
            # effective size and mean interval on 1-mm slab averages: per-slice
            # series are oversampled (lag-1 > 0.98) and their AR(1) n_eff is
            # set by the cap rather than by the data
            ns1 = max(1, int(round(float(cfg["profiles"]["comparison_slab_mm"]) / v)))
            d.update({f"neff_{k}": val for k, val in st.mean_ci_neff(_slab_average(y, ns1)).items()})
            d["neff_slab_mm"] = ns1 * v
            d["trend_per_10mm"] = {k: (val * 10 if k in ("slope", "slope_se", "ci_low", "ci_high") else val)
                                   for k, val in st.trend_test(pos, y, n_surr, seed=cfg.seed).items()}
            d["cv_by_slab_mm"] = {}
            for t in cfg["profiles"]["slab_mm"]:
                ns = max(1, int(round(t / v)))
                ya = _slab_average(y, ns)
                if len(ya) > 2:
                    d["cv_by_slab_mm"][str(t)] = {"n_slabs": len(ya), "slab_mm": ns * v, "cv": st.cv(ya)}
            ax_sum[kind] = d
        summary["axes"][a] = ax_sum
    arrays["z_uncorrected_legacyT"] = prof_raw_z
    arrays["z_uncorrected_binary"] = raw_bin
    arrays["z_uncorrected_mixing"] = raw_mix
    np.savez_compressed(ctx.data("profiles", "npz"), **arrays)
    # drift diagnostic: association between solid-peak level and slice
    # porosity, before and (if applied) after the correction, with
    # circular-block bootstrap intervals
    mode = ctx.drift["mode_smooth"]          # same series as step_qc
    blk = int(cfg["preprocessing"].get("drift_rho_block", max(10, len(mode) // 10)))
    rho_u = st.spearman_ci(mode, prof_raw_z, 1000, seed=cfg.seed, block=blk)
    rho_c = st.spearman_ci(mode, prof_leg["z"]["binary"], 1000, seed=cfg.seed, block=blk) if ctx.drift_corrected else None
    summary["drift_diagnostic"] = {
        "spearman_mode_vs_porosity_uncorrected": rho_u["rho"],
        "spearman_mode_vs_porosity_uncorrected_ci": [rho_u["ci_low"], rho_u["ci_high"]],
        "spearman_mode_vs_porosity_corrected": rho_c["rho"] if rho_c else None,
        "spearman_mode_vs_porosity_corrected_ci": [rho_c["ci_low"], rho_c["ci_high"]] if rho_c else None,
        "correction_applied": ctx.drift_corrected,
    }
    # sensitivity of the z statistics to the drift correction
    pos = arrays["z_pos_mm"]
    sens = {}
    for kind, y_corr, y_raw in (("binary", prof["z"]["binary"], raw_bin), ("mixing", prof["z"]["mixing"], raw_mix),
                                ("legacy", prof_leg["z"]["binary"], prof_raw_z)):
        sens[kind] = {}
        for tag, y in (("as_analysed", y_corr), ("uncorrected", y_raw)):
            tr = st.trend_test(pos, y, n_surr, seed=cfg.seed)
            sens[kind][tag] = {"mean": float(y.mean()), "cv": st.cv(y), "trend_per_10mm": tr["slope"] * 10,
                               "trend_p": tr["p_value"]}
    summary["drift_sensitivity_z"] = sens
    save_json(ctx.data("profiles"), summary)


# ---------------------------------------------------------------------------
# 6. REV / integral range
# ---------------------------------------------------------------------------
def step_rev(ctx: SampleContext):
    """Variance-volume scaling of the binary (f = 0.5, mean - 1.1 sd) and the
    mass-balance porosity, with block-bootstrap SEs of alpha and a Monte Carlo
    test against stationary media with a finite integral range."""
    cfg = ctx.cfg["rev"]
    stack = ctx.stack()
    block = int(cfg["block_voxels"])
    conf = float(ctx.cfg["stats"]["confidence"])
    seed = ctx.cfg.seed
    n_win = int(ctx.cfg.scaled("rev", "n_windows"))
    fit_min = float(cfg["fit_min_mm"])
    cache = ctx.cfg.cache_dir / f"{ctx.sample.id}_rev_blocks.npz"
    key = {"T_primary": ctx.T_primary, "T_legacy": ctx.T_legacy, "block": block, "drift": ctx.drift_corrected}
    grids = {}
    if cache.exists():
        with np.load(cache) as z:
            if all(np.isclose(float(z[f"meta_{k}"]), float(v)) for k, v in key.items()):
                grids = {k: z[k] for k in ("primary", "legacy", "mixing")}
    if not grids:
        with timed(log, f"REV block grids {ctx.sample.id}"):
            grids = {"primary": revmod.block_counts(stack, ctx.T_primary, block),
                     "legacy": revmod.block_counts(stack, ctx.T_legacy, block),
                     "mixing": revmod.block_grey_sums(stack, block)}
        np.savez_compressed(cache, **grids, **{f"meta_{k}": float(v) for k, v in key.items()})
    I_s, I_p = ctx.I_s, ctx.I_p
    to_mix = lambda g: (I_s - g) / (I_s - I_p)  # noqa: E731 - window mean grey -> mass-balance porosity
    vol = float(np.prod(ctx.shape)) * ctx.voxel_mm**3
    out = {}
    for tag, T in (("primary", ctx.T_primary), ("legacy", ctx.T_legacy), ("mixing", None)):
        counts = grids[tag]
        tr = to_mix if tag == "mixing" else None
        domain = float(np.prod(counts.shape)) * (block * ctx.voxel_mm) ** 3
        if tag == "mixing":
            phi = float(to_mix(counts.sum() / (counts.size * block**3)))
        else:
            phi = float(counts.sum() / (counts.size * block**3))
        rows = revmod.variance_curve(counts, block, ctx.voxel_mm, cfg["edges_mm"], n_win, seed=seed, transform=tr)
        fit = revmod.integral_range(rows, phi, fit_min, float(cfg["target_relative_error"]), vol, conf,
                                    domain_volume_mm3=domain)
        with timed(log, f"REV block bootstrap {ctx.sample.id} ({tag})"):
            bb = revmod.alpha_block_bootstrap(counts, block, ctx.voxel_mm, cfg["edges_mm"], n_win, fit_min,
                                              float(cfg["bootstrap_slab_mm"]), int(ctx.cfg.scaled("rev", "n_boot_alpha")),
                                              conf, seed, tr)
        zc = bb["z"]
        fit.update({"alpha_se_boot": bb["alpha_se_boot"], "alpha_boot": bb,
                    "alpha_ci_low": fit["alpha"] - zc * bb["alpha_se_boot"],
                    "alpha_ci_high": fit["alpha"] + zc * bb["alpha_se_boot"],
                    "alpha_finite_domain_se_boot": bb["alpha_finite_domain_se_boot"]})
        out[tag] = {"threshold": T, "porosity_blocks": phi, "curve": rows, "fit": fit}
    with timed(log, f"REV null simulations {ctx.sample.id}"):
        null = revmod.alpha_null_test(
            {t: (out[t]["fit"]["alpha"], out[t]["fit"]["alpha_finite_domain"]) for t in out},
            grids["primary"].shape, block, ctx.voxel_mm, cfg["edges_mm"], n_win, fit_min,
            {"primary": out["primary"]["porosity_blocks"], "legacy": out["legacy"]["porosity_blocks"], "mixing": None},
            float(cfg["null_corr_mm"]), int(ctx.cfg.scaled("rev", "null_n")), seed)
    for tag in out:
        f = out[tag]["fit"]
        f["null"] = {**{k: v for k, v in null.items() if not isinstance(v, dict)}, **null[tag]["alpha"]}
        f["null_finite_domain"] = null[tag]["alpha_finite_domain"]
        nl = null[tag]["alpha"]
        f["null_testable"] = bool((nl["n_valid"] + 1) * (1 - conf) / 2 >= 1)
        f["alpha_vs_null"] = ("not testable" if not f["null_testable"] else
                              "below" if f["alpha"] < nl["null_q025"] else
                              "above" if f["alpha"] > nl["null_q975"] else "within")
        f["alpha_within_null_band"] = f["alpha_vs_null"] == "within" if f["null_testable"] else None
        log.info("%s REV (%s): alpha=%.2f (block-bootstrap SE %.3f; finite-domain %.2f); null %.2f [%.2f, %.2f], "
                 "p=%.3f; SE(phi, stack)=%.4f (conservative %.4f)", ctx.sample.id, tag, f["alpha"], f["alpha_se_boot"],
                 f["alpha_finite_domain"], f["null"]["null_mean"], f["null"]["null_q025"], f["null"]["null_q975"],
                 f["null"]["p_one_sided"], f["porosity_se_total_powerlaw"], f["porosity_se_total_conservative"])
    save_json(ctx.data("rev"), out)


__all__ = ["step_ingest", "step_qc", "step_segmentation", "step_legacy", "step_profiles", "step_rev",
           "load_json"]
