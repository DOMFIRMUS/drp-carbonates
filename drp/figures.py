"""Publication figures (PDF + 600-dpi PNG), built from saved results only.

Figure list (suggested placement in brackets):
  F01  micro-CT slices, segmentation overlays and scale bars        [main]
  F02  grey levels on the partial-volume scale; porosity vs criterion [main]
  F03  porosity budget: image vs laboratory; sub-resolution porosity  [main]
  F04  solid-peak drift along z and its effect on porosity            [supp]
  F05  porosity profiles along x, y, z                                [supp]
  F06  variance-volume scaling (REV)                                  [main]
  F07  specific surface: estimators, criterion, resolution            [main]
  F08  box counting: counts, local slope, saturation, criterion       [main]
  F09  connectivity and pore-size distribution                        [supp]
  F10  two-point correlation functions                                [supp]
  F11  effective resolution from edge-spread profiles                 [main]
  F12  replicate sub-volumes                                          [supp]
  F13  resolution (coarsening) experiment                             [main]
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from . import histogram as H
from . import plotting as P
from .config import Config
from .context import SampleContext
from .util import load_json
from .volume import central_cube

log = logging.getLogger("drp")

METHOD_MARKERS = {"legacy": ("*", "mean − 1.1 sd"), "otsu": ("o", "Otsu"), "li": ("s", "Li"),
                  "kittler": ("^", "Kittler–Illingworth"), "triangle": ("D", "Triangle"),
                  "primary": ("P", "f = 0.5"), "lab": ("X", "lab-matched")}
AXIS_NAME = {"x": "x", "y": "y", "z": "z"}


def plain_log_axis(ax, axis="x", ticks=None):
    """Log axis with plain-number tick labels (no 2x10^0 clutter)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    a = ax.xaxis if axis == "x" else ax.yaxis
    if ticks is not None:
        a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(fmt)
    a.set_minor_formatter(NullFormatter())


class Fig:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        f = cfg["figures"]
        P.apply_style(f["font"], f["font_size"])
        self.W2 = f["width_double_mm"]
        self.W1 = f["width_single_mm"]
        self.formats = f["formats"]
        self.dpi = f["dpi"]
        self.ctxs = [SampleContext(cfg, s) for s in cfg.samples]
        self.R = load_json(cfg.data_dir / "results.json")

    def save(self, fig, name):
        P.save(fig, self.cfg.fig_dir / name, self.formats, self.dpi)
        log.info("figure %s", name)

    def npz(self, ctx, step):
        with np.load(ctx.data(step, "npz")) as z:
            return {k: z[k] for k in z.files}

    @staticmethod
    def legend_samples(ax, ctxs, loc="best", extra=()):
        handles = [Line2D([], [], color=c.sample.color, lw=1.4, ls=P.sample_style(i)["linestyle"],
                          label=c.sample.label) for i, c in enumerate(ctxs)]
        ax.legend(handles=handles + list(extra), loc=loc)

    # -----------------------------------------------------------------------
    def f01_slices(self):
        n = len(self.ctxs)
        fig, axes = P.figure(self.W2, 62 * n, n, 3)
        axes = np.atleast_2d(axes)
        zoom_mm = 4.0
        for i, c in enumerate(self.ctxs):
            stack = c.stack()
            box = central_cube(stack.shape, c.vox(float(self.cfg.scaled("volumes", "central_cube_mm"))))
            z = (box.z0 + box.z1) // 2
            g = stack.grey(z, z + 1)[0].astype(np.float32)
            v = c.voxel_mm
            ny, nx = g.shape
            lo, hi = np.percentile(g, [0.5, 99.5])
            ext = [0, nx * v, ny * v, 0]
            ax = axes[i, 0]
            ax.imshow(g, cmap="gray", vmin=lo, vmax=hi, extent=ext, interpolation="nearest")
            cx, cy = nx * v / 2, ny * v / 2
            ax.add_patch(Rectangle((cx - zoom_mm / 2, cy - zoom_mm / 2), zoom_mm, zoom_mm, fill=False,
                                   ec="#ffd34d", lw=0.9))
            ax.set_axis_off()
            P.scale_bar(ax, float(self.cfg["figures"]["scale_bar_mm"]), f"{self.cfg['figures']['scale_bar_mm']:g} mm")
            ax.text(-0.04, 0.5, f"{c.sample.label}\n{c.voxel_um:.2f} µm voxel", transform=ax.transAxes,
                    rotation=90, ha="right", va="center")
            # zoom
            hz = int(round(zoom_mm / 2 / v))
            yc, xc = ny // 2, nx // 2
            sub = g[yc - hz: yc + hz, xc - hz: xc + hz]
            ext_z = [0, sub.shape[1] * v, sub.shape[0] * v, 0]
            for j in (1, 2):
                ax = axes[i, j]
                ax.imshow(sub, cmap="gray", vmin=lo, vmax=hi, extent=ext_z, interpolation="nearest")
                ax.set_axis_off()
            leg_mask = sub < c.T_legacy
            pri_mask = sub < c.T_primary
            ov = np.zeros(sub.shape + (4,))
            ov[leg_mask & ~pri_mask] = (0.93, 0.63, 0.0, 0.85)   # pore only at mean - 1.1 sd
            ov[pri_mask] = (0.89, 0.29, 0.28, 0.95)              # pore at f = 0.5
            axes[i, 2].imshow(ov, extent=ext_z, interpolation="nearest")
            P.scale_bar(axes[i, 1], 1.0, "1 mm")
            P.scale_bar(axes[i, 2], 1.0, "1 mm")
        fig.legend(handles=[Patch(color=(0.89, 0.29, 0.28), label="pore at f = 0.5 (majority-pore voxels)"),
                            Patch(color=(0.93, 0.63, 0.0), label="added by mean − 1.1 sd")],
                   loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
        P.panel_labels(axes.ravel(), images=True, y=1.01)
        self.save(fig, "F01_slices")

    # -----------------------------------------------------------------------
    def f02_segmentation(self):
        fig, (a1, a2) = P.figure(self.W2, 72, 1, 2)
        for i, c in enumerate(self.ctxs):
            st = P.sample_style(i)
            h = self.npz(c, "histograms")["corrected"]
            I_s, I_p = c.I_s, c.I_p
            lev = np.nonzero(h)[0]
            f = (I_s - lev) / (I_s - I_p)
            bins = np.arange(-0.6, 1.25, 0.01)
            dens, _ = np.histogram(f, bins=bins, weights=h[lev])
            dens = dens / (h.sum() * 0.01)
            centers = 0.5 * (bins[1:] + bins[:-1])
            ok = dens > 0
            a1.semilogy(centers[ok], dens[ok], color=c.sample.color, ls=st["linestyle"], label=c.sample.label)
            a1.axvline(c.seg["f_lab"], color=c.sample.color, lw=0.8, ls=":")
            fc = c.seg["f_curve_fine"]
            ff = np.asarray(fc["f"])
            a2.fill_between(ff, fc["porosity_Ip_hi"], fc["porosity_Ip_lo"], color=c.sample.color, alpha=0.15, lw=0)
            a2.plot(ff, fc["porosity"], color=c.sample.color, ls=st["linestyle"])
            a2.axhline(c.sample.lab_porosity, color=c.sample.color, lw=0.9, ls="--")
            a2.axhline(c.seg["mixing"]["unclipped"], color=c.sample.color, lw=0.9, ls=":")
            for key, t in c.seg["thresholds_by_key"].items():
                mk, _ = METHOD_MARKERS[key]
                a2.plot(t["f_equiv"], t["porosity"], mk, color=c.sample.color, ms=5, mec="white", mew=0.5, zorder=5)
        a1.axvline(0.5, color=P.INK_2, lw=0.8)
        a1.set_xlabel(r"Partial-volume pore fraction $f = (I_s - I)/(I_s - I_p)$")
        a1.set_ylabel("Density (per unit f)")
        a1.set_xlim(-0.6, 1.2)
        a1.set_ylim(1e-6, None)
        a1.text(0.51, 0.97, "f = 0.5", transform=a1.get_xaxis_transform(), va="top", fontsize=7, color=P.INK_2)
        self.legend_samples(a1, self.ctxs, "lower left",
                            extra=[Line2D([], [], color=P.NEUTRAL, ls=":", label="f reproducing lab porosity")])
        a2.set_xlabel(r"Segmentation criterion $f$ (pore if $f(I) > f$)")
        a2.set_ylabel("Porosity")
        a2.set_xlim(0, 1)
        a2.set_ylim(0, None)
        meth = [Line2D([], [], marker=m, ls="", color=P.NEUTRAL, label=lab) for m, lab in METHOD_MARKERS.values()]
        meth += [Line2D([], [], color=P.NEUTRAL, ls="--", label="laboratory"),
                 Line2D([], [], color=P.NEUTRAL, ls=":", label="grey mass balance"),
                 Patch(color=P.NEUTRAL, alpha=0.3, label="pore end-member range")]
        a2.legend(handles=meth, ncol=2, loc="upper right", fontsize=7, handlelength=1.4, columnspacing=0.8)
        P.panel_labels([a1, a2])
        self.save(fig, "F02_segmentation")

    # -----------------------------------------------------------------------
    def f03_budget(self):
        fig, (a1, a2) = P.figure(self.W2, 62, 1, 2, gridspec_kw={"width_ratios": [2.2, 1]})
        rows = [("Laboratory", None), ("Grey mass balance", "mixing"),
                ("Binary, mean − 1.1 sd", "legacy_threshold"), ("Binary, f = 0.5", "primary_threshold")]
        z = 1.96
        yt, yl = [], []
        for i, c in enumerate(self.ctxs):
            b = self.R["samples"][c.sample.id]["porosity_budget"]
            base = i * (len(rows) + 1)
            for j, (lab, key) in enumerate(rows):
                y = base + j
                yt.append(y)
                yl.append(f"{c.sample.label}: {lab}")
                if key is None:
                    x, se = b["lab"], b["lab_sd_assumed"]
                    a1.errorbar(x, y, xerr=z * se, fmt="s", color=c.sample.color, ms=4, capsize=2, lw=1)
                    a1.axvline(x, color=c.sample.color, lw=0.6, ls="--", alpha=0.6)
                    continue
                d = b[key]
                if "endmember_range" in d:
                    lo, hi = d["endmember_range"]
                    a1.plot([lo, hi], [y, y], color=c.sample.color, lw=4, alpha=0.3, solid_capstyle="butt")
                a1.errorbar(d["estimate"], y, xerr=z * (d["se_spatial"] or 0), fmt="o", color=c.sample.color,
                            ms=4, capsize=2, lw=1)
        a1.set_yticks(yt, yl)
        a1.invert_yaxis()
        a1.set_xlabel("Porosity")
        a1.grid(axis="y", visible=False)
        a1.legend(handles=[Line2D([], [], color=P.NEUTRAL, marker="o", ls="-", label="estimate ± 95 % (spatial)"),
                           Line2D([], [], color=P.NEUTRAL, lw=4, alpha=0.3, label="end-member range"),
                           Line2D([], [], color=P.NEUTRAL, marker="s", ls="-", label="lab ± 95 % (assumed sd)")],
                  loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=7)
        for i, c in enumerate(self.ctxs):
            ib = self.R["samples"][c.sample.id]["porosity_budget"]["implied_background"]
            lo, hi = ib["endmember_range"]
            a2.plot([lo, hi], [i, i], color=c.sample.color, lw=4, alpha=0.3, solid_capstyle="butt")
            a2.errorbar(ib["estimate"], i, xerr=z * (ib["se"] or 0), fmt="o", color=c.sample.color, ms=4, capsize=2)
        a2.axvline(0, color=P.INK_2, lw=0.7)
        a2.axvline(self.cfg["segmentation"]["background_porosity_scenario"], color=P.NEUTRAL, lw=0.8, ls="--")
        a2.text(self.cfg["segmentation"]["background_porosity_scenario"] + 0.004, 1.55, "5 % assumed\nby depositors",
                fontsize=7, ha="left", va="center", color=P.INK_2)
        a2.set_yticks(range(len(self.ctxs)), [c.sample.label for c in self.ctxs])
        a2.set_ylim(-0.6, 1.7)
        a2.invert_yaxis()
        a2.set_xlabel("Sub-resolution porosity\nof the solid phase")
        a2.grid(axis="y", visible=False)
        P.panel_labels([a1, a2])
        self.save(fig, "F03_porosity_budget")

    # -----------------------------------------------------------------------
    def f04_drift(self):
        fig, (a1, a2) = P.figure(self.W2, 60, 1, 2)
        for i, c in enumerate(self.ctxs):
            q = self.npz(c, "qc")
            pr = self.npz(c, "profiles")
            zmm = (np.arange(len(q["mode_raw"])) + 0.5) * c.voxel_mm
            ref = float(np.median(q["mode_smooth"]))
            a1.plot(zmm, q["mode_raw"] / ref, color=c.sample.color, lw=0.4, alpha=0.45)
            a1.plot(zmm, q["mode_smooth"] / ref, color=c.sample.color, ls=P.sample_style(i)["linestyle"], lw=1.3)
            a2.plot(zmm, pr["z_uncorrected_legacyT"], color=c.sample.color, lw=0.5, alpha=0.45)
            if c.drift_corrected:
                a2.plot(zmm, pr["z_legacy"], color=c.sample.color, lw=1.1, ls=P.sample_style(i)["linestyle"])
        a1.set_xlabel("Position along z (mm)")
        a1.set_ylabel("Solid-peak grey level / median")
        self.legend_samples(a1, self.ctxs, "upper left",
                            extra=[Line2D([], [], color=P.NEUTRAL, lw=0.5, label="per slice"),
                                   Line2D([], [], color=P.NEUTRAL, lw=1.3, label="running median")])
        a2.set_xlabel("Position along z (mm)")
        a2.set_ylabel("Slice porosity (mean − 1.1 sd)")
        a2.legend(handles=[Line2D([], [], color=P.NEUTRAL, lw=0.5, alpha=0.6, label="uncorrected"),
                           Line2D([], [], color=P.NEUTRAL, lw=1.1, label="drift-corrected (" + ", ".join(
                               c.sample.label.lower() for c in self.ctxs if c.drift_corrected) + ")")],
                  loc="upper left")
        P.panel_labels([a1, a2])
        self.save(fig, "F04_drift_qc")

    # -----------------------------------------------------------------------
    def f05_profiles(self):
        fig, axes = P.figure(self.W2, 100, 2, 3, sharey="row")
        slab = float(self.cfg["profiles"]["comparison_slab_mm"])
        for row, kind, ylab in ((0, "mixing", "Porosity (grey mass balance)"), (1, "binary", "Porosity (f = 0.5)")):
            for col, a in enumerate("xyz"):
                ax = axes[row, col]
                for i, c in enumerate(self.ctxs):
                    pr = self.npz(c, "profiles")
                    y = pr[f"{a}_{kind}"]
                    pos = pr[f"{a}_pos_mm"]
                    ax.plot(pos, y, color=c.sample.color, lw=0.35, alpha=0.35)
                    ns = max(1, int(round(slab / c.voxel_mm)))
                    k = len(y) // ns
                    ax.plot(pos[: k * ns].reshape(k, ns).mean(1), y[: k * ns].reshape(k, ns).mean(1),
                            color=c.sample.color, lw=1.2, ls=P.sample_style(i)["linestyle"])
                ax.set_xlabel(f"Position along {a} (mm)")
                if col == 0:
                    ax.set_ylabel(ylab)
        fig.legend(handles=[Line2D([], [], color=c.sample.color, lw=1.2, ls=P.sample_style(i)["linestyle"],
                                   label=c.sample.label) for i, c in enumerate(self.ctxs)] +
                   [Line2D([], [], color=P.NEUTRAL, lw=0.4, label="per slice"),
                    Line2D([], [], color=P.NEUTRAL, lw=1.2, label=f"{slab:g}-mm slabs")],
                   loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
        P.panel_labels(axes.ravel())
        self.save(fig, "F05_profiles")

    # -----------------------------------------------------------------------
    def f06_rev(self):
        fig, (a1, a2) = P.figure(self.W2, 66, 1, 2)
        for i, c in enumerate(self.ctxs):
            rv = self.R["samples"][c.sample.id]["rev"]
            for tag, ls, mk in (("legacy", "-", "o"), ("primary", "--", "s")):
                rows = rv[tag]["curve"]
                L = np.array([r["edge_mm"] for r in rows])
                cv = np.array([r["cv"] for r in rows])
                a1.loglog(L, cv, mk, color=c.sample.color, ms=3, mfc="white" if tag == "primary" else c.sample.color)
                f = rv[tag]["fit"]
                sel = (L >= f["fit_min_mm"] * (1 - 1e-9)) & (L <= f["fit_max_mm"] * (1 + 1e-9))
                if sel.sum() > 2:
                    V = L[sel] ** 3
                    cv_fit = np.sqrt(f["K"] * V ** -f["alpha"]) / rv[tag]["porosity_blocks"]
                    a1.loglog(L[sel], cv_fit, color=c.sample.color, ls=ls, lw=1)
                if tag == "legacy":
                    a2.fill_between(L, [r["p05"] for r in rows], [r["p95"] for r in rows], color=c.sample.color,
                                    alpha=0.15, lw=0)
                    a2.plot(L, [r["mean"] for r in rows], color=c.sample.color, ls=P.sample_style(i)["linestyle"])
                    a2.axhline(c.sample.lab_porosity, color=c.sample.color, lw=0.8, ls="--")
        # reference: finite integral range -> CV ~ L^-3/2
        xr = np.array([2.0, 20.0])
        a1.loglog(xr, 0.5 * (xr / 2.0) ** -1.5, color=P.INK_2, lw=0.8, ls=":")
        a1.text(2.6, 0.06, "dotted: slope −3/2\n(finite REV)", fontsize=7, color=P.INK_2, va="top")
        a1.set_xlabel("Cubic window edge (mm)")
        a1.set_ylabel("CV of window porosity")
        a1.legend(handles=[Line2D([], [], color=c.sample.color, label=c.sample.label) for c in self.ctxs] +
                  [Line2D([], [], color=P.NEUTRAL, marker="o", ls="-", label="mean − 1.1 sd"),
                   Line2D([], [], color=P.NEUTRAL, marker="s", mfc="white", ls="--", label="f = 0.5")],
                  loc="lower left")
        a2.set_xscale("log")
        a2.set_xlabel("Cubic window edge (mm)")
        a2.set_ylabel("Window porosity (mean − 1.1 sd)")
        a2.legend(handles=[Patch(color=P.NEUTRAL, alpha=0.3, label="5th–95th percentile"),
                           Line2D([], [], color=P.NEUTRAL, ls="--", label="laboratory")], loc="upper right")
        P.panel_labels([a1, a2])
        self.save(fig, "F06_rev")

    # -----------------------------------------------------------------------
    def f07_surface(self):
        fig, (a1, a2, a3) = P.figure(self.W2, 64, 1, 3)
        est = [("sv_crofton13", "Crofton (13 dir.)"), ("sv_crofton3", "Crofton (3 axes)"),
               ("sv_mc_grey", "MC, grey iso-surface"), ("sv_mc_binary", "MC, binary"), ("sv_faces", "voxel faces")]
        for i, c in enumerate(self.ctxs):
            cube = self.R["samples"][c.sample.id]["cube"]
            if not cube:
                continue
            s = cube["surface"]
            ys = np.arange(len(est)) + (i - 0.5) * 0.18
            a1.plot([s.get(k, np.nan) for k, _ in est], ys, P.MARKERS[i], color=c.sample.color, label=c.sample.label)
            sw = [w for w in cube["sweep"] if w["criterion"] == "f"]
            a2.plot([w["f"] for w in sw], [w["sv_crofton13"] for w in sw], marker=P.MARKERS[i], color=c.sample.color,
                    ls=P.sample_style(i)["linestyle"], ms=3.5)
            lg = [w for w in cube["sweep"] if w["criterion"] == "legacy"][0]
            a2.plot(self.R["samples"][c.sample.id]["segmentation"]["thresholds_by_key"]["legacy"]["f_equiv"],
                    lg["sv_crofton13"], "*", color=c.sample.color, ms=7, mec="white", mew=0.4)
        a1.set_yticks(range(len(est)), [n for _, n in est])
        a1.invert_yaxis()
        a1.set_xlabel(r"$S_V$ (mm$^{-1}$), central cube, f = 0.5")
        a1.grid(axis="y", visible=False)
        a1.legend(loc="upper right")
        a2.set_yscale("log")
        plain_log_axis(a2, "y", [0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3])
        a2.set_xlabel("Segmentation criterion f")
        a2.set_ylabel(r"$S_V$, Crofton (mm$^{-1}$)")
        a2.legend(handles=[Line2D([], [], marker="*", ls="", color=P.NEUTRAL, ms=7, label="mean − 1.1 sd")],
                  loc="upper right")
        res = self.R.get("resolution")
        if res:
            for i, c in enumerate(self.ctxs):
                rows = res["per_sample"][c.sample.id]["rows"]
                v = np.array([r["voxel_um"] for r in rows])
                sv = np.array([r["sv_crofton13"] for r in rows])
                a3.loglog(v, sv, marker=P.MARKERS[i], color=c.sample.color, ls=P.sample_style(i)["linestyle"], ms=3.5)
                a3.plot(v[0], sv[0], P.MARKERS[i], color=c.sample.color, ms=6)
                sc = res["per_sample"][c.sample.id]["scaling"].get("sv_crofton13")
                if sc:
                    a3.text(v[-1], sv[-1], f" slope\n {sc['slope']:.2f}", color=c.sample.color, fontsize=7,
                            va="center")
            a3.set_xlabel("Voxel size (µm)")
            a3.set_ylabel(r"$S_V$, Crofton (mm$^{-1}$)")
            plain_log_axis(a3, "x", [30, 45, 60, 90, 135, 180])
            plain_log_axis(a3, "y", [0.15, 0.2, 0.3, 0.4, 0.5, 0.7])
        P.panel_labels([a1, a2, a3])
        self.save(fig, "F07_specific_surface")

    # -----------------------------------------------------------------------
    def f08_boxcount(self):
        fig, axes = P.figure(self.W2, 120, 2, 2)
        (a1, a2), (a3, a4) = axes
        fcfg = self.cfg["fractal"]
        for i, c in enumerate(self.ctxs):
            if not self.R["samples"][c.sample.id]["cube"]:
                continue
            arr = self.npz(c, "cube")
            fd = self.R["samples"][c.sample.id]["fractal_diagnostics"]
            s, n = arr["box_size_mm"], arr["box_count"]
            col, st = c.sample.color, P.sample_style(i)
            a1.loglog(s, n, P.MARKERS[i], color=col, ms=3.5)
            fit = fd["common_range_fit"]
            xx = np.array([fit["s_min"], fit["s_max"]])
            a1.loglog(xx, np.exp(fit["intercept"]) * xx ** -fit["D"], color=col, ls=st["linestyle"], lw=1,
                      label=f"{c.sample.label}: D = {fit['D']:.2f}, R² = {fit['r2']:.3f}")
            a2.semilogx(s, arr["box_local_slope"], marker=P.MARKERS[i], color=col, ls=st["linestyle"], ms=3)
            xo = fd["crossover_mm_occ50"]
            for ax in (a2, a3):
                ax.axvline(xo, color=col, lw=0.8, ls=":")
            a3.semilogx(s, fd["occupancy"], marker=P.MARKERS[i], color=col, ls=st["linestyle"], ms=3)
            lg = fd["legacy"]
            a3.semilogx(lg["sizes_mm_metadata_voxel"][1:], lg["occupancy"][1:], P.MARKERS[i], mfc="white", mec=col,
                        ms=5, ls="")
            sw = [w for w in self.R["samples"][c.sample.id]["cube"]["sweep"] if w["criterion"] == "f"]
            dd = np.array([w["D"] for w in sw], float)
            lo = np.array([w.get("D_ci_low", np.nan) for w in sw], float)
            hi = np.array([w.get("D_ci_high", np.nan) for w in sw], float)
            a4.errorbar([w["f"] for w in sw], dd, yerr=[dd - lo, hi - dd],
                        marker=P.MARKERS[i], color=col, ls=st["linestyle"], ms=3.5, capsize=2)
            a4.axhline(lg["D_published_algorithm"], color=col, lw=0.8, ls="--")
        a1.axvspan(fcfg["fit_min_mm"], fcfg["fit_max_mm"], color=P.GRID, alpha=0.5, lw=0)
        a1.set_xlabel("Box edge s (mm)")
        a1.set_ylabel("Boxes containing interface N(s)")
        a1.legend(loc="lower left", fontsize=7)
        for yv in (2, 3):
            a2.axhline(yv, color=P.INK_2, lw=0.6, ls="--")
        a2.set_xlabel("Box edge s (mm)")
        a2.set_ylabel("Local slope −d ln N / d ln s")
        a2.legend(handles=[Line2D([], [], color=P.NEUTRAL, ls=":", label="occupancy = 0.5 (saturation onset)")],
                  loc="upper left", fontsize=7)
        a3.axhline(0.5, color=P.INK_2, lw=0.6, ls="--")
        a3.set_xlabel("Box edge s (mm)")
        a3.set_ylabel("Occupancy N(s) s³ / V")
        a3.legend(handles=[Line2D([], [], color=P.NEUTRAL, marker="o", label="this work (central cube)"),
                           Line2D([], [], color=P.NEUTRAL, marker="o", mfc="white", ls="",
                                  label="previous fit (porespy sizes)")], loc="lower right", fontsize=7)
        a4.set_xlabel("Segmentation criterion f")
        a4.set_ylabel("D (common range, 95 % fit interval)")
        a4.legend(handles=[Line2D([], [], color=P.NEUTRAL, ls="--", label="D of the earlier analysis")],
                  loc="lower left", fontsize=7)
        P.panel_labels([a1, a2, a3, a4])
        self.save(fig, "F08_box_counting")

    # -----------------------------------------------------------------------
    def f09_connectivity_psd(self):
        fig, (a1, a2, a3) = P.figure(self.W2, 62, 1, 3)
        for i, c in enumerate(self.ctxs):
            sid = c.sample.id
            col, st = c.sample.color, P.sample_style(i)
            if not self.R["samples"][sid]["cube"]:
                continue
            arr = self.npz(c, "cube")
            v = np.sort(arr["cluster_volumes_6"]) * c.voxel_mm**3
            ccdf = 1 - np.arange(len(v)) / len(v)
            a1.loglog(v, ccdf, color=col, ls=st["linestyle"])
            psd = (self.R["samples"][sid].get("psd") or {}).get("curves", {})
            for tag, ls in (("primary", st["linestyle"]), ("legacy", ":")):
                if tag in psd:
                    a2.semilogx(psd[tag]["radius_um"], psd[tag]["fraction_ge"], color=col, ls=ls, marker=P.MARKERS[i],
                                ms=2.5)
            a2.axvline(2 * c.voxel_um, color=col, lw=0.7, ls="--")
            sw = [w for w in self.R["samples"][sid]["cube"]["sweep"] if w["criterion"] == "f"]
            a3.plot([w["f"] for w in sw], [w["largest_cluster_fraction"] for w in sw], marker=P.MARKERS[i],
                    color=col, ls=st["linestyle"], ms=3.5)
            a3.plot([w["f"] for w in sw], [w["percolating_porosity_any"] / max(w["porosity"], 1e-12) for w in sw],
                    marker=P.MARKERS[i], color=col, ls=":", ms=3, mfc="white")
        a1.set_xlabel(r"Cluster volume (mm$^3$)")
        a1.set_ylabel("Fraction of clusters ≥ volume")
        self.legend_samples(a1, self.ctxs, "lower left")
        a2.set_xlabel("Pore radius r (µm)")
        a2.set_ylabel("Pore-volume fraction in pores ≥ r")
        a2.legend(handles=[Line2D([], [], color=P.NEUTRAL, label="f = 0.5"),
                           Line2D([], [], color=P.NEUTRAL, ls=":", label="mean − 1.1 sd"),
                           Line2D([], [], color=P.NEUTRAL, ls="--", lw=0.7, label="2 voxels")], loc="upper right",
                  fontsize=7)
        a3.set_xlabel("Segmentation criterion f")
        a3.set_ylabel("Fraction of pore volume")
        a3.legend(handles=[Line2D([], [], color=P.NEUTRAL, marker="o", label="largest cluster"),
                           Line2D([], [], color=P.NEUTRAL, marker="o", mfc="white", ls=":",
                                  label="percolating clusters")], loc="upper right", fontsize=7)
        P.panel_labels([a1, a2, a3])
        self.save(fig, "F09_connectivity_psd")

    # -----------------------------------------------------------------------
    def f10_correlation(self):
        fig, ax = P.figure(self.W1 - 2, 62)
        for i, c in enumerate(self.ctxs):
            if not self.R["samples"][c.sample.id]["cube"]:
                continue
            arr = self.npz(c, "cube")
            for a in "xyz":
                r = arr[f"s2_{a}_lag_mm"]
                sel = r <= 2.0
                ax.plot(r[sel], arr[f"c_{a}"][sel], color=c.sample.color, ls=P.LINESTYLES[a], lw=1)
        ax.axhline(np.exp(-1), color=P.INK_2, lw=0.6, ls="--")
        ax.axhline(0, color=P.INK_2, lw=0.5)
        ax.set_xlabel("Lag r (mm)")
        ax.set_ylabel("Normalised autocovariance C(r)")
        ax.legend(handles=[Line2D([], [], color=c.sample.color, label=c.sample.label) for c in self.ctxs] +
                  [Line2D([], [], color=P.NEUTRAL, ls=P.LINESTYLES[a], label=f"along {a}") for a in "xyz"],
                  loc="upper right")
        self.save(fig, "F10_two_point_correlation")

    # -----------------------------------------------------------------------
    def f11_esf(self):
        fig, (a1, a2) = P.figure(self.W2, 62, 1, 2)
        data = []
        for i, c in enumerate(self.ctxs):
            p = c.data("esf", "npz")
            if not p.exists():
                continue
            e = self.npz(c, "esf")
            x = e["grid_vox"] * c.voxel_um
            a1.fill_between(x, e["q25"], e["q75"], color=c.sample.color, alpha=0.18, lw=0)
            a1.plot(x, e["mean"], color=c.sample.color, ls=P.sample_style(i)["linestyle"], label=c.sample.label)
            data.append((c, e["sigma_kept"] * c.voxel_um))
        a1.set_xlabel("Distance across the pore wall (µm)")
        a1.set_ylabel("Normalised grey level")
        a1.legend(loc="lower right")
        for i, (c, sig) in enumerate(data):
            bp = a2.boxplot(sig, positions=[i], widths=0.45, showfliers=False, patch_artist=True)
            for b in bp["boxes"]:
                b.set(facecolor=c.sample.color, alpha=0.35, edgecolor=c.sample.color)
            for k in ("whiskers", "caps", "medians"):
                for l in bp[k]:
                    l.set(color=c.sample.color)
            a2.plot([i - 0.3, i + 0.3], [c.voxel_um] * 2, color=P.INK, lw=1, ls="--")
        a2.set_xticks(range(len(data)), [c.sample.label for c, _ in data])
        a2.set_ylabel("Edge-spread σ (µm)")
        a2.legend(handles=[Line2D([], [], color=P.INK, ls="--", label="voxel size")], loc="upper left")
        a2.grid(axis="x", visible=False)
        P.panel_labels([a1, a2])
        self.save(fig, "F11_effective_resolution")

    # -----------------------------------------------------------------------
    def f12_replicates(self):
        p = self.cfg.data_dir / "replicates_all.csv"
        if not p.exists():
            return
        df = pd.read_csv(p)
        metrics = [("porosity_mixing", "Porosity (mass balance)"), ("porosity", "Porosity (f = 0.5)"),
                   ("sv_crofton13", r"$S_V$, Crofton (mm$^{-1}$)"), ("mean_chord_mm", "Mean pore chord (mm)"),
                   ("D_unsat", "D (unsaturated range)"), ("crossover_mm", "Crossover length (mm)"),
                   ("largest_cluster_fraction", "Largest-cluster fraction")]
        fig, axes = P.figure(self.W2, 108, 2, 4)
        axes = axes.ravel()
        rng = np.random.default_rng(0)
        for k, (m, lab) in enumerate(metrics):
            ax = axes[k]
            if m not in df:
                ax.set_visible(False)
                continue
            for i, c in enumerate(self.ctxs):
                y = df.loc[df["sample"] == c.sample.id, m].dropna().to_numpy()
                if not len(y):
                    continue
                bp = ax.boxplot(y, positions=[i], widths=0.5, showfliers=False, patch_artist=True)
                for b in bp["boxes"]:
                    b.set(facecolor="none", edgecolor=c.sample.color)
                for kk in ("whiskers", "caps", "medians"):
                    for l in bp[kk]:
                        l.set(color=c.sample.color)
                ax.plot(i + rng.uniform(-0.18, 0.18, len(y)), y, ".", color=c.sample.color, ms=1.8, alpha=0.5)
            ax.set_xticks(range(len(self.ctxs)), [c.sample.label for c in self.ctxs])
            ax.set_ylabel(lab)
            ax.grid(axis="x", visible=False)
        ax = axes[7]
        for i, c in enumerate(self.ctxs):
            d = df[df["sample"] == c.sample.id]
            ax.plot(d["porosity"], d["sv_crofton13"], P.MARKERS[i], color=c.sample.color, ms=2, alpha=0.5)
        ax.set_xlabel("Porosity (f = 0.5)")
        ax.set_ylabel(r"$S_V$, Crofton (mm$^{-1}$)")
        P.panel_labels(axes)
        self.save(fig, "F12_replicates")

    # -----------------------------------------------------------------------
    def f13_resolution(self):
        res = self.R.get("resolution")
        if not res:
            return
        fig, axes = P.figure(self.W2, 110, 2, 2)
        (a1, a2), (a3, a4) = axes
        for i, c in enumerate(self.ctxs):
            rows = res["per_sample"][c.sample.id]["rows"]
            v = np.array([r["voxel_um"] for r in rows])
            col, st = c.sample.color, P.sample_style(i)
            kw = dict(marker=P.MARKERS[i], color=col, ls=st["linestyle"], ms=3.5)
            a1.plot(v, [r["porosity"] for r in rows], **kw)
            a1.plot(v, [r["porosity_mixing"] for r in rows], color=col, ls=":", lw=0.9)
            a2.loglog(v, [r["sv_crofton13"] for r in rows], **kw)
            a3.plot(v, [r.get("D_unsat") for r in rows], **kw)
            a4.plot(v, [r["largest_cluster_fraction"] for r in rows], **kw)
            for ax in (a1, a2, a3, a4):
                ax.axvline(c.voxel_um, color=col, lw=0.6, ls="--", alpha=0.7)
        a1.set_ylabel("Porosity (binary f = 0.5; dotted: mass balance)")
        fig.legend(handles=[Line2D([], [], color=c.sample.color, ls=P.sample_style(i)["linestyle"],
                                   marker=P.MARKERS[i], label=c.sample.label) for i, c in enumerate(self.ctxs)] +
                   [Line2D([], [], color=P.NEUTRAL, ls=":", label="grey mass balance (a)"),
                    Line2D([], [], color=P.NEUTRAL, ls="--", lw=0.6, label="native voxel size")],
                   loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
        a2.set_ylabel(r"$S_V$, Crofton (mm$^{-1}$)")
        plain_log_axis(a2, "x", [30, 45, 60, 90, 135, 180])
        plain_log_axis(a2, "y", [0.15, 0.2, 0.3, 0.4, 0.5, 0.7])
        a3.set_ylabel("D (unsaturated range)")
        a4.set_ylabel("Largest-cluster fraction")
        for ax in (a1, a2, a3, a4):
            ax.set_xlabel("Voxel size (µm)")
        P.panel_labels([a1, a2, a3, a4])
        self.save(fig, "F13_resolution_experiment")


def step_figures(cfg: Config):
    import json

    F = Fig(cfg)
    failed = []
    for name in [m for m in dir(F) if m[:1] == "f" and m[1:3].isdigit()]:
        key = "F" + name[1:3]
        for old in cfg.fig_dir.glob(f"{key}_*.*"):          # no stale or renamed leftovers
            old.unlink()
        try:
            getattr(F, name)()
        except Exception as exc:  # noqa: BLE001 - keep producing the other figures
            log.exception("figure %s failed: %s", name, exc)
            failed.append(key)
    for old in cfg.fig_dir.glob("GA_*.*"):
        old.unlink()
    try:
        from .graphical_abstract import graphical_abstract

        graphical_abstract(cfg)
    except Exception as exc:  # noqa: BLE001
        log.exception("graphical abstract failed: %s", exc)
        failed.append("GA")
    (cfg.fig_dir / "FAILED.json").write_text(json.dumps(failed), encoding="utf-8")
