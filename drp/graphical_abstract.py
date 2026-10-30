"""Graphical abstract (one landscape figure), built from saved results only.

Four messages, one per panel, each drawn from the same numbers as the
article: support (no REV within 20 mm), segmentation (which rankings survive
the criterion), resolution (porosity below the voxel) and box counting
(saturated boxes behind the dimension of the earlier analysis). Sample colours follow the
article figures; text stays in ink colours.
"""
from __future__ import annotations

import logging

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

from . import plotting as P
from .config import Config
from .context import SampleContext
from .util import load_json
from .volume import central_cube

log = logging.getLogger("drp")

W_MM, H_MM = 174.0, 96.0
ACCENT_BG = "#f3f2ef"


def _panel(fig, rect, title, takeaway):
    """Rounded card with a title, a plot axis and a one-line takeaway."""
    x, y, w, h = rect
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, facecolor=ACCENT_BG, edgecolor="none", zorder=-5))
    fig.text(x + 0.012, y + h - 0.045, title, fontsize=8, fontweight="bold", color=P.INK, va="bottom")
    fig.text(x + 0.012, y + 0.018, takeaway, fontsize=7, color=P.INK_2, va="bottom", linespacing=1.15)
    ax = fig.add_axes([x + 0.075, y + 0.155, w - 0.10, h - 0.255])
    ax.set_facecolor("white")
    ax.set_zorder(2)
    return ax


def _label(ax, x, y, text, color, ls="-", marker=None, fontsize=7):
    """Direct label in ink with a short coloured glyph in front of it (axes-fraction position)."""
    ax.add_line(Line2D([x, x + 0.03, x + 0.06], [y, y, y], transform=ax.transAxes, color=color, ls=ls, lw=1.4,
                       marker=marker, ms=3.5, markevery=[1] if marker else None, clip_on=False))
    ax.text(x + 0.085, y, text, transform=ax.transAxes, fontsize=fontsize, color=P.INK, va="center", ha="left")


def graphical_abstract(cfg: Config):
    R = load_json(cfg.data_dir / "results.json")
    ctxs = [SampleContext(cfg, s) for s in cfg.samples]
    S = R["samples"]
    A, B = ctxs[0].sample.id, ctxs[1].sample.id
    col = {c.sample.id: c.sample.color for c in ctxs}
    lab = {c.sample.id: c.sample.label for c in ctxs}
    P.apply_style(cfg["figures"]["font"], 8)
    fig = P.plt.figure(figsize=(W_MM * P.MM, H_MM * P.MM))

    # ----------------------------------------------------------------- header
    fig.text(0.012, 0.955, "What micro-CT measures in carbonate rocks depends on volume, resolution and segmentation",
             fontsize=9.5, fontweight="bold", color=P.INK, va="center")
    fig.text(0.012, 0.912, "Indiana Limestone and Silurian Dolomite (public micro-CT stacks) re-analysed with an "
             "open, tested pipeline; Monte Carlo nulls and sensitivity analyses", fontsize=7, color=P.INK_2,
             va="center")

    # ----------------------------------------------------------------- images
    img_x, img_w = 0.012, 0.165
    for i, c in enumerate(ctxs):
        stack = c.stack()
        box = central_cube(stack.shape, c.vox(float(cfg["volumes"]["central_cube_mm"])))
        z = (box.z0 + box.z1) // 2
        g = stack.grey(z, z + 1)[0].astype(np.float32)
        half = int(round(4.0 / c.voxel_mm))
        yc, xc = g.shape[0] // 2, g.shape[1] // 2
        sub = g[yc - half: yc + half, xc - half: xc + half]
        lo, hi = np.percentile(sub, [0.5, 99.5])
        y0 = 0.475 - i * 0.415
        ax = fig.add_axes([img_x, y0, img_w, 0.33])
        ax.imshow(sub, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ov = np.zeros(sub.shape + (4,))
        rgb = tuple(int(c.sample.color[k:k + 2], 16) / 255 for k in (1, 3, 5))
        ov[sub < c.T_primary] = (*rgb, 0.9)
        ax.imshow(ov, interpolation="nearest")
        ax.set_axis_off()
        n = sub.shape[0]
        P.scale_bar(ax, 2.0 / c.voxel_mm, "2 mm")
        ax.add_patch(Rectangle((-0.5, -0.5), n, n, fill=False, ec=c.sample.color, lw=1.6))
        lab_phi = S[c.sample.id]["acquisition"]["lab_porosity"]
        ax.text(0.0, 1.09, c.sample.long_label.split(" (")[0], transform=ax.transAxes, fontsize=7, color=P.INK,
                fontweight="bold", va="bottom")
        ax.text(0.0, 1.015, f"voxel {c.voxel_um:.2f} µm · lab porosity {100 * lab_phi:.1f} %",
                transform=ax.transAxes, fontsize=6.5, color=P.INK_2, va="bottom")
    fig.text(img_x, 0.035, "coloured: pore voxels at f = 0.5", fontsize=6.6, color=P.INK_2)

    # ----------------------------------------------------------------- panels
    x0, gap = 0.195, 0.012
    pw = (1 - x0 - 0.008 - gap) / 2
    ph = 0.405
    rects = [(x0, 0.465, pw, ph), (x0 + pw + gap, 0.465, pw, ph), (x0, 0.045, pw, ph), (x0 + pw + gap, 0.045, pw, ph)]

    # (1) support --------------------------------------------------------
    al = {sid: S[sid]["rev"]["primary"]["fit"]["alpha"] for sid in (A, B)}
    ald = {sid: S[sid]["sensitivity"]["z_detrended"]["primary"]["alpha"] for sid in (A, B)}
    band = [min(S[s]["rev"]["primary"]["fit"]["null"]["null_q025"] for s in (A, B)),
            max(S[s]["rev"]["primary"]["fit"]["null"]["null_q975"] for s in (A, B))]
    ax = _panel(fig, rects[0], "1  Volume (support)",
                f"No REV within 20 mm: α = {min(al.values()):.2f}–{max(al.values()):.2f} "
                f"({min(ald.values()):.2f}–{max(ald.values()):.2f} detrended),\n"
                f"below stationary fields, 1-mm kernel ({band[0]:.2f}–{band[1]:.2f})")
    for sid in (A, B):
        rv = S[sid]["rev"]["primary"]
        L = np.array([r["edge_mm"] for r in rv["curve"]])
        cv = np.array([r["cv"] for r in rv["curve"]])
        ax.loglog(L, cv, "o", color=col[sid], ms=2.8, mfc="white", mew=0.9)
        f = rv["fit"]
        sel = (L >= f["fit_min_mm"] * 0.999) & (L <= f["fit_max_mm"] * 1.001)
        ax.loglog(L[sel], np.sqrt(f["K"] * (L[sel] ** 3) ** -f["alpha"]) / rv["porosity_blocks"], color=col[sid], lw=1.6)
    xr = np.array([2.0, 20.0])
    y_ref = np.sqrt(S[B]["rev"]["primary"]["fit"]["K"] * 8.0 ** -S[B]["rev"]["primary"]["fit"]["alpha"]) / \
        S[B]["rev"]["primary"]["porosity_blocks"]
    ax.loglog(xr, y_ref * (xr / 2.0) ** -1.5, color=P.NEUTRAL, lw=1.0, ls=":")
    ax.set_xlabel("Window edge (mm)", fontsize=7, labelpad=1)
    ax.set_ylabel("CV of porosity", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=6.5)
    _label(ax, 0.03, 0.27, f"{lab[A]}  α = {al[A]:.2f}", col[A])
    _label(ax, 0.03, 0.09, f"{lab[B]}  α = {al[B]:.2f}", col[B])
    _label(ax, 0.55, 0.90, "finite REV (α = 1)", P.NEUTRAL, ls=":")

    # (2) segmentation ---------------------------------------------------
    sr = [r for r in R["sweep_ratios"] if r["matched"]]
    fx = [r["f"] for r in sr]
    sv = [r[f"sv_crofton13_ratio_{A}_over_{B}"] for r in sr]
    spv = [r[f"s_per_vp_ratio_{A}_over_{B}"] for r in sr]
    ax = _panel(fig, rects[1], "2  Segmentation criterion",
                f"The $S_V$ ranking flips with the criterion ({min(sv):.2f}–{max(sv):.2f});\n"
                f"surface per pore volume keeps it ({min(spv):.2f}–{max(spv):.2f})")
    ax.axhline(1.0, color=P.INK_2, lw=0.8)
    ax.plot(fx, sv, color=P.INK, marker="o", ms=3.5, lw=1.5)
    ax.plot(fx, spv, color=P.NEUTRAL, marker="s", ms=3.5, lw=1.5, ls="--")
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0.5, 6.0)
    ax.set_xlabel("Pore if partial-volume fraction > f", fontsize=7, labelpad=1)
    ax.set_ylabel(f"{lab[A]} / {lab[B]}", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=6.5)
    _label(ax, 0.03, 0.93, "surface per pore volume $S/V_p$", P.NEUTRAL, ls="--", marker="s")
    _label(ax, 0.03, 0.76, "specific surface $S_V$", P.INK, marker="o")
    ax.text(0.985, 0.335, f"↑ {lab[A]} higher", transform=ax.transAxes, fontsize=6.5, color=P.INK_2, ha="right",
            va="bottom")
    ax.text(0.985, 0.04, f"↓ {lab[B]} higher", transform=ax.transAxes, fontsize=6.5, color=P.INK_2, ha="right",
            va="bottom")

    # (3) resolution: porosity below the voxel ------------------------------
    ax = _panel(fig, rects[2], "3  Resolution",
                "Threshold-free mass balance vs laboratory porosity:\n"
                "very different amounts lie below the voxel")
    ys = [1, 0]
    for yy, sid in zip(ys, (A, B)):
        b = S[sid]["porosity_budget"]
        seen, labv = 100 * b["mixing"]["estimate"], 100 * b["lab"]
        ax.barh(yy, seen, height=0.46, color=col[sid], edgecolor="white", lw=1.0)
        ax.barh(yy, max(labv - seen, 0), left=seen, height=0.46, color="white", edgecolor=col[sid], lw=1.0,
                hatch="////")
        ax.text(labv + 0.4, yy, f"unseen {labv - seen:.1f} %", fontsize=6.5, color=P.INK, va="center", ha="left")
    ax.set_yticks(ys, [lab[A], lab[B]], fontsize=7)
    ax.set_xlim(0, 22)
    ax.set_ylim(-0.55, 2.05)
    ax.set_xlabel("Porosity (%)", fontsize=7, labelpad=1)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=P.NEUTRAL, label="seen (mass balance)"),
                       Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=P.NEUTRAL, hatch="////",
                                 label="laboratory − image")], loc="upper left", ncol=2, fontsize=6.5,
              handlelength=1.4, borderaxespad=0.2, columnspacing=1.0)

    # (4) box counting ----------------------------------------------------
    pubs = [S[s]["fractal_diagnostics"]["legacy"]["D_published_algorithm"] for s in (A, B)]
    ax = _panel(fig, rects[3], "4  Box counting",
                f"Earlier D = {min(pubs):.2f}–{max(pubs):.2f} was fitted on saturated boxes;\n"
                "below saturation the sizes span under half a decade")
    xo = []
    for sid, c in zip((A, B), ctxs):
        with np.load(c.data("cube", "npz")) as zf:
            s_mm, ls = zf["box_size_mm"], zf["box_local_slope"]
        ax.semilogx(s_mm, ls, color=col[sid], lw=1.5, marker="o", ms=2.6)
        xo.append(S[sid]["fractal_diagnostics"]["crossover_mm_occ50"])
    ax.axvspan(min(xo), 6.0, color="#e9e7e2", lw=0, zorder=0)
    ax.text(5.6, 1.3, "saturated: occupancy > 0.5", fontsize=6.5, color=P.INK_2, va="bottom", ha="right")
    lg = S[A]["fractal_diagnostics"]["legacy"]["sizes_mm_metadata_voxel"][1:]
    ax.annotate("", xy=(min(max(lg), 5.5), 3.14), xytext=(min(lg), 3.14),
                arrowprops=dict(arrowstyle="|-|", color=P.INK, lw=0.8, shrinkA=0, shrinkB=0))
    ax.text(5.2, 3.2, "earlier fit", fontsize=6.5, color=P.INK, ha="right", va="bottom")
    for yv in (2, 3):
        ax.axhline(yv, color=P.NEUTRAL, lw=0.6, ls="--")
    ax.set_ylim(1.2, 3.85)
    ax.set_xlim(0.05, 6.0)
    ax.set_xlabel("Box edge (mm)", fontsize=7, labelpad=1)
    ax.set_ylabel("Local slope", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=6.5)
    from matplotlib.ticker import FixedLocator as FL, FuncFormatter as FF

    ax.xaxis.set_major_locator(FL([0.1, 0.3, 1, 3]))
    ax.xaxis.set_major_formatter(FF(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    _label(ax, 0.03, 0.58, lab[A], col[A], marker="o")
    _label(ax, 0.03, 0.43, lab[B], col[B], marker="o")

    # ----------------------------------------------------------------- footer
    n_tests = 39                      # tests/test_estimators.py (pytest count)
    fig.text(0.992, 0.012, f"open Python pipeline · {n_tests} tests on synthetic volumes · code and derived data on Zenodo",
             fontsize=6.5, color=P.INK_2, ha="right", va="bottom")
    P.save(fig, cfg.fig_dir / "GA_graphical_abstract", ("png", "pdf"), 600)
    log.info("graphical abstract written")


def step_graphical_abstract(cfg: Config):
    graphical_abstract(cfg)
