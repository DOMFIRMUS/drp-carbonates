"""Publication figure style and helpers.

Conventions (journal production rules and accessibility):
* no titles inside figures - the caption carries them;
* panel letters (a), (b), ... at the top-left of each panel;
* sizes in journal column widths, text 7-8 pt, thin marks, recessive grid;
* one colour per sample everywhere (colour follows the specimen), with
  line style / marker as secondary encoding so figures survive greyscale and
  colour-vision deficiency; blue/orange are the first two slots of a
  palette validated for all pairs under CVD simulation;
* every figure is written as vector PDF plus 600-dpi PNG.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MM = 1 / 25.4
INK = "#1a1a1a"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8983"
AXIS_COLORS = {"x": "#1baf7a", "y": "#4a3aa7", "z": "#e34948"}  # used only when samples are not coloured
LINESTYLES = {"x": "-", "y": "--", "z": ":"}
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def apply_style(font: str = "Arial", size: float = 8.0):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font, "DejaVu Sans"],
        "font.size": size,
        "axes.titlesize": size,
        "axes.labelsize": size,
        "xtick.labelsize": size - 1,
        "ytick.labelsize": size - 1,
        "legend.fontsize": size - 1,
        "legend.frameon": False,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "lines.linewidth": 1.1,
        "lines.markersize": 4,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.default": "regular",
    })


def figure(width_mm: float, height_mm: float, nrows=1, ncols=1, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width_mm * MM, height_mm * MM),
                             constrained_layout=True, **kw)
    return fig, axes


def panel_labels(axes, start: str = "a", x: float = -0.02, y: float = 1.02, images: bool = False):
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.text(x if not images else 0.0, y, f"({chr(ord(start) + i)})", transform=ax.transAxes,
                ha="right" if not images else "left", va="bottom", fontweight="bold")


def scale_bar(ax, length: float, label: str, color: str = "white", loc: str = "lower right",
              pad_frac: float = 0.05, thickness_frac: float = 0.018):
    """Draw a scale bar on an image axis drawn with imshow(origin='upper').

    `length` is in data units of the axis (pixels, or mm when the image was
    drawn with a physical `extent`). The text gets a dark halo so it stays
    legible on bright and dark backgrounds.
    """
    import matplotlib.patheffects as pe

    x0, x1 = sorted(ax.get_xlim())
    y_top, y_bot = sorted(ax.get_ylim())
    w, h = x1 - x0, y_bot - y_top
    t = thickness_frac * h
    px = x1 - pad_frac * w - length if "right" in loc else x0 + pad_frac * w
    py = y_bot - pad_frac * h - t  # rectangle spans [py, py + t]; y grows downwards on images
    halo = [pe.withStroke(linewidth=1.6, foreground="black")]
    bar = matplotlib.patches.Rectangle((px, py), length, t, facecolor=color, edgecolor="black", lw=0.5)
    ax.add_patch(bar)
    ax.text(px + length / 2, py - 0.012 * h, label, color=color, ha="center", va="bottom",
            fontsize=plt.rcParams["font.size"] - 0.5, fontweight="bold", path_effects=halo)


MAX_WIDTH_MM = 174.0     # Springer double-column width
MAX_HEIGHT_MM = 234.0


def save(fig, path_stem: Path, formats=("png", "pdf"), dpi: int = 600) -> tuple[float, float]:
    """Save in every format and return the final (tight) size in mm.

    The size is checked against the journal artwork limits; a figure that
    grows beyond them (e.g. a label sticking out of a panel) is reported.
    """
    import logging

    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(path_stem.with_suffix("." + fmt), dpi=dpi)
    plt.close(fig)
    w = h = float("nan")
    png = path_stem.with_suffix(".png")
    if png.exists():
        from PIL import Image

        with Image.open(png) as im:
            w, h = im.size[0] / dpi * 25.4, im.size[1] / dpi * 25.4
        if w > MAX_WIDTH_MM + 0.5 or h > MAX_HEIGHT_MM + 0.5:
            logging.getLogger("drp").warning("figure %s is %.1f x %.1f mm, beyond %.0f x %.0f mm", path_stem.name,
                                             w, h, MAX_WIDTH_MM, MAX_HEIGHT_MM)
    return w, h


def sample_style(i: int) -> dict:
    """Secondary encoding per sample index (line style + marker)."""
    return {"linestyle": ["-", "--", "-.", ":"][i % 4], "marker": MARKERS[i % len(MARKERS)]}
