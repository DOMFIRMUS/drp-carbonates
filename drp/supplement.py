"""Supplementary material (DOCX): estimator validation, supplementary figures
and supplementary tables, assembled from pipeline outputs."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import fractal, rev, sharpness, surface
from .captions import ORDER, captions
from .config import Config
from .util import load_json
from .volume import ArraySource

log = logging.getLogger("drp")

MAIN_TABLES = ["T01_acquisition", "T05_porosity_budget", "T08_rev", "T09_morphology"]
# order of first citation in the manuscript (Table S1 is the synthetic validation, written separately)
SUPP_TABLES = ["T02_legacy_replication", "T04_endmembers", "T10_segmentation_sweep", "T03_segmentation",
               "T16_endmember_sensitivity", "T14_drift_sensitivity", "T15_rev_sensitivity", "T13_resolution",
               "T06_profiles", "T07_profile_comparison", "T11_replicates", "T12_ancova"]


def menger_sponge(level: int) -> np.ndarray:
    """Menger sponge (True = solid) of edge 3**level voxels; D = ln 20 / ln 3."""
    m = np.ones((1, 1, 1), bool)
    for _ in range(level):
        n = m.shape[0]
        new = np.zeros((3 * n,) * 3, bool)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if (i == 1) + (j == 1) + (k == 1) < 2:
                        new[i * n:(i + 1) * n, j * n:(j + 1) * n, k * n:(k + 1) * n] = m
        m = new
    return m


def boolean_spheres(n: int, r: int, phi: float, seed: int = 0) -> np.ndarray:
    """Boolean model of equal spheres (True = pore): smooth interfaces, no fractal scaling."""
    rng = np.random.default_rng(seed)
    npts = int(-np.log(1 - phi) * n**3 / (4 / 3 * np.pi * r**3))
    b = np.zeros((n, n, n), bool)
    zz, yy, xx = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    ball = zz**2 + yy**2 + xx**2 <= r * r
    for c in rng.integers(0, n, (npts, 3)):
        lo = c - r
        sl = tuple(slice(max(0, a), min(n, a + 2 * r + 1)) for a in lo)
        bs = tuple(slice(sl[i].start - a, sl[i].stop - a) for i, a in enumerate(lo))
        b[sl] |= ball[bs]
    return b


def validation_table() -> pd.DataFrame:
    """Estimators on synthetic volumes with known answers (related cases are in the tests)."""
    rows = []
    for r in (12.3, 25.7):
        n = 80
        z, y, x = np.ogrid[:n, :n, :n]
        c = (n - 1) / 2
        m = (z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2 <= r**2
        area = 4 * np.pi * r**2
        true = area / n**3
        s = surface.specific_surface(ArraySource(m), 1.0, slab=16)
        # Intercept counting normalises by the length of test line actually
        # sampled; for an isolated object inside the domain the reference
        # volume is therefore the one covered by the voxel pairs of each
        # direction (in a space-filling medium the two coincide).
        cnt = surface.transition_counts(ArraySource(m), slab=16)
        ref13 = area * np.sum(surface.crofton_weights_13() / cnt["pairs"])
        ref3 = area * np.mean(1.0 / cnt["pairs"][:3])
        for k, name, ref in (("sv_crofton13", "Crofton, 13 directions", ref13), ("sv_crofton3", "Crofton, 3 axes", ref3),
                             ("sv_mc_binary", "Marching cubes, binary", true), ("sv_faces", "Voxel faces", true)):
            rows.append({"Estimator": name, "Test object": f"digital sphere, r = {r} voxels",
                         "Known value": ref, "Estimate": s[k], "Relative error": s[k] / ref - 1})
    n, r = 64, 20.4
    z, y, x = np.ogrid[:n, :n, :n]
    dist = np.sqrt((z - 31.5) ** 2 + (y - 31.5) ** 2 + (x - 31.5) ** 2)
    grey = np.clip(1000 + 100 * (dist - r), 0, 3000)
    a = surface.mc_area(ArraySource(grey), 1000.0, 1.0)
    rows.append({"Estimator": "Marching cubes, grey iso-surface", "Test object": f"smooth sphere, r = {r}",
                 "Known value": 4 * np.pi * r**2, "Estimate": a, "Relative error": a / (4 * np.pi * r**2) - 1})
    xx = np.arange(128)
    planes = np.broadcast_to(((xx + 7) % 64) < 32, (128, 128, 128)).copy()
    sizes = np.array([2, 3, 4, 6, 8, 12, 16])
    d_pl = fractal.fit_dimension(sizes, fractal.box_count(ArraySource(planes), sizes, 6)["count"])["D"]
    d_pl_old = fractal.fit_dimension(sizes, fractal.box_count(ArraySource(planes), sizes, 6, mode="mixed")["count"])["D"]
    rng = np.random.default_rng(0)
    noise = rng.random((96, 96, 96)) < 0.5
    s2 = np.array([2, 3, 4, 6, 8])
    d_no = fractal.fit_dimension(s2, fractal.box_count(ArraySource(noise), s2, 2)["count"])["D"]
    rows += [
        {"Estimator": "Box counting, boundary voxels", "Test object": "parallel planes", "Known value": 2.0,
         "Estimate": d_pl, "Relative error": d_pl / 2 - 1},
        {"Estimator": "Box counting, porespy definition (both phases)", "Test object": "parallel planes",
         "Known value": 2.0, "Estimate": d_pl_old, "Relative error": d_pl_old / 2 - 1},
        {"Estimator": "Box counting, boundary voxels", "Test object": "white noise, φ = 0.5", "Known value": 3.0,
         "Estimate": d_no, "Relative error": d_no / 3 - 1},
    ]
    counts = rng.binomial(8, 0.2, size=(60, 60, 60)).astype(np.int32)
    fit = rev.integral_range(rev.variance_curve(counts, 2, 1.0, [4, 6, 8, 12, 16, 24], 3000, seed=1), 0.2, 4,
                             0.05, 120.0**3)
    rows += [{"Estimator": "Variance–volume exponent α", "Test object": "white noise, φ = 0.2", "Known value": 1.0,
              "Estimate": fit["alpha"], "Relative error": fit["alpha"] - 1},
             {"Estimator": "Integral range A3 (voxel³)", "Test object": "white noise, φ = 0.2", "Known value": 1.0,
              "Estimate": fit["A3_mm3_if_alpha1"], "Relative error": fit["A3_mm3_if_alpha1"] - 1}]
    # alpha of a correlated stationary medium (finite integral range) on a stack-like grid
    al, afd = [], []
    for k in range(300):
        g = rev.gaussian_null_field((150, 70, 70), 1.0 / 0.18, np.random.default_rng(1000 + k))
        c = (g < np.quantile(g, 0.08)).astype(np.int32) * 64
        rr = rev.variance_curve(c, 4, 0.045, [2, 3, 4, 5, 6, 8, 10, 12], 2000, seed=k)
        f = rev.integral_range(rr, 0.08, 2.0, 0.05, float(np.prod(c.shape)) * 0.18**3)
        al.append(f["alpha"])
        afd.append(f["alpha_finite_domain"])
    for name, v in (("straight-line fit", np.asarray(al)), ("finite-domain fit", np.asarray(afd))):
        rows.append({"Estimator": f"Variance–volume exponent α, {name} (300 fields: mean ± SD [2.5–97.5 %])",
                     "Test object": "thresholded Gaussian field, 1-mm correlation, 27×13×13 mm", "Known value": 1.0,
                     "Estimate": f"{v.mean():.3f} ± {v.std(ddof=1):.3f} [{np.quantile(v, 0.025):.2f}–"
                                 f"{np.quantile(v, 0.975):.2f}]",
                     "Relative error": f"{v.mean() - 1:+.3f} (SE {v.std(ddof=1) / np.sqrt(len(v)):.3f})"})
    # a true fractal and a smooth non-fractal medium through the same box counting
    pore = ~menger_sponge(5)
    s3 = np.array([3, 9, 27, 81])
    d_m = fractal.fit_dimension(s3, fractal.box_count(ArraySource(pore), s3, 1)["count"])["D"]
    rows.append({"Estimator": "Box counting, boundary voxels (aligned boxes 3–81 voxels)",
                 "Test object": "Menger sponge, level 5 (243³)",
                 "Known value": float(np.log(20) / np.log(3)), "Estimate": d_m,
                 "Relative error": d_m / (np.log(20) / np.log(3)) - 1})
    # the same sponge through the production path (geometric sizes, 4 offsets,
    # unsaturated plateau): every box size is saturated, so the occupancy rule
    # cannot resolve a dimension this high
    s5 = np.unique(np.rint(np.geomspace(2, 60, 10)).astype(int))
    bc5 = fractal.box_count(ArraySource(pore), s5, 4)
    occ5 = fractal.occupancy(s5, bc5["count"], float(pore.size))
    pl5 = fractal.plateau(s5.astype(float), bc5["count"], float(pore.size), 2, 0.5, 0.05)
    rows.append({"Estimator": "Box counting, production path: sizes with occupancy ≤ 0.5",
                 "Test object": "Menger sponge, level 5 (243³)", "Known value": "D = 2.727",
                 "Estimate": f"{pl5['n_eligible']} of {len(s5)} sizes (occupancy {occ5.min():.2f}–{occ5.max():.2f})",
                 "Relative error": "D_unsat undefined above ≈ 2.6"})
    bm = boolean_spheres(160, 6, 0.08)
    s4 = np.array([2, 3, 4, 6, 8, 11, 16, 22, 32])
    bc = fractal.box_count(ArraySource(bm), s4, 4)
    pl = fractal.plateau(s4.astype(float), bc["count"], float(bm.size), 2, 0.5, 0.05)
    rows.append({"Estimator": "Box counting: longest constant-slope run (slope)",
                 "Test object": f"Boolean spheres, r = 6 voxels, φ = {bm.mean():.3f} (not fractal)",
                 "Known value": "no scaling", "Estimate": f"{pl['mean_slope']:.2f} over {pl['decades']:.2f} decade",
                 "Relative error": None})
    n, r, sig = 72, 22.0, 2.0
    z, y, x = np.ogrid[:n, :n, :n]
    ball = (z - 35.5) ** 2 + (y - 35.5) ** 2 + (x - 35.5) ** 2 <= r**2
    g = ndi.gaussian_filter(np.where(ball, 10000.0, 50000.0), sig) + rng.normal(0, 300, ball.shape)
    sp, prof = sharpness.edge_profiles(g, 30000.0, 2000.0, 800, 12, seed=0)
    est = sharpness.fit_profiles(sp, prof, 40000.0, 0.6, 0.97).get("sigma_median", np.nan)
    rows.append({"Estimator": "Edge-spread σ (voxels)", "Test object": "blurred sphere + noise", "Known value": sig,
                 "Estimate": est, "Relative error": est / sig - 1})
    for width in (16, 40):
        n, sig = 110, 2.5
        g = np.full((n, n, n), 50000.0)
        g[:, :, 35:35 + width] = 10000.0
        g = ndi.gaussian_filter(g, sig) + rng.normal(0, 300, g.shape)
        sp, prof = sharpness.edge_profiles(g, 30000.0, 4000.0, 600, 14, seed=0)
        est = sharpness.fit_profiles(sp, prof, 40000.0, 0.6, 0.97).get("sigma_median", np.nan)
        rows.append({"Estimator": "Edge-spread σ (voxels)", "Test object": f"blurred pore slab {width} voxels wide + noise",
                     "Known value": sig, "Estimate": est, "Relative error": est / sig - 1})
    return pd.DataFrame(rows)


def _add_table(doc, df: pd.DataFrame, font_pt: float = 7.0):
    from docx.shared import Pt

    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Table Grid"
    for j, c in enumerate(df.columns):
        cell = t.rows[0].cells[j]
        cell.text = str(c)
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(font_pt)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for j, v in enumerate(row):
            if isinstance(v, float):
                txt = "–" if not np.isfinite(v) else (f"{v:.4g}" if abs(v) < 1e4 else f"{v:,.0f}")
            else:
                txt = "–" if v is None or (isinstance(v, str) and v == "nan") else str(v)
            cells[j].text = txt
            for p in cells[j].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(font_pt)


def step_supplement(cfg: Config):
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.shared import Mm, Pt

    R = load_json(cfg.data_dir / "results.json")
    caps = captions(R, cfg)
    titles = {name: title for name, title, _ in __import__("drp.tables", fromlist=["TABLES"]).TABLES}
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Arial"
    st.font.size = Pt(10)
    sub = cfg.raw.get("submission", {})
    doc.add_heading("Supplementary material", level=0)
    # journal requirement: every ESM file names the article, journal, authors
    # and the corresponding author's affiliation and e-mail
    for line in (f"Article: {sub.get('article_title', cfg['project']['title'])}",
                 f"Journal: {sub.get('journal', '')}",
                 f"Authors: {', '.join(sub.get('authors', []))}",
                 f"Corresponding author: {sub.get('corresponding_author', '')}, {sub.get('corresponding_affiliation', '')}"
                 f" ({sub.get('corresponding_email', '')})"):
        doc.add_paragraph(line)
    doc.add_paragraph("All values were produced by drp-carbonates (python run_all.py); full-precision tables (CSV) and "
                      "derived data are archived with the article (see Availability of data and material).")
    doc.add_heading("S1. Validation of the estimators on synthetic volumes", level=1)
    doc.add_paragraph("Table S1. Estimators applied to synthetic volumes with known answers (related cases are "
                      "checked by the automated tests in tests/test_estimators.py).")
    _add_table(doc, validation_table(), 8)

    doc.add_heading("S2. Supplementary figures", level=1)
    s = 0
    for key, where in ORDER:
        if where != "S":
            continue
        s += 1
        png = cfg.fig_dir / f"{key}_*.png"
        files = sorted(cfg.fig_dir.glob(f"{key}_*.png"))
        if len(files) != 1:
            log.warning("supplementary figure %s: expected one file, found %d", key, len(files))
            continue
        doc.add_picture(str(files[0]), width=Mm(170))
        doc.add_paragraph(f"Fig. S{s}. {caps[key]}")

    sec = doc.add_section()
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = sec.page_height, sec.page_width
    doc.add_heading("S3. Supplementary tables", level=1)
    for k, name in enumerate(SUPP_TABLES, start=2):
        p = cfg.table_dir / f"{name}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, encoding="utf-8-sig")
        doc.add_paragraph(f"Table S{k}. {titles.get(name, name)}.")
        _add_table(doc, df, 6.5 if len(df.columns) > 9 else 7.5)
        doc.add_paragraph("")
    out = cfg.output_dir / "Supplementary_Material.docx"
    doc.save(out)
    log.info("supplementary material written: %s", out)
