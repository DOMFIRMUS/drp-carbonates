"""Derived-data package for archiving (Zenodo), built from pipeline outputs.

Everything is written as open formats (CSV, JSON, compressed TIFF) with a data
dictionary, so the reported results can be checked or reused without Python.
The full source reconstructions are not redistributed: the package points to the
original dataset (doi:10.17612/V09Y-AW80). The two cropped stacks analysed in the
article are archived next to this package (make_release.py --with-stacks).
"""
from __future__ import annotations

import json
import logging
import shutil

import numpy as np
import pandas as pd
import tifffile

from .config import Config
from .context import SampleContext
from .util import load_json, save_json

log = logging.getLogger("drp")

DICTIONARY = """# Derived data for "{title}"

Produced by `drp-carbonates` v{version} (`python run_all.py`), from the public
micro-CT images of Ferreira et al., *Carbonates: porosity and permeability voxel
to voxel*, Digital Porous Media Portal, doi:10.17612/V09Y-AW80 (the full reconstructions are not redistributed; the two cropped stacks analysed in the
article, `*_analysed_stack_*.zip`, are in the same Zenodo record under the source's
ODC-BY 1.0 licence, with attribution to Ferreira et al. 2020). Licence of this package: CC BY 4.0.

Units: lengths in mm unless the column name says `_um` (micrometres) or `_vox`
(voxels); porosity as a fraction; specific surface S_V in mm⁻¹.
Segmentation: `f` is the partial-volume pore fraction f(I) = (I_s − I)/(I_s − I_p);
a voxel is pore if f(I) > f; the primary criterion is f = {primary_f}.
`legacy` = threshold mean − 1.1 sd of the analysed stack.

## Top level
| File | Content |
|---|---|
| `results.json` | every number reported in the article, with uncertainties (machine-readable) |
| `tables/*.csv` | the article and supplementary tables T01–T14 at full precision (journal numbering: see the article) |
| `replicates_all.csv` | one row per 5-mm sub-volume of both samples (columns below) |
| `provenance.json` | software versions, configuration and code hashes of the last run |
| `provenance_steps.json` | per step and sample: when it ran, with which configuration and code hash |
| `config.toml` | analysis parameters used |

## Per sample (`<sample>/`)
| File | Columns / content |
|---|---|
| `acquisition.json` | stack shape (z, y, x), voxel size from metadata, extents, slice labels |
| `qc_drift.csv` | `z_mm, solid_peak_raw, solid_peak_smooth, gain_scale, slice_mean, slice_sd, slice_porosity_raw_legacyT` |
| `segmentation.json` | end-members I_s, I_p (+range), thresholds of every method with porosity, k- and f-equivalents, mass-balance porosity, implied sub-resolution porosity |
| `histogram.csv` | `grey_level, count` of the analysed (drift-corrected where applied) stack, non-zero bins |
| `porosity_vs_criterion.csv` | `f, porosity, porosity_Ip_low, porosity_Ip_high` |
| `profile_{{x,y,z}}.csv` | `position_mm, porosity_f, porosity_mass_balance, porosity_legacy` per slice; for x and y also the same averaged over the central z window of common height (`*_zwin`) |
| `profile_z_uncorrected.csv` | z profiles of the stack without drift correction (sensitivity) |
| `rev_curve.csv` | requested and realised window edge, volume, mean, variance, CV, 5th/95th percentiles of window porosity (`porosity` = primary, legacy or mixing = mass balance) |
| `rev_fit.json` | variance–volume fits: alpha (straight line and finite-domain), block-bootstrap SE, Monte Carlo null band and p, whole-stack SE |
| `box_counting_central_cube.csv` | `box_mm, box_vox, N_boundary, N_sd, occupancy, local_slope, N_porespy_definition` |
| `two_point_{{x,y,z}}.csv` | `lag_mm, S2, C` (normalised autocovariance) |
| `cluster_volumes_{{6,26}}.csv` | `volume_vox, volume_mm3` of every connected pore cluster in the central cube |
| `psd.csv` | `radius_um, radius_vox, fraction_ge_primary, fraction_ge_legacy` (pore-volume fraction in pores ≥ r) |
| `esf_mean_edge.csv`, `esf_sigma.csv` | mean normalised edge profile; σ of every kept edge (µm) |
| `resolution.csv` | metrics after block averaging to coarser voxels |
| `central_cube_mask_f{primary_tag}.tif` | binary pore mask (1 = pore) of the 20-mm central cube, uint8, zlib-compressed, z-y-x order; voxel size in the TIFF resolution tags and in `acquisition.json` |

## `replicates_all.csv`
`sample, cube, layer` (z-layer index used as bootstrap cluster), `zc_mm, yc_mm, xc_mm, edge_mm`,
`porosity` (f = {primary_f}), `porosity_mixing` (mass balance), `porosity_legacyT`, `sv_crofton13`, `sv_mc_grey`,
`D` (matched range 0.1–1.25 mm), `D_r2`, `D_common_grid` (same nominal range, refitted on one
interpolated grid of sizes common to both samples), `D_unsat` (unsaturated range; empty when fewer than three unsaturated sizes),
`D_unsat_npts`, `D_unsat_2pt_fill` (two-point slope where only two unsaturated sizes exist), `D_window` (fixed
0.1–0.28 mm window), `crossover_mm` (occupancy 0.5), `largest_cluster_fraction`, `percolates_z`,
`percolating_porosity_any`, `s_per_vp` (= S_V/φ), `mean_chord_mm` (= 4φ/S_V), `k_kc_md` (Kozeny–Carman proxy
φ³/(5 S_V²) in mD, resolved pores only), `box_mm_k, box_n_k` (box-counting curve).
"""


def _write_csv(df: pd.DataFrame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.8g")


PER_SAMPLE_STEPS = ["ingest", "qc", "segmentation", "legacy", "profiles", "rev", "cube", "psd", "esf", "replicates", "sensitivity",
                    "resolution"]


def _check_provenance(cfg: Config):
    """Refuse to package outputs that no longer match the analysis configuration and code."""
    from . import provenance as prv

    keys = [f"{s}:{x.id}" for s in PER_SAMPLE_STEPS for x in cfg.samples] + prv.UPSTREAM_OF_EXPORT
    stale = prv.stale_steps(cfg.output_dir, cfg.path, keys)
    if stale:
        raise RuntimeError("outputs that do not match the current configuration/code; rerun these steps before "
                           "exporting: " + "; ".join(f"{k} ({v})" for k, v in sorted(stale.items())))


def step_export(cfg: Config):
    _check_provenance(cfg)                    # before touching the last good package
    final = cfg.output_dir / "zenodo_data"
    out = cfg.output_dir / "zenodo_data.tmp"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    from . import __version__

    primary_f = float(cfg["segmentation"]["primary_f"])
    (out / "README.md").write_text(DICTIONARY.format(title=cfg.raw.get("submission", {}).get("article_title", cfg["project"]["title"]),
                                                      version=__version__,
                                                      primary_f=primary_f,
                                                      primary_tag=f"{int(round(primary_f * 100)):03d}"),
                                   encoding="utf-8")
    shutil.copy(cfg.data_dir / "results.json", out / "results.json")
    shutil.copy(cfg.path, out / "config.toml")
    for name in ("provenance.json", "provenance_steps.json"):
        if (cfg.output_dir / name).exists():
            shutil.copy(cfg.output_dir / name, out / name)
    (out / "tables").mkdir()
    for p in sorted(cfg.table_dir.glob("T*.csv")):
        shutil.copy(p, out / "tables" / p.name)
    if (cfg.data_dir / "replicates_all.csv").exists():
        shutil.copy(cfg.data_dir / "replicates_all.csv", out / "replicates_all.csv")

    for s in cfg.samples:
        c = SampleContext(cfg, s)
        d = out / s.id
        d.mkdir()
        shutil.copy(c.data("acquisition"), d / "acquisition.json")
        shutil.copy(c.data("segmentation"), d / "segmentation.json")
        q = np.load(c.data("qc", "npz"))
        nz = len(q["mode_raw"])
        _write_csv(pd.DataFrame({
            "z_mm": (np.arange(nz) + 0.5) * c.voxel_mm, "solid_peak_raw": q["mode_raw"],
            "solid_peak_smooth": q["mode_smooth"], "gain_scale": q["scale"] if c.drift_corrected else np.ones(nz),
            "slice_mean": q["slice_mean"], "slice_sd": q["slice_sd"],
            "slice_porosity_raw_legacyT": q["slice_porosity_raw_legacyT"]}), d / "qc_drift.csv")
        h = np.load(c.data("histograms", "npz"))["corrected"]
        nzb = np.nonzero(h)[0]
        _write_csv(pd.DataFrame({"grey_level": nzb, "count": h[nzb]}), d / "histogram.csv")
        fc = c.seg["f_curve_fine"]
        _write_csv(pd.DataFrame({"f": fc["f"], "porosity": fc["porosity"], "porosity_Ip_low": fc["porosity_Ip_lo"],
                                 "porosity_Ip_high": fc["porosity_Ip_hi"]}), d / "porosity_vs_criterion.csv")
        pr = np.load(c.data("profiles", "npz"))
        for a in "xyz":
            cols = {"position_mm": pr[f"{a}_pos_mm"], "porosity_f": pr[f"{a}_binary"],
                    "porosity_mass_balance": pr[f"{a}_mixing"], "porosity_legacy": pr[f"{a}_legacy"]}
            if f"{a}_zwin_binary" in pr:
                cols.update({"porosity_f_zwin": pr[f"{a}_zwin_binary"],
                             "porosity_mass_balance_zwin": pr[f"{a}_zwin_mixing"],
                             "porosity_legacy_zwin": pr[f"{a}_zwin_legacy"]})
            _write_csv(pd.DataFrame(cols), d / f"profile_{a}.csv")
        if "z_uncorrected_binary" in pr:
            _write_csv(pd.DataFrame({"position_mm": pr["z_pos_mm"], "porosity_f": pr["z_uncorrected_binary"],
                                     "porosity_mass_balance": pr["z_uncorrected_mixing"],
                                     "porosity_legacy": pr["z_uncorrected_legacyT"]}), d / "profile_z_uncorrected.csv")
        rv = load_json(c.data("rev"))
        rows = []
        for tag in rv:
            for r in rv[tag]["curve"]:
                rows.append({"porosity": tag, **r})
        _write_csv(pd.DataFrame(rows), d / "rev_curve.csv")
        save_json(d / "rev_fit.json", {tag: rv[tag]["fit"] for tag in rv})
        if c.data("cube", "npz").exists():
            z = np.load(c.data("cube", "npz"))
            cube = load_json(c.data("cube"))
            vol = cube["edge_mm"] ** 3
            _write_csv(pd.DataFrame({"box_mm": z["box_size_mm"], "box_vox": z["box_size_vox"],
                                     "N_boundary": z["box_count"], "N_sd": z["box_count_sd"],
                                     "occupancy": z["box_count"] * z["box_size_mm"] ** 3 / vol,
                                     "local_slope": z["box_local_slope"],
                                     "N_porespy_definition": z["box_count_mixed"]}), d / "box_counting_central_cube.csv")
            for a in "xyz":
                _write_csv(pd.DataFrame({"lag_mm": z[f"s2_{a}_lag_mm"], "S2": z[f"s2_{a}"], "C": z[f"c_{a}"]}),
                           d / f"two_point_{a}.csv")
            for k in (6, 26):
                v = np.sort(z[f"cluster_volumes_{k}"])[::-1]
                _write_csv(pd.DataFrame({"volume_vox": v, "volume_mm3": v * c.voxel_mm**3}),
                           d / f"cluster_volumes_{k}.csv")
            grey = np.load(cfg.cache_dir / f"{s.id}_cube_grey.npy", mmap_mode="r")
            mask_path = d / f"central_cube_mask_f{int(round(primary_f * 100)):03d}.tif"
            res = 10.0 / c.voxel_mm  # pixels per cm
            with tifffile.TiffWriter(mask_path) as tw:
                for k in range(grey.shape[0]):
                    tw.write((np.asarray(grey[k]) < c.T_primary).astype(np.uint8), compression="zlib",
                             resolution=(res, res), resolutionunit="CENTIMETER",
                             description=json.dumps({"voxel_um": c.voxel_um, "pore": 1,
                                                     "criterion_f": primary_f, "threshold": c.T_primary})
                             if k == 0 else None, contiguous=False)
        if c.data("psd").exists():
            p = load_json(c.data("psd"))["curves"]
            _write_csv(pd.DataFrame({"radius_um": p["primary"]["radius_um"], "radius_vox": p["primary"]["radius_vox"],
                                     "fraction_ge_primary": p["primary"]["fraction_ge"],
                                     "fraction_ge_legacy": p["legacy"]["fraction_ge"]}), d / "psd.csv")
        if c.data("esf", "npz").exists():
            e = np.load(c.data("esf", "npz"))
            _write_csv(pd.DataFrame({"distance_um": e["grid_vox"] * c.voxel_um, "mean": e["mean"], "q25": e["q25"],
                                     "q75": e["q75"]}), d / "esf_mean_edge.csv")
            _write_csv(pd.DataFrame({"sigma_um": e["sigma_kept"] * c.voxel_um}), d / "esf_sigma.csv")
        if c.data("resolution").exists():
            rr = load_json(c.data("resolution"))["rows"]
            _write_csv(pd.DataFrame([{k: v for k, v in r.items() if k not in ("box_size_mm", "box_count", "shape")}
                                     for r in rr]), d / "resolution.csv")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    save_json(out / "manifest.json", {"files": sorted(str(p.relative_to(out)).replace("\\", "/")
                                                      for p in out.rglob("*") if p.is_file()),
                                      "total_bytes": size})
    if final.exists():
        shutil.rmtree(final)
    out.rename(final)
    log.info("derived-data package: %s (%.1f MB)", final, size / 1e6)
