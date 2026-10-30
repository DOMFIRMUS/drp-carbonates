#!/usr/bin/env python
"""Build the archive files for Zenodo and copy the submission material.

    python make_release.py --dest ../SUBMISSAO

Creates
  <dest>/05_codigo/drp-carbonates/                 clean copy of the code (no outputs, no caches)
  <dest>/05_codigo/drp-carbonates-v<ver>.zip
  <dest>/06_dados/drp-carbonates-derived-data-v<ver>.zip   (outputs/zenodo_data)
  <dest>/02_figuras/, 03_tabelas/, 04_material_suplementar/  (from outputs/)
Figures are numbered in the journal order (captions.ORDER) and written as
vector PDF, 600-dpi PNG and 600-dpi LZW TIFF (RGB), named Fig<n>.
Run `python run_all.py` first; the script refuses to package a missing or
stale derived-data folder or an incomplete set of figures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from drp import __version__  # noqa: E402
from drp.captions import ORDER  # noqa: E402

EXCLUDE_DIRS = {"outputs", "outputs_quick", "__pycache__", ".pytest_cache", ".git"}


def code_files(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and not (set(p.relative_to(root).parts) & EXCLUDE_DIRS) and p.suffix != ".pyc":
            yield p


def zip_dir(src: Path, files, zpath: Path, arc_root: str):
    zpath.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            z.write(f, f"{arc_root}/{f.relative_to(src).as_posix()}")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_outputs(out: Path):
    """Fail early instead of shipping an empty, stale or incomplete package."""
    from drp import provenance as prv
    from drp.config import Config

    cfg = Config(HERE / "config.toml")
    per_sample = ["ingest", "qc", "segmentation", "legacy", "profiles", "rev", "cube", "psd", "esf", "replicates",
                  "resolution", "sensitivity"]
    keys = [f"{s}:{x.id}" for s in per_sample for x in cfg.samples] + prv.RELEASED
    stale = prv.stale_steps(out, cfg.path, keys)
    if stale:
        sys.exit("outputs do not match the current configuration/code; rerun:\n  "
                 + "\n  ".join(f"{k}: {v}" for k, v in sorted(stale.items())))
    probs = prv.order_problems(out, [x.id for x in cfg.samples], per_sample)
    if probs:
        sys.exit("steps ran out of order: " + "; ".join(probs))
    zd = out / "zenodo_data"
    if not (zd / "results.json").exists():
        sys.exit("outputs/zenodo_data is missing: run `python run_all.py --steps export` first")
    for name, src in (("results.json", out / "data" / "results.json"),
                      ("replicates_all.csv", out / "data" / "replicates_all.csv")):
        if not (zd / name).exists():
            sys.exit(f"outputs/zenodo_data/{name} is missing: rerun the export step")
        if src.exists() and _sha(src) != _sha(zd / name):
            sys.exit(f"outputs/zenodo_data/{name} differs from outputs/data/{name}: rerun the export step")
    failed = out / "figures" / "FAILED.json"
    if failed.exists() and json.loads(failed.read_text(encoding="utf-8")):
        sys.exit(f"figures failed in the last run: {failed.read_text(encoding='utf-8')}")
    for key, _ in ORDER + [("GA", "M")]:
        for ext in ("pdf", "png"):
            n = len(list((out / "figures").glob(f"{key}_*.{ext}")))
            if n != 1:
                sys.exit(f"expected exactly one {key}_*.{ext} in outputs/figures, found {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(HERE.parent / "SUBMISSAO"))
    ap.add_argument("--fig-prefix", default="Fig", help="file prefix for main-text figures (e.g. Fig, Figure)")
    ap.add_argument("--with-stacks", action="store_true",
                    help="also zip the analysed (cropped) TIFF stacks into 06_dados/ for the Zenodo data record")
    args = ap.parse_args()
    dest = Path(args.dest).resolve()
    out = HERE / "outputs"
    check_outputs(out)

    # code
    code_dir = dest / "05_codigo" / "drp-carbonates"
    if code_dir.exists():
        shutil.rmtree(code_dir)
    files = list(code_files(HERE))
    for f in files:
        t = code_dir / f.relative_to(HERE)
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, t)
    zip_dir(HERE, files, dest / "05_codigo" / f"drp-carbonates-v{__version__}.zip", f"drp-carbonates-v{__version__}")

    # derived data
    zd = out / "zenodo_data"
    zip_dir(zd, [p for p in sorted(zd.rglob("*")) if p.is_file()],
            dest / "06_dados" / f"drp-carbonates-derived-data-v{__version__}.zip",
            f"drp-carbonates-derived-data-v{__version__}")

    # figures: journal numbering
    fdir = dest / "02_figuras"
    sdir = dest / "04_material_suplementar"
    for d in (fdir, sdir):
        d.mkdir(parents=True, exist_ok=True)
    m = s = 0
    for key, where in ORDER:
        src = sorted((out / "figures").glob(f"{key}_*.pdf"))
        if not src:
            continue
        stem = src[0].stem
        if where == "M":
            m += 1
            name = f"{args.fig_prefix}{m}_{stem.split('_', 1)[1]}"
            target = fdir
        else:
            s += 1
            name = f"FigS{s}_{stem.split('_', 1)[1]}"
            target = sdir / "figuras_suplementares"
            target.mkdir(exist_ok=True)
        for ext in (".pdf", ".png"):
            shutil.copy2(src[0].with_suffix(ext), target / f"{name}{ext}")
        from PIL import Image

        with Image.open(src[0].with_suffix(".png")) as im:
            im.convert("RGB").save(target / f"{name}.tif", compression="tiff_lzw", dpi=(600, 600))
    ga = out / "figures" / "GA_graphical_abstract"
    for ext in (".pdf", ".png"):
        shutil.copy2(ga.with_suffix(ext), fdir / f"GraphicalAbstract{ext}")
    from PIL import Image

    with Image.open(ga.with_suffix(".png")) as im:
        im.convert("RGB").save(fdir / "GraphicalAbstract.tif", compression="tiff_lzw", dpi=(600, 600))
    shutil.copy2(out / "FIGURE_CAPTIONS.md", fdir / "LEGENDAS_FIGURAS.md")
    # tables
    tdir = dest / "03_tabelas"
    tdir.mkdir(parents=True, exist_ok=True)
    for p in (out / "tables").glob("*"):
        shutil.copy2(p, tdir / p.name)
    shutil.copy2(out / "Supplementary_Material.docx", sdir / "ESM_1.docx")   # convert to ESM_1.pdf for upload
    shutil.copy2(out / "RESULTS_SUMMARY.md", dest / "01_manuscrito" / "SINTESE_DOS_RESULTADOS.md"
                 if (dest / "01_manuscrito").exists() else dest / "SINTESE_DOS_RESULTADOS.md")
    if args.with_stacks:
        # the crop offsets within the public reconstructions were not recorded, so the analysed stacks
        # themselves are archived (source licence ODC-BY 1.0 allows redistribution with attribution)
        from drp.config import Config

        for smp in Config(HERE / "config.toml").samples:
            files = sorted(smp.image_dir.glob(smp.pattern))
            z = dest / "06_dados" / f"{smp.source_name}_analysed_stack_{len(files)}_slices.zip"
            zip_dir(smp.image_dir, files, z, smp.image_dir.name)
            print(f"stack archived: {z.name}")
    print(f"release v{__version__} written to {dest}")


if __name__ == "__main__":
    main()
