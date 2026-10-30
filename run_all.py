#!/usr/bin/env python
"""Run the complete analysis: every number, table and figure of the article.

Usage
-----
    python run_all.py                       # everything
    python run_all.py --steps figures tables report
    python run_all.py --samples calcite1 --steps cube
    python run_all.py --quick               # smoke test (outputs_quick/, not for publication)

Steps run in the order listed in STEPS; each writes to outputs/data and can be
re-run on its own once its inputs exist.
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from drp import captions, compare, export, figures, report, sensitivity, steps, steps_morph, supplement, tables  # noqa: E402
from drp import provenance as prv  # noqa: E402
from drp.config import Config  # noqa: E402
from drp.context import SampleContext  # noqa: E402
from drp.util import save_json, setup_logging, timed  # noqa: E402

PER_SAMPLE = {
    "ingest": steps.step_ingest,
    "qc": steps.step_qc,
    "segmentation": steps.step_segmentation,
    "legacy": steps.step_legacy,
    "profiles": steps.step_profiles,
    "rev": steps.step_rev,
    "cube": steps_morph.step_cube,
    "psd": steps_morph.step_psd,
    "esf": steps_morph.step_esf,
    "replicates": steps_morph.step_replicates,
    "resolution": steps_morph.step_resolution,
    "sensitivity": sensitivity.step_sensitivity,
}
GLOBAL = {
    "compare": compare.step_compare,
    "tables": tables.step_tables,
    "figures": figures.step_figures,
    "report": report.step_report,
    "captions": captions.step_captions,
    "supplement": supplement.step_supplement,
    "export": export.step_export,
}
STEPS = list(PER_SAMPLE) + list(GLOBAL)


def code_sha256() -> str:
    """Hash of the analysis code (drp/*.py and this file)."""
    h = hashlib.sha256()
    for p in sorted((HERE / "drp").glob("*.py")) + [Path(__file__).resolve()]:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def provenance(cfg: Config, steps_run=None) -> dict:
    pkgs = ["numpy", "scipy", "pandas", "matplotlib", "scikit-image", "statsmodels", "tifffile", "openpyxl",
            "python-docx", "tabulate", "jinja2", "pillow"]
    versions = {}
    for p in pkgs:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    return {
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "config_file": str(cfg.path),
        "config_sha256": hashlib.sha256(cfg.path.read_bytes()).hexdigest(),
        "analysis_config_sha256": prv.analysis_config_sha256(cfg.path),
        "code_sha256": code_sha256(),
        "code_files": prv.code_hashes(),
        "steps_run": steps_run,
        "quick_mode": cfg.quick,
    }


def record_step(cfg: Config, key: str, prov: dict):
    """Append when/with what each step (and sample) last ran to provenance_steps.json."""
    prv.record_step(cfg.output_dir, key, prov)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.toml"))
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS)
    ap.add_argument("--samples", nargs="+", default=None)
    ap.add_argument("--quick", action="store_true", help="small volumes, few resamples (testing only)")
    args = ap.parse_args(argv)

    cfg = Config(args.config, quick=args.quick)
    log = setup_logging(cfg.log_dir)
    samples = [s for s in cfg.samples if args.samples is None or s.id in args.samples]
    wanted = [s for s in STEPS if s in args.steps]
    prov = provenance(cfg, wanted)
    save_json(cfg.output_dir / "provenance.json", prov)
    t0 = time.perf_counter()
    for step in wanted:
        if step in PER_SAMPLE:
            for s in samples:
                with timed(log, f"[{step}] {s.id}"):
                    PER_SAMPLE[step](SampleContext(cfg, s))
                record_step(cfg, f"{step}:{s.id}", prov)
        else:
            with timed(log, f"[{step}]"):
                GLOBAL[step](cfg)
            record_step(cfg, step, prov)
    log.info("finished %d step(s) in %.1f min", len(wanted), (time.perf_counter() - t0) / 60)


if __name__ == "__main__":
    main()
