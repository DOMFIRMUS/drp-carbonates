"""Provenance of every output: which configuration and which code produced it.

`run_all.py` records, after each step (and sample), the time, the hash of the
*analysis* configuration (the parsed TOML without the [submission] section,
so comments and the article title can change freely) and the hash of every
code file the step depends on, plus any of those files that were edited while
the run was going on. `stale_steps` compares these records with the current
configuration and code, so the export and the release refuse to package
outputs that no longer correspond to them.
"""
from __future__ import annotations

import hashlib
import json
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COMPUTE = ["drp/io.py", "drp/steps.py", "drp/steps_morph.py", "drp/volume.py", "drp/histogram.py", "drp/surface.py",
           "drp/fractal.py", "drp/rev.py", "drp/stats.py", "drp/connectivity.py", "drp/correlation.py",
           "drp/poresize.py", "drp/sharpness.py", "drp/context.py", "drp/config.py", "drp/util.py"]
GLOBAL_DEPS = {
    "sensitivity": ["drp/sensitivity.py", "drp/rev.py", "drp/volume.py", "drp/context.py", "drp/util.py"],
    "compare": COMPUTE + ["drp/compare.py", "drp/sensitivity.py"],
    "tables": ["drp/tables.py"],
    "figures": ["drp/figures.py", "drp/plotting.py", "drp/graphical_abstract.py"],
    "report": ["drp/report.py"],
    "captions": ["drp/captions.py"],
    "supplement": ["drp/supplement.py", "drp/captions.py", "drp/tables.py"],
    "export": ["drp/export.py"],
}
# steps whose outputs feed the derived-data package (checked by export) and
# everything a release ships (checked by make_release)
UPSTREAM_OF_EXPORT = ["compare", "tables"]
RELEASED = ["compare", "tables", "figures", "report", "captions", "supplement", "export"]


def deps(step: str) -> list[str]:
    return GLOBAL_DEPS.get(step, COMPUTE)


def analysis_config_sha256(path: Path) -> str:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    data.pop("submission", None)
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def file_sha256(rel: str) -> str:
    p = ROOT / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""


def code_hashes() -> dict[str, str]:
    files = sorted({f for d in list(GLOBAL_DEPS.values()) + [COMPUTE] for f in d} | {"run_all.py", "drp/provenance.py"})
    return {f: file_sha256(f) for f in files}


def record_step(output_dir: Path, key: str, prov: dict):
    """Append when/with what a step (and sample) ran to provenance_steps.json."""
    p = Path(output_dir) / "provenance_steps.json"
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    step = key.split(":")[0]
    start = prov["code_files"]
    changed = sorted(f for f in deps(step) if file_sha256(f) != start.get(f, ""))
    data[key] = {"run_at": time.strftime("%Y-%m-%d %H:%M:%S"), "config_sha256": prov["config_sha256"],
                 "analysis_config_sha256": prov["analysis_config_sha256"],
                 "code_files": {f: start.get(f, "") for f in deps(step)},
                 "code_changed_during_run": changed, "python": prov["python"], "quick_mode": prov["quick_mode"]}
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def stale_steps(output_dir: Path, config_path: Path, keys: list[str]) -> dict[str, str]:
    """Steps whose outputs do not correspond to the current configuration and code.

    Returns {key: reason}. A missing record counts as stale.
    """
    p = Path(output_dir) / "provenance_steps.json"
    if not p.exists():
        return {k: "no provenance record" for k in keys}
    data = json.loads(p.read_text(encoding="utf-8"))
    cur_a = analysis_config_sha256(config_path)
    cur_raw = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    out = {}
    for k in keys:
        r = data.get(k)
        if r is None:
            out[k] = "never run"
            continue
        if r.get("analysis_config_sha256"):
            if r["analysis_config_sha256"] != cur_a:
                out[k] = "analysis configuration changed"
                continue
        elif r.get("config_sha256") != cur_raw:
            out[k] = "configuration changed"
            continue
        if r.get("code_changed_during_run"):
            out[k] = "code edited while it ran: " + ", ".join(r["code_changed_during_run"])
            continue
        changed = [f for f, h in (r.get("code_files") or {}).items() if file_sha256(f) != h]
        if changed:
            out[k] = "code changed since: " + ", ".join(changed)
    return out


def order_problems(output_dir: Path, sample_ids: list[str], per_sample: list[str]) -> list[str]:
    """Steps that ran before the steps they depend on."""
    p = Path(output_dir) / "provenance_steps.json"
    if not p.exists():
        return ["no provenance record"]
    data = json.loads(p.read_text(encoding="utf-8"))
    t = {k: v.get("run_at", "") for k, v in data.items()}
    last_sample = max((t.get(f"{s}:{x}", "") for s in per_sample for x in sample_ids), default="")
    probs = []
    if t.get("compare", "") < last_sample:
        probs.append("compare ran before the last per-sample step")
    for s in ("tables", "figures", "report", "captions", "supplement", "export"):
        if t.get(s, "") < t.get("compare", ""):
            probs.append(f"{s} ran before compare")
    for s in ("figures", "tables"):
        if t.get("supplement", "") < t.get(s, ""):
            probs.append(f"supplement ran before {s}")
    if t.get("export", "") < t.get("tables", ""):
        probs.append("export ran before tables")
    return probs
