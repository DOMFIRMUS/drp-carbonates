"""Configuration loading and output-directory layout."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    id: str
    label: str
    long_label: str
    source_name: str
    image_dir: Path
    pattern: str
    voxel_size_um: float | None  # None -> read from TIFF metadata
    lab_porosity: float
    lab_porosity_sd: float
    color: str
    drift_correction: bool | None = None  # None -> [preprocessing] default


class Config:
    """Thin wrapper around the TOML file with resolved paths."""

    def __init__(self, path: str | Path, quick: bool = False):
        self.path = Path(path).resolve()
        self.root = self.path.parent
        with open(self.path, "rb") as fh:
            self.raw = tomllib.load(fh)
        self.quick = quick

        out = self.root / self.raw["project"]["output_dir"]
        if quick:
            out = out.with_name(out.name + "_quick")
        self.output_dir = out
        self.cache_dir = out / "cache"
        self.data_dir = out / "data"
        self.fig_dir = out / "figures"
        self.table_dir = out / "tables"
        self.log_dir = out / "logs"
        for d in (self.cache_dir, self.data_dir, self.fig_dir, self.table_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.samples = [self._sample(s) for s in self.raw["samples"]]

    def _sample(self, s: dict) -> Sample:
        vox = s.get("voxel_size_um", "auto")
        return Sample(
            id=s["id"],
            label=s["label"],
            long_label=s.get("long_label", s["label"]),
            source_name=s.get("source_name", s["id"]),
            image_dir=(self.root / s["image_dir"]).resolve(),
            pattern=s.get("pattern", "*.tif"),
            voxel_size_um=None if vox == "auto" else float(vox),
            lab_porosity=float(s["lab_porosity"]),
            lab_porosity_sd=float(s.get("lab_porosity_sd", 0.0)),
            color=s["color"],
            drift_correction=s.get("drift_correction"),
        )

    def __getitem__(self, section: str) -> dict:
        return self.raw[section]

    @property
    def seed(self) -> int:
        return int(self.raw["project"]["seed"])

    def sample(self, sample_id: str) -> Sample:
        for s in self.samples:
            if s.id == sample_id:
                return s
        raise KeyError(sample_id)

    # Quick mode shrinks the expensive parts so the whole pipeline can be
    # smoke-tested in a few minutes. Numbers from quick runs are not for
    # publication.
    def scaled(self, section: str, key: str):
        value = self.raw[section][key]
        if not self.quick:
            return value
        shrink = {
            ("volumes", "central_cube_mm"): 8.0,
            ("volumes", "psd_cube_mm"): 4.0,
            ("volumes", "esf_cube_mm"): 4.0,
            ("volumes", "replicate_cube_mm"): 4.0,
            ("rev", "n_windows"): 300,
            ("rev", "n_boot_alpha"): 20,
            ("rev", "null_n"): 10,
            ("stats", "n_surrogates"): 200,
            ("stats", "n_boot"): 500,
            ("esf", "n_edges"): 1500,
            ("fractal", "n_offsets"): 2,
        }
        return shrink.get((section, key), value)
