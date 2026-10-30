"""Per-sample context shared by the pipeline steps."""
from __future__ import annotations

from functools import cached_property

import numpy as np

from .config import Config, Sample
from .util import load_json
from .volume import GreyStack


class SampleContext:
    def __init__(self, cfg: Config, sample: Sample):
        self.cfg = cfg
        self.sample = sample

    # -- files ---------------------------------------------------------------
    def data(self, step: str, ext: str = "json"):
        return self.cfg.data_dir / f"{step}_{self.sample.id}.{ext}"

    @property
    def grey_path(self):
        return self.cfg.cache_dir / f"{self.sample.id}_grey.npy"

    # -- acquisition ---------------------------------------------------------
    @cached_property
    def meta(self) -> dict:
        return load_json(self.cfg.cache_dir / f"{self.sample.id}_meta.json")

    @property
    def voxel_um(self) -> float:
        return float(self.meta["voxel_size_um"])

    @property
    def voxel_mm(self) -> float:
        return self.voxel_um / 1000.0

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.meta["shape"])

    def vox(self, mm: float) -> int:
        return int(round(mm / self.voxel_mm))

    # -- stacks --------------------------------------------------------------
    @cached_property
    def drift(self) -> dict:
        with np.load(self.data("qc", "npz")) as z:
            return {k: z[k] for k in z.files}

    @property
    def drift_corrected(self) -> bool:
        flag = self.sample.drift_correction
        return bool(self.cfg["preprocessing"]["drift_correction"] if flag is None else flag)

    def stack(self, corrected: bool | None = None) -> GreyStack:
        if corrected is None:
            corrected = self.drift_corrected
        scale = self.drift["scale"] if corrected else None
        return GreyStack(self.grey_path, self.voxel_um, scale)

    # -- segmentation --------------------------------------------------------
    @cached_property
    def seg(self) -> dict:
        return load_json(self.data("segmentation"))

    @property
    def I_s(self) -> float:
        return float(self.seg["I_s"])

    @property
    def I_p(self) -> float:
        return float(self.seg["I_p"])

    def T_f(self, f: float) -> float:
        """Integer threshold of partial-volume criterion f (pore if I < T)."""
        return float(np.ceil(self.I_s - f * (self.I_s - self.I_p)))

    @property
    def T_primary(self) -> float:
        return self.T_f(float(self.cfg["segmentation"]["primary_f"]))

    @property
    def T_legacy(self) -> float:
        """mean - k sd on the (drift-corrected) whole-stack histogram."""
        return float(self.seg["thresholds_by_key"]["legacy"]["T"])
