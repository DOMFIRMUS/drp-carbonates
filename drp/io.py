"""TIFF-stack discovery, acquisition-metadata parsing and caching.

The stacks are stored as one 16-bit TIFF per slice. Each file carries an
ImageJ slice label written by the depositors, e.g.

    Calcite-1_OriginalImage_16bits-U_895x898x1516_45.26um0000.tif

which records the size of the original reconstruction (nx x ny x nz) and the
voxel size. That label is the only acquisition metadata shipped with the
images, so it is parsed here and treated as authoritative.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import tifffile

log = logging.getLogger(__name__)

_LABEL_RE = re.compile(r"_(\d+)x(\d+)x(\d+)_([0-9]+(?:\.[0-9]+)?)\s*um", re.IGNORECASE)
_INDEX_RE = re.compile(r"(\d+)\.tiff?$", re.IGNORECASE)


def list_slices(image_dir: Path, pattern: str = "*.tif") -> list[Path]:
    """Return the slice files sorted by their numeric index; checks contiguity."""
    files = list(Path(image_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {image_dir}")

    def index(p: Path) -> int:
        m = _INDEX_RE.search(p.name)
        if not m:
            raise ValueError(f"Cannot read slice index from {p.name}")
        return int(m.group(1))

    files.sort(key=index)
    idx = np.array([index(f) for f in files])
    if not np.array_equal(idx, np.arange(idx[0], idx[0] + len(idx))):
        missing = sorted(set(range(idx[0], idx[-1] + 1)) - set(idx.tolist()))
        raise ValueError(f"Slice indices are not contiguous; missing {missing[:10]}")
    return files


def slice_label(path: Path) -> str | None:
    with tifffile.TiffFile(path) as tf:
        tag = tf.pages[0].tags.get("IJMetadata")
        if tag is None:
            return None
        props = tag.value.get("Properties", {}) if isinstance(tag.value, dict) else {}
        label = props.get("Slice_Label")
        if label is None and isinstance(tag.value, dict):
            labels = tag.value.get("Labels")
            label = labels[0] if labels else None
        return label


def parse_label(label: str | None) -> dict:
    """Extract original dimensions and voxel size from an ImageJ slice label."""
    if not label:
        return {}
    m = _LABEL_RE.search(label)
    if not m:
        return {"label": label}
    nx, ny, nz, vox = m.groups()
    return {
        "label": label,
        "original_nx": int(nx),
        "original_ny": int(ny),
        "original_nz": int(nz),
        "voxel_size_um": float(vox),
    }


def ingest(sample, cache_dir: Path, mode_bins: int = 512) -> dict:
    """Stack the slices into a (z, y, x) uint16 .npy file (read later as memmap).

    Also records, per slice, the grey-level histogram (binned) and simple
    statistics, and the exact 16-bit histogram of the whole stack. Returns the
    metadata dictionary, which is written next to the cache.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy_path = cache_dir / f"{sample.id}_grey.npy"
    meta_path = cache_dir / f"{sample.id}_meta.json"
    hist_path = cache_dir / f"{sample.id}_slice_histograms.npz"

    files = list_slices(sample.image_dir, sample.pattern)
    first = tifffile.imread(files[0])
    ny, nx = first.shape
    nz = len(files)
    parsed = parse_label(slice_label(files[0]))
    parsed_last = parse_label(slice_label(files[-1]))

    meta_vox = parsed.get("voxel_size_um")
    if sample.voxel_size_um is None:
        if meta_vox is None:
            raise ValueError(f"{sample.id}: voxel size is 'auto' but no metadata label was found")
        voxel = meta_vox
        voxel_source = "ImageJ slice label"
    else:
        voxel = sample.voxel_size_um
        voxel_source = "config override"
        if meta_vox is not None and abs(meta_vox - voxel) > 1e-6:
            log.error("%s: configured voxel size %.3f um differs from metadata %.3f um",
                      sample.id, voxel, meta_vox)

    if meta_path.exists() and npy_path.exists() and hist_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        same_bins = meta.get("mode_bins") in (None, mode_bins)
        if meta.get("shape") == [nz, ny, nx] and meta.get("n_files") == nz and same_bins:
            if meta.get("voxel_size_um") != voxel or meta.get("voxel_source") != voxel_source:
                # every step reads the voxel size from this file: keep it in sync
                meta["voxel_size_um"] = voxel
                meta["voxel_source"] = voxel_source
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            meta["mode_bins"] = mode_bins
            log.info("%s: using cached stack %s", sample.id, npy_path.name)
            return meta

    log.info("%s: ingesting %d slices of %dx%d (%s)", sample.id, nz, ny, nx, first.dtype)
    if first.dtype != np.uint16:
        raise TypeError(f"Expected uint16 slices, found {first.dtype}")

    arr = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.uint16, shape=(nz, ny, nx))
    total_hist = np.zeros(65536, np.int64)
    slice_hists = []
    means = np.empty(nz)
    sds = np.empty(nz)
    # Coarse binning for per-slice histograms: fixed after the first slice so
    # that all slices share the same bins.
    lo, hi = np.percentile(first, [0.1, 99.9])
    width = max(1, int(round((hi - lo) / mode_bins)))
    nb = 65536 // width + 1
    for z, f in enumerate(files):
        img = tifffile.imread(f)
        if img.shape != (ny, nx):
            raise ValueError(f"Slice {f.name} has shape {img.shape}, expected {(ny, nx)}")
        arr[z] = img
        h = np.bincount(img.ravel(), minlength=65536)
        total_hist += h
        slice_hists.append(np.bincount(img.ravel() // width, minlength=nb).astype(np.int32))
        v = np.arange(65536, dtype=np.float64)
        n = h.sum()
        means[z] = (h * v).sum() / n
        sds[z] = np.sqrt((h * (v - means[z]) ** 2).sum() / n)
    arr.flush()
    del arr
    np.savez_compressed(hist_path, slice_hists=np.array(slice_hists), bin_width=width,
                        total_hist=total_hist, slice_mean=means, slice_sd=sds)

    meta = {
        "sample": sample.id,
        "n_files": nz,
        "shape": [nz, ny, nx],
        "dtype": "uint16",
        "first_file": files[0].name,
        "last_file": files[-1].name,
        "metadata_label_first": parsed.get("label"),
        "metadata_label_last": parsed_last.get("label"),
        "original_dims_xyz": [parsed.get("original_nx"), parsed.get("original_ny"), parsed.get("original_nz")],
        "voxel_size_um_metadata": meta_vox,
        "voxel_size_um": voxel,
        "voxel_source": voxel_source,
        "slice_hist_bin_width": width,
        "mode_bins": mode_bins,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def open_stack(cache_dir: Path, sample_id: str) -> np.memmap:
    return np.load(cache_dir / f"{sample_id}_grey.npy", mmap_mode="r")


def load_slice_histograms(cache_dir: Path, sample_id: str) -> dict:
    with np.load(cache_dir / f"{sample_id}_slice_histograms.npz") as z:
        return {k: z[k] for k in z.files}
