"""Memory-conscious access to large (z, y, x) grey-level stacks.

Nothing here loads a whole stack into memory. Metrics consume *sources*: any
object with a ``shape`` attribute and a ``slab(z0, z1)`` method returning the
sub-array ``[z0:z1]``. Sources can be an in-memory array, a memory-mapped file,
or a lazily thresholded view of the drift-corrected grey levels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Box:
    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.z1 - self.z0, self.y1 - self.y0, self.x1 - self.x0)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def as_dict(self) -> dict:
        return {"z": [self.z0, self.z1], "y": [self.y0, self.y1], "x": [self.x0, self.x1]}


def full_box(shape) -> Box:
    return Box(0, shape[0], 0, shape[1], 0, shape[2])


def central_cube(shape, edge_vox: int) -> Box:
    """Cube of `edge_vox` voxels centred in the stack (clipped to the stack)."""
    starts = []
    for n in shape:
        e = min(edge_vox, n)
        starts.append(((n - e) // 2, (n - e) // 2 + e))
    (z0, z1), (y0, y1), (x0, x1) = starts
    return Box(z0, z1, y0, y1, x0, x1)


def tile_cubes(shape, edge_vox: int) -> list[Box]:
    """Non-overlapping cubes tiling the stack, the grid centred in each axis."""
    boxes = []
    ranges = []
    for n in shape:
        k = n // edge_vox
        off = (n - k * edge_vox) // 2
        ranges.append([(off + i * edge_vox, off + (i + 1) * edge_vox) for i in range(k)])
    for z0, z1 in ranges[0]:
        for y0, y1 in ranges[1]:
            for x0, x1 in ranges[2]:
                boxes.append(Box(z0, z1, y0, y1, x0, x1))
    return boxes


# ---------------------------------------------------------------------------
# Stack with drift correction
# ---------------------------------------------------------------------------
class GreyStack:
    """Memory-mapped uint16 stack with an optional per-slice gain correction.

    The corrected grey level of slice z is ``rint(raw * scale[z])`` clipped to
    uint16. Keeping corrected values integer makes every porosity computed from
    a histogram identical to the one obtained by thresholding the voxels.
    """

    def __init__(self, npy_path: Path, voxel_um: float, scale: np.ndarray | None = None):
        self.raw = np.load(npy_path, mmap_mode="r")
        self.shape = self.raw.shape
        self.voxel_um = float(voxel_um)
        nz = self.shape[0]
        self.scale = np.ones(nz, np.float64) if scale is None else np.asarray(scale, np.float64)
        if self.scale.shape != (nz,):
            raise ValueError("scale must have one entry per slice")
        self.corrected = not np.allclose(self.scale, 1.0)

    def grey(self, z0: int, z1: int, box: Box | None = None) -> np.ndarray:
        """Corrected grey levels of slices [z0, z1) (absolute indices)."""
        if box is None:
            raw = np.asarray(self.raw[z0:z1])
        else:
            raw = np.asarray(self.raw[z0:z1, box.y0:box.y1, box.x0:box.x1])
        if not self.corrected:
            return np.array(raw, copy=True)
        out = np.empty(raw.shape, np.uint16)
        tmp = np.empty(raw.shape[1:], np.float32)
        for k in range(raw.shape[0]):
            np.multiply(raw[k], np.float32(self.scale[z0 + k]), out=tmp)
            np.rint(tmp, out=tmp)
            np.clip(tmp, 0, 65535, out=tmp)
            out[k] = tmp
        return out

    def histogram(self, box: Box | None = None, slab: int = 24) -> np.ndarray:
        box = box or full_box(self.shape)
        h = np.zeros(65536, np.int64)
        for z0 in range(box.z0, box.z1, slab):
            z1 = min(z0 + slab, box.z1)
            h += np.bincount(self.grey(z0, z1, box).ravel(), minlength=65536)
        return h


class ArraySource:
    """Source backed by an in-memory or memory-mapped array."""

    def __init__(self, arr: np.ndarray):
        self.arr = arr
        self.shape = arr.shape

    def slab(self, z0: int, z1: int) -> np.ndarray:
        return np.asarray(self.arr[z0:z1])


class GreySource:
    """Corrected grey levels inside a box (coordinates relative to the box)."""

    def __init__(self, stack: GreyStack, box: Box):
        self.stack, self.box = stack, box
        self.shape = box.shape

    def slab(self, z0: int, z1: int) -> np.ndarray:
        return self.stack.grey(self.box.z0 + z0, self.box.z0 + z1, self.box)


class MaskSource:
    """Pore mask (grey < T) inside a box, computed lazily slab by slab."""

    def __init__(self, stack: GreyStack, box: Box, threshold: float):
        self.stack, self.box = stack, box
        self.threshold = threshold
        self.shape = box.shape

    def slab(self, z0: int, z1: int) -> np.ndarray:
        return self.stack.grey(self.box.z0 + z0, self.box.z0 + z1, self.box) < self.threshold


def materialize(source, path: Path | None = None, dtype=None, slab: int = 32) -> ArraySource:
    """Evaluate a lazy source once, in memory or into a .npy memmap."""
    first = source.slab(0, 1)
    dtype = dtype or first.dtype
    if path is None:
        out = np.empty(source.shape, dtype)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        out = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=source.shape)
    for z0 in range(0, source.shape[0], slab):
        z1 = min(z0 + slab, source.shape[0])
        out[z0:z1] = source.slab(z0, z1)
    if path is not None:
        out.flush()
        del out
        out = np.load(path, mmap_mode="r")
    return ArraySource(out)


def threshold_source(grey: ArraySource, threshold: float) -> ArraySource:
    return ArraySource(np.asarray(grey.arr) < threshold)


# ---------------------------------------------------------------------------
# Drift correction
# ---------------------------------------------------------------------------
def histogram_mode(hist: np.ndarray, bin_centers: np.ndarray, sigma_bins: float = 2.0) -> float:
    """Mode of a histogram after Gaussian smoothing, refined by a parabola fit."""
    from scipy.ndimage import gaussian_filter1d

    hs = gaussian_filter1d(hist.astype(np.float64), sigma_bins)
    i = int(np.argmax(hs))
    if 0 < i < len(hs) - 1 and hs[i - 1] > 0 and hs[i + 1] > 0:
        y0, y1, y2 = np.log(hs[i - 1 : i + 2])
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        delta = float(np.clip(delta, -1, 1))
    else:
        delta = 0.0
    step = bin_centers[1] - bin_centers[0]
    return float(bin_centers[i] + delta * step)


def drift_scale(slice_hists: np.ndarray, bin_width: int, window: int) -> dict:
    """Per-slice gain that makes the solid-peak mode constant along z."""
    from scipy.ndimage import median_filter

    centers = (np.arange(slice_hists.shape[1]) + 0.5) * bin_width
    modes = np.array([histogram_mode(h, centers) for h in slice_hists])
    smooth = median_filter(modes, size=window, mode="nearest")
    ref = float(np.median(smooth))
    return {"mode_raw": modes, "mode_smooth": smooth, "reference": ref, "scale": ref / smooth}


# ---------------------------------------------------------------------------
# Area-weighted downsampling (resolution experiment)
# ---------------------------------------------------------------------------
def _weights_1d(n_in: int, factor: float) -> sparse.csr_matrix:
    """Matrix averaging input cells over output cells of width `factor`."""
    n_out = int(np.floor(n_in / factor + 1e-9))
    rows, cols, vals = [], [], []
    for j in range(n_out):
        a, b = j * factor, (j + 1) * factor
        for i in range(int(np.floor(a)), int(np.ceil(b - 1e-12))):
            w = min(b, i + 1) - max(a, i)
            if w > 1e-12:
                rows.append(j)
                cols.append(i)
                vals.append(w / factor)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_out, n_in))


def downsample(source, factor: float) -> np.ndarray:
    """Exact area-weighted block average by a (possibly non-integer) factor.

    Streams through the source slice by slice, so memory use is set by the
    output volume. Returns float32 grey levels.
    """
    nz, ny, nx = source.shape
    if abs(factor - 1.0) < 1e-9:
        return materialize(source, dtype=np.float32).arr.astype(np.float32)
    wz, wy, wx = _weights_1d(nz, factor), _weights_1d(ny, factor), _weights_1d(nx, factor)
    wx_t = wx.T.tocsr()
    out = np.zeros((wz.shape[0], wy.shape[0], wx.shape[0]), np.float32)
    wz_csc = wz.tocsc()
    for i in range(nz):
        col = wz_csc[:, i]
        if col.nnz == 0:
            continue
        s = source.slab(i, i + 1)[0].astype(np.float32)
        reduced = np.asarray((wy @ s) @ wx_t, dtype=np.float32)  # (ny', nx')
        for j, w in zip(col.indices, col.data):
            out[j] += np.float32(w) * reduced
    return out
