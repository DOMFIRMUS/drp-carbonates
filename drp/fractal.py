"""Box-counting dimension of the pore-solid interface.

The interface is represented by its *boundary voxels* (pore voxels with at
least one solid 6-neighbour) and N(s) is the number of boxes of edge s that
contain at least one boundary voxel. For an interface that is self-similar
over a range of scales, N(s) ~ s^(-D) with 2 <= D <= 3.

Differences from the original workflow (porespy.metrics.boxcount), all of
which bias D:
* porespy counts a box when it holds both phases. A box of edge s contains
  only s - 1 internal faces, so for a plane N(s) ~ (s - 1) / s^3 and the
  local slope is 1 (not 2) at s = 2 voxels: D is biased low at small boxes.
  Counting boundary voxels removes that bias (tests: planes -> 2.00);
* only complete boxes are inspected and N(s) is extrapolated to the whole
  domain from the occupied fraction (no truncated edge boxes, which are
  always "occupied" at large s and flatten the curve);
* counts are averaged over several grid offsets (grid-placement bias);
* sizes are chosen in physical units, so both samples are fitted over the
  same physical range regardless of voxel size;
* the fit reports its range, standard error, 95 % CI and R^2, and the local
  slope d log N / d log s is returned to check that a scaling range exists.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


class BoundarySource:
    """Pore voxels with at least one solid 6-neighbour (inside the domain)."""

    def __init__(self, mask_source):
        self.src = mask_source
        self.shape = mask_source.shape

    def slab(self, z0: int, z1: int) -> np.ndarray:
        nz = self.shape[0]
        a, b = max(0, z0 - 1), min(nz, z1 + 1)
        m = self.src.slab(a, b).astype(bool, copy=False)
        solid_nb = np.zeros(m.shape, bool)
        solid_nb[1:] |= ~m[:-1]
        solid_nb[:-1] |= ~m[1:]
        solid_nb[:, 1:] |= ~m[:, :-1]
        solid_nb[:, :-1] |= ~m[:, 1:]
        solid_nb[:, :, 1:] |= ~m[:, :, :-1]
        solid_nb[:, :, :-1] |= ~m[:, :, 1:]
        out = m & solid_nb
        return out[z0 - a : z0 - a + (z1 - z0)]


def _count_boxes(source, s: int, offset=(0, 0, 0), mode: str = "occupied") -> tuple[int, int]:
    """Complete boxes of edge s that are occupied (mode='occupied') or hold
    both phases (mode='mixed'), and the number of boxes inspected."""
    nz, ny, nx = source.shape
    oz, oy, ox = offset
    kz, ky, kx = (nz - oz) // s, (ny - oy) // s, (nx - ox) // s
    if min(kz, ky, kx) == 0:
        return 0, 0
    hits = 0
    for iz in range(kz):
        z0 = oz + iz * s
        m = source.slab(z0, z0 + s)[:, oy : oy + ky * s, ox : ox + kx * s]
        any_z = m.any(axis=0).reshape(ky, s, kx, s).any(axis=(1, 3))
        if mode == "mixed":
            all_z = m.all(axis=0).reshape(ky, s, kx, s).all(axis=(1, 3))
            any_z &= ~all_z
        hits += int(np.count_nonzero(any_z))
    return hits, kz * ky * kx


def box_sizes(voxel_um: float, s_min_mm: float, s_max_mm: float, n: int,
              min_vox: int, max_vox: int) -> np.ndarray:
    """Integer box edges (voxels) approximating a geometric series in mm."""
    target = np.geomspace(s_min_mm, s_max_mm, n) * 1000.0 / voxel_um
    sizes = np.unique(np.clip(np.rint(target), min_vox, max_vox).astype(int))
    return sizes[sizes <= max_vox]


def matched_box_sizes(voxel_um: float, s_min_mm: float, s_max_mm: float, n: int, max_vox: int,
                      min_vox: int = 2) -> np.ndarray:
    """Integer box edges whose physical sizes all lie inside [s_min_mm, s_max_mm].

    Rounding a geometric series in mm to whole voxels can push the end sizes
    outside the fitting range, which then silently drops them, so two samples
    (or two resolutions) end up fitted over different windows. Here the end
    points are the first and last whole-voxel sizes inside the range and the
    interior sizes are geometric between them, so every size is fitted and
    the realised ranges of different voxel sizes agree within one voxel.
    """
    k_lo = max(min_vox, int(np.ceil(s_min_mm * 1000.0 / voxel_um - 1e-9)))
    k_hi = min(int(np.floor(s_max_mm * 1000.0 / voxel_um + 1e-9)), int(max_vox))
    if k_hi < k_lo:
        return np.array([], int)
    return np.unique(np.rint(np.geomspace(k_lo, k_hi, n)).astype(int))


def grid_with_fit_range(voxel_um: float, box_min_mm: float, box_max_mm: float, fit_min_mm: float,
                        fit_max_mm: float, n_fit: int, n_total: int, min_vox: int, max_vox: int) -> tuple:
    """Box sizes spanning [box_min, box_max] whose sizes inside the fitting
    range are `matched_box_sizes(fit_min, fit_max, n_fit)`. Returns (all, fit)."""
    fit = matched_box_sizes(voxel_um, fit_min_mm, fit_max_mm, n_fit, max_vox, min_vox)
    other = box_sizes(voxel_um, box_min_mm, box_max_mm, n_total, min_vox, max_vox)
    v = voxel_um / 1000.0
    other = other[(other * v < fit_min_mm * (1 - 1e-9)) | (other * v > fit_max_mm * (1 + 1e-9))]
    return np.unique(np.concatenate([other, fit])).astype(int), fit


def box_count(mask_source, sizes, n_offsets: int = 4, seed: int = 0, mode: str = "boundary") -> dict:
    """Interface-box count per box size, averaged over grid offsets.

    Only complete boxes are inspected, so the number of boxes that fit
    changes with size and offset. The count is therefore reported as the
    fraction of inspected boxes that are mixed times the number of boxes the
    whole domain holds (V / s^3): an estimate of N(s) for the full domain that
    is free of edge-truncation bias.
    """
    if mode == "boundary":          # mask given; boundary computed on the fly
        source, how = BoundarySource(mask_source), "occupied"
    elif mode == "occupied":        # source already is the boundary set
        source, how = mask_source, "occupied"
    else:                           # porespy definition (both phases in box)
        source, how = mask_source, "mixed"
    rng = np.random.default_rng(seed)
    vol = float(np.prod(source.shape))
    counts, spread, totals, raw = [], [], [], []
    for s in sizes:
        offs = [(0, 0, 0)]
        if s > 1:
            for _ in range(n_offsets - 1):
                offs.append(tuple(int(v) for v in rng.integers(0, s, 3)))
        c = [_count_boxes(source, int(s), o, how) for o in offs]
        frac = np.array([ci[0] / ci[1] for ci in c if ci[1] > 0], float)
        n_dom = vol / float(s) ** 3
        counts.append(frac.mean() * n_dom)
        spread.append(frac.std(ddof=1) * n_dom if len(frac) > 1 else 0.0)
        totals.append(np.mean([ci[1] for ci in c]))
        raw.append(np.mean([ci[0] for ci in c]))
    return {"size_vox": np.asarray(sizes), "count": np.asarray(counts),
            "count_sd": np.asarray(spread), "boxes": np.asarray(totals), "count_raw": np.asarray(raw)}


def local_slope(size, count) -> np.ndarray:
    """-d log N / d log s by centred differences."""
    ok = count > 0
    ls, ln = np.log(size[ok]), np.log(count[ok])
    out = np.full(len(size), np.nan)
    out[ok] = -np.gradient(ln, ls)
    return out


def fit_dimension(size, count, s_min=None, s_max=None, confidence=0.95) -> dict:
    """OLS fit of log N = a - D log s over [s_min, s_max] (same unit as size)."""
    size = np.asarray(size, float)
    count = np.asarray(count, float)
    sel = count > 0
    if s_min is not None:
        sel &= size >= s_min * (1 - 1e-9)
    if s_max is not None:
        sel &= size <= s_max * (1 + 1e-9)
    x, y = np.log(size[sel]), np.log(count[sel])
    if len(x) < 3:
        return {"D": np.nan, "n_points": int(len(x))}
    res = stats.linregress(x, y)
    tcrit = stats.t.ppf(0.5 + confidence / 2, len(x) - 2)
    return {
        "D": -res.slope,
        "se": res.stderr,
        "ci_low": -res.slope - tcrit * res.stderr,
        "ci_high": -res.slope + tcrit * res.stderr,
        "r2": res.rvalue**2,
        "intercept": res.intercept,
        "n_points": int(len(x)),
        "s_min": float(size[sel].min()),
        "s_max": float(size[sel].max()),
        "decades": float(np.log10(size[sel].max() / size[sel].min())),
    }


MIN_PLATEAU_PTS = 3          # a constant-slope run needs at least three box sizes to be reported


def fit_on_common_sizes(size, count, s_lo: float, s_hi: float, n_grid: int = 8, confidence: float = 0.95) -> dict:
    """D fitted on one geometric grid of sizes shared by several curves.

    Integer box sizes of scans with different voxels never coincide, so even
    "matched" ranges differ by up to one voxel at each end, and because the
    local slope varies with s this biases a contrast of D. Here log N is
    interpolated (piecewise linear in log s) at n_grid sizes between s_lo and
    s_hi, which must lie inside the measured range of every curve compared,
    and D is fitted on those common sizes.
    """
    size = np.asarray(size, float)
    count = np.asarray(count, float)
    ok = (count > 0) & np.isfinite(count)
    size, count = size[ok], count[ok]
    if len(size) < 2 or s_lo < size.min() * (1 - 1e-9) or s_hi > size.max() * (1 + 1e-9) or s_hi <= s_lo:
        return {"D": np.nan, "n_points": 0}
    grid = np.geomspace(s_lo, s_hi, n_grid)
    ln = np.interp(np.log(grid), np.log(size), np.log(count))
    f = fit_dimension(grid, np.exp(ln), confidence=confidence)
    f["interpolated"] = True
    return f


def slope_trend(size, count, volume, max_occupancy: float = 0.9) -> dict:
    """Spearman correlation of the local slope with log s over sizes below an occupancy.

    A self-similar interface has a constant local slope (correlation near 0);
    a finite set of resolved pores approaching saturation has a slope that
    rises steadily with s (correlation near +1).
    """
    from scipy import stats as sps

    size = np.asarray(size, float)
    ls = local_slope(size, count)
    occ = occupancy(size, count, volume)
    ok = np.isfinite(ls) & (occ <= max_occupancy)
    if ok.sum() < 4:
        return {"rho": np.nan, "n": int(ok.sum()), "max_occupancy": max_occupancy}
    return {"rho": float(sps.spearmanr(np.log(size[ok]), ls[ok]).statistic), "n": int(ok.sum()),
            "max_occupancy": max_occupancy}


def plateau(size, count, volume, s_min=None, max_occupancy: float = 0.5, tol: float = 0.05) -> dict:
    """Longest run of consecutive box sizes with a constant local slope.

    Only unsaturated sizes (occupancy <= max_occupancy) at or above `s_min`
    (the fitting floor or the effective resolution) are considered. A run
    qualifies when every local slope in it lies within `tol` of the run mean.
    Returns the run's extent in mm and decades, its mean slope, and the
    min-max of the local slope over the eligible sizes.
    """
    size = np.asarray(size, float)
    count = np.asarray(count, float)
    occ = occupancy(size, count, volume)
    ls = local_slope(size, count)
    ok = (occ <= max_occupancy) & np.isfinite(ls)
    if s_min is not None:
        ok &= size >= s_min * (1 - 1e-9)
    idx = np.nonzero(ok)[0]
    out = {"n_eligible": int(len(idx)), "tol": tol, "max_occupancy": max_occupancy,
           "slope_min": float(ls[idx].min()) if len(idx) else np.nan,
           "slope_max": float(ls[idx].max()) if len(idx) else np.nan,
           "npts": 0, "s_min_mm": np.nan, "s_max_mm": np.nan, "decades": 0.0, "mean_slope": np.nan}
    if not len(idx):
        return out
    best, best_len = None, 0
    for i in range(len(idx)):
        for j in range(i, len(idx)):
            if idx[j] - idx[i] != j - i:          # sizes must be consecutive
                break
            seg = ls[idx[i : j + 1]]
            if np.all(np.abs(seg - seg.mean()) <= tol) and j - i + 1 > best_len:
                best, best_len = (i, j), j - i + 1
    run = idx[best[0] : best[1] + 1] if best else idx[:0]
    if len(run) >= MIN_PLATEAU_PTS:
        out.update({"npts": int(len(run)), "s_min_mm": float(size[run[0]]), "s_max_mm": float(size[run[-1]]),
                    "decades": float(np.log10(size[run[-1]] / size[run[0]])), "mean_slope": float(ls[run].mean())})
    return out


def window_sensitivity(size, count, min_decades: float = 0.8) -> list[dict]:
    """D fitted over every contiguous window spanning >= min_decades."""
    size = np.asarray(size, float)
    out = []
    for i in range(len(size)):
        for j in range(i + 2, len(size)):
            if np.log10(size[j] / size[i]) + 1e-9 < min_decades:
                continue
            f = fit_dimension(size[i : j + 1], count[i : j + 1])
            f.update({"i": i, "j": j})
            out.append(f)
    return out


def occupancy(size, count, volume) -> np.ndarray:
    """Fraction of all boxes of edge s that are occupied: N(s) s^3 / V."""
    size = np.asarray(size, float)
    return np.asarray(count, float) * size**3 / float(volume)


def crossover_size(size, occ, level: float = 0.5) -> float:
    """Box edge at which the occupancy reaches `level` (log interpolation).

    Beyond this size almost every box contains interface, N(s) -> V/s^3 and
    the local slope tends to 3 whatever the geometry: box counting stops
    measuring the interface and starts measuring the box. The crossover is a
    characteristic spacing of the interface (here, of the resolved pores).
    """
    size = np.asarray(size, float)
    occ = np.asarray(occ, float)
    above = np.nonzero(occ >= level)[0]
    if not len(above):
        return float("nan")
    i = above[0]
    if i == 0:
        return float(size[0])
    x0, x1 = np.log(size[i - 1]), np.log(size[i])
    y0, y1 = occ[i - 1], occ[i]
    return float(np.exp(x0 + (level - y0) * (x1 - x0) / (y1 - y0)))


def unsaturated_fit(size, count, volume, s_min=None, max_occupancy: float = 0.5,
                    confidence: float = 0.95) -> dict:
    """D fitted only where boxes are not saturated (occupancy <= max_occupancy)."""
    size = np.asarray(size, float)
    occ = occupancy(size, count, volume)
    sel = occ <= max_occupancy
    if s_min is not None:
        sel &= size >= s_min * (1 - 1e-9)
    if sel.sum() < 3:
        return {"D": np.nan, "n_points": int(sel.sum()), "max_occupancy": max_occupancy}
    f = fit_dimension(size[sel], np.asarray(count, float)[sel], confidence=confidence)
    f["max_occupancy"] = max_occupancy
    return f


def legacy_boxcount(mask: np.ndarray, bins: int = 10) -> dict:
    """Exact re-implementation of porespy.metrics.boxcount (porespy 2.x).

    Box sizes are unique(int(logspace(1, log10(min(shape)), bins))); the image
    is tiled from the origin and truncated edge boxes are included; a box
    counts when it holds both phases. The original notebook then fitted
    log N against log s after dropping the first size.
    """
    shape = mask.shape
    Ds = np.unique(np.logspace(1, np.log10(min(shape)), bins).astype(int))
    N = []
    for d in Ds:
        starts = [np.arange(0, n, d) for n in shape]
        count = 0
        for z0 in starts[0]:
            m = mask[z0 : z0 + d]
            any_z = m.any(axis=0)
            all_z = m.all(axis=0)
            a = np.logical_or.reduceat(np.logical_or.reduceat(any_z, starts[1], axis=0), starts[2], axis=1)
            b = np.logical_and.reduceat(np.logical_and.reduceat(all_z, starts[1], axis=0), starts[2], axis=1)
            count += int(np.count_nonzero(a) - np.count_nonzero(b))
        N.append(count)
    N = np.array(N, float)
    tiles = np.array([np.prod([int(np.ceil(n / d)) for n in shape]) for d in Ds], float)
    coeffs = np.polyfit(np.log(Ds[1:]), np.log(N[1:]), 1)
    fit = fit_dimension(Ds[1:], N[1:])
    return {"size_vox": Ds, "count": N, "tiles": tiles, "occupancy": N / tiles,
            "D_legacy": float(-coeffs[0]), "fit": fit}
