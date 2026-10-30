"""Specific surface area S_V (interface area per unit bulk volume).

Three independent estimators are implemented, because the value of S_V on a
voxel grid depends on how the interface is reconstructed:

* ``crofton``  - stereological intercept counting (Cauchy-Crofton formula):
  S_V = 2 <P_L>, where P_L is the number of phase changes per unit length of
  test line and <.> is the average over directions. The 13 lattice directions
  of the cubic grid are weighted by the solid angle of their Voronoi cells on
  the unit sphere (Ohser & Muecklich 2000; Lang et al. 2001). Unbiased for
  isotropic interfaces and insensitive to the voxel staircase.
* ``mc``       - area of a marching-cubes isosurface (Lorensen & Cline 1987,
  Lewiner et al. 2003), either of the binary image at level 0.5 (as in the
  original workflow) or of the grey levels at the segmentation threshold
  (sub-voxel interface).
* ``faces``    - count of exposed voxel faces; biased high by ~3/2 for smooth
  isotropic interfaces and reported only as an upper bound.

All estimators process the volume slab by slab (with one overlapping slice),
which gives exactly the same result as processing the whole volume.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

# 13 lattice directions (dz, dy, dx), one per antipodal pair.
DIRECTIONS_13 = np.array(
    [
        (0, 0, 1), (0, 1, 0), (1, 0, 0),
        (0, 1, 1), (0, 1, -1), (1, 0, 1), (1, 0, -1), (1, 1, 0), (1, -1, 0),
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
    ],
    dtype=int,
)
AXES_3 = DIRECTIONS_13[:3]


@lru_cache(maxsize=1)
def crofton_weights_13(n_points: int = 2_000_000) -> np.ndarray:
    """Voronoi solid-angle weights of the 13 direction pairs (sum to 1).

    Computed numerically with a Fibonacci lattice on the sphere, which is
    deterministic and accurate to ~1e-5.
    """
    i = np.arange(n_points) + 0.5
    phi = np.arccos(1 - 2 * i / n_points)
    theta = np.pi * (1 + 5**0.5) * i
    pts = np.stack([np.cos(phi), np.sin(phi) * np.sin(theta), np.sin(phi) * np.cos(theta)], 1)
    dirs = DIRECTIONS_13 / np.linalg.norm(DIRECTIONS_13, axis=1, keepdims=True)
    # antipodal pairs share a weight -> use |cos|
    counts = np.zeros(13)
    for start in range(0, n_points, 200_000):
        c = np.abs(pts[start : start + 200_000] @ dirs.T)
        counts += np.bincount(np.argmax(c, axis=1), minlength=13)
    return counts / counts.sum()


def _pair_views(m: np.ndarray, d, n_start: int):
    """Views (a, b) with b = a shifted by direction d; a starts in [0, n_start)."""
    dz, dy, dx = d

    def sl(delta, n):
        if delta == 0:
            return slice(0, n), slice(0, n)
        if delta > 0:
            return slice(0, n - delta), slice(delta, n)
        return slice(-delta, n), slice(0, n + delta)

    ny, nx = m.shape[1], m.shape[2]
    ya, yb = sl(dy, ny)
    xa, xb = sl(dx, nx)
    if dz == 0:
        za, zb = slice(0, n_start), slice(0, n_start)
    else:
        n_avail = min(n_start, m.shape[0] - dz)
        za, zb = slice(0, n_avail), slice(dz, dz + n_avail)
    return m[za, ya, xa], m[zb, yb, xb]


def transition_counts(source, directions=DIRECTIONS_13, slab: int = 32) -> dict:
    """Phase changes and number of voxel pairs along each lattice direction."""
    nz = source.shape[0]
    trans = np.zeros(len(directions), np.int64)
    pairs = np.zeros(len(directions), np.int64)
    n_pore = 0
    for z0 in range(0, nz, slab):
        z1 = min(z0 + slab, nz)
        m = source.slab(z0, min(z1 + 1, nz)).astype(bool, copy=False)
        n_pore += int(m[: z1 - z0].sum())
        for k, d in enumerate(directions):
            a, b = _pair_views(m, d, z1 - z0)
            trans[k] += int(np.count_nonzero(a != b))
            pairs[k] += a.size
    return {"transitions": trans, "pairs": pairs, "n_pore": n_pore}


def sv_crofton(counts: dict, voxel: float, directions=DIRECTIONS_13, weights=None) -> float:
    """S_V (1/length unit of `voxel`) from intercept counts."""
    lengths = np.linalg.norm(directions, axis=1) * voxel
    p_l = counts["transitions"] / (counts["pairs"] * lengths)
    if weights is None:
        weights = crofton_weights_13() if len(directions) == 13 else np.full(len(directions), 1 / len(directions))
    return float(2.0 * np.sum(weights * p_l))


def sv_faces(counts: dict, voxel: float) -> float:
    """Exposed-face area per unit volume (upper bound). Needs the 3 axes first."""
    t = counts["transitions"][:3]
    # one face per phase change; n_vox ~ pairs along an axis + boundary slice
    n_vox = counts["n_vox"] if "n_vox" in counts else counts["pairs"][0]
    return float(t.sum() * voxel**2 / (n_vox * voxel**3))


def mc_area(source, level: float, voxel: float, slab: int = 32) -> float:
    """Marching-cubes isosurface area, accumulated over overlapping slabs.

    Marching cubes is local to each 2x2x2 cell, so slabs sharing one slice
    partition the cells exactly and the areas add up.
    """
    from skimage.measure import marching_cubes, mesh_surface_area

    nz = source.shape[0]
    area = 0.0
    for z0 in range(0, nz - 1, slab):
        z1 = min(z0 + slab + 1, nz)
        v = source.slab(z0, z1).astype(np.float32)
        if v.shape[0] < 2 or v.min() >= level or v.max() <= level:
            continue
        verts, faces, _, _ = marching_cubes(v, level=level, spacing=(voxel, voxel, voxel),
                                            allow_degenerate=False)
        area += mesh_surface_area(verts, faces)
    return float(area)


def specific_surface(mask_source, voxel: float, grey_source=None, threshold=None,
                     slab: int = 32, do_mc: bool = True) -> dict:
    """All S_V estimators for one volume. Lengths in the unit of `voxel`."""
    nz, ny, nx = mask_source.shape
    volume = nz * ny * nx * voxel**3
    counts = transition_counts(mask_source, slab=slab)
    counts["n_vox"] = nz * ny * nx
    phi = counts["n_pore"] / counts["n_vox"]
    out = {
        "porosity": phi,
        "volume": volume,
        "sv_crofton13": sv_crofton(counts, voxel),
        "sv_crofton3": sv_crofton({k: v[:3] for k, v in counts.items() if k in ("transitions", "pairs")},
                                  voxel, AXES_3),
        "sv_faces": sv_faces(counts, voxel),
        "p_l_axes": (counts["transitions"][:3] / (counts["pairs"][:3] * voxel)).tolist(),
    }
    if do_mc:
        out["sv_mc_binary"] = mc_area(mask_source, 0.5, voxel, slab) / volume
        if grey_source is not None and threshold is not None:
            out["sv_mc_grey"] = mc_area(grey_source, float(threshold) - 0.5, voxel, slab) / volume
    for key in [k for k in out if k.startswith("sv_")]:
        out[key.replace("sv_", "s_per_vp_")] = out[key] / phi if phi > 0 else np.nan
    return out
