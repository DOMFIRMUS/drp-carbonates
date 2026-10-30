"""Continuous pore-size distribution by morphological opening.

For each radius r the pore phase is opened with a ball of radius r, using two
Euclidean distance transforms instead of explicit structuring elements:
seeds are voxels whose distance to the solid is >= r (a ball of radius r fits
there) and the opening is the set of pore voxels within distance < r of a
seed. F(r) = |opening_r| / |pore| is the fraction of pore volume that belongs
to pores of radius >= r (Muench & Holzer 2008, "continuous PSD").
The resolution limit enters directly: nothing below ~1 voxel can be resolved,
and radii of 1-2 voxels are dominated by partial-volume blur.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def continuous_psd(pore: np.ndarray, radii_vox) -> dict:
    pore = np.asarray(pore, bool)
    n_pore = int(pore.sum())
    if n_pore == 0:
        return {"radius_vox": np.asarray(radii_vox, float), "fraction_ge": np.zeros(len(radii_vox))}
    dt = ndi.distance_transform_edt(pore).astype(np.float32)
    frac = []
    for r in radii_vox:
        seeds = dt >= r
        if not seeds.any():
            frac.append(0.0)
            continue
        d_seed = ndi.distance_transform_edt(~seeds).astype(np.float32)
        opened = (d_seed < r) & pore
        frac.append(opened.sum() / n_pore)
        del d_seed, opened, seeds
    frac = np.minimum.accumulate(np.asarray(frac, float))  # enforce monotone
    return {"radius_vox": np.asarray(radii_vox, float), "fraction_ge": frac,
            "dt_hist": np.bincount(np.floor(dt[pore]).astype(int))}


def median_radius(radius, fraction_ge) -> float:
    """Radius at which half of the pore volume lies in larger pores."""
    radius = np.asarray(radius, float)
    f = np.asarray(fraction_ge, float)
    if f[0] < 0.5:
        return float("nan")
    i = np.nonzero(f < 0.5)[0]
    if not len(i):
        return float("nan")
    i = i[0]
    # log-linear interpolation
    x0, x1 = np.log(radius[i - 1]), np.log(radius[i])
    return float(np.exp(x0 + (0.5 - f[i - 1]) * (x1 - x0) / (f[i] - f[i - 1])))
