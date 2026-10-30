"""Directional two-point correlation function of the pore phase.

S2(r) = P(x in pore and x + r e in pore) along each lattice axis e, computed
with zero-padded FFTs over all lines of the volume (non-periodic estimator:
each lag is normalised by the number of voxel pairs actually available).
The normalised autocovariance C(r) = (S2(r) - phi^2) / (phi - phi^2) decays
from 1 to 0; its decay length is the correlation length of the microstructure,
which sets the block size needed for approximately independent sub-samples.
"""
from __future__ import annotations

import numpy as np


def _autocorr_lines(lines: np.ndarray, max_lag: int) -> np.ndarray:
    """Sum over lines of sum_i m_i m_{i+r}, for r = 0..max_lag (lines on last axis)."""
    n = lines.shape[-1]
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(lines.astype(np.float32), n=nfft, axis=-1)
    ac = np.fft.irfft(f * np.conj(f), n=nfft, axis=-1)[..., : max_lag + 1]
    return ac.reshape(-1, max_lag + 1).sum(axis=0)


def two_point(source, max_lag: int, slab: int = 16) -> dict:
    """S2 along x, y (from z-slabs) and z (from y-blocks). Lags in voxels."""
    nz, ny, nx = source.shape
    lag = {a: min(max_lag, n - 1) for a, n in zip("zyx", (nz, ny, nx))}
    acc = {a: np.zeros(lag[a] + 1) for a in "zyx"}
    n_pore = 0
    for z0 in range(0, nz, slab):
        m = source.slab(z0, min(z0 + slab, nz)).astype(np.float32)
        n_pore += m.sum()
        acc["x"] += _autocorr_lines(m, lag["x"])
        acc["y"] += _autocorr_lines(np.swapaxes(m, 1, 2), lag["y"])
    # z: gather columns block by block (a y-block of all slices)
    yb = max(1, int(8e6 // (nz * nx)))
    for y0 in range(0, ny, yb):
        cols = []
        for z0 in range(0, nz, slab):
            cols.append(source.slab(z0, min(z0 + slab, nz))[:, y0 : y0 + yb, :])
        block = np.concatenate(cols, axis=0).astype(np.float32)  # (nz, yb, nx)
        acc["z"] += _autocorr_lines(np.moveaxis(block, 0, -1), lag["z"])
    phi = n_pore / (nz * ny * nx)
    out = {"porosity": float(phi)}
    dims = {"z": nz, "y": ny, "x": nx}
    for a in "zyx":
        r = np.arange(lag[a] + 1)
        n_lines = nz * ny * nx / dims[a]
        s2 = acc[a] / (n_lines * (dims[a] - r))
        out[a] = {"lag_vox": r, "s2": s2, "c": (s2 - phi**2) / (phi - phi**2)}
    return out


def correlation_lengths(lag, c) -> dict:
    """1/e decay length and integral scale (integral of C up to its first zero)."""
    lag = np.asarray(lag, float)
    c = np.asarray(c, float)
    below = np.nonzero(c < np.exp(-1))[0]
    if len(below) and below[0] > 0:
        i = below[0]
        l_e = lag[i - 1] + (np.exp(-1) - c[i - 1]) * (lag[i] - lag[i - 1]) / (c[i] - c[i - 1])
    else:
        l_e = np.nan
    zero = np.nonzero(c <= 0)[0]
    end = zero[0] if len(zero) else len(c)
    integral = float(np.trapezoid(c[:end], lag[:end])) if end > 1 else np.nan
    return {"l_1e": float(l_e), "integral_scale": integral,
            "first_zero": float(lag[end]) if end < len(lag) else np.nan}
