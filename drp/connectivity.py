"""Connected-component analysis of the pore phase, slab by slab.

Each slab is labelled with ``scipy.ndimage.label``; labels that touch across
slab boundaries are merged with a sparse-graph connected-components pass. The
result is identical to labelling the whole volume at once (see tests) while
memory stays proportional to one slab.

Reported: number of clusters, cluster-volume distribution, the share of pore
volume in the largest cluster, and whether clusters span the domain in each
direction (percolation). 6-connectivity (faces) is the conservative topology
for flow; 26-connectivity (faces, edges, corners) is the permissive one. The
gap between them measures how much of the connectivity hangs on single-voxel
contacts, i.e. on features at the resolution limit.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def _structure(connectivity: int) -> np.ndarray:
    return ndi.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])


def _boundary_pairs(a: np.ndarray, b: np.ndarray, connectivity: int) -> np.ndarray:
    """Label pairs linking slice a (last of slab k) and slice b (first of slab k+1)."""
    shifts = [(0, 0)] if connectivity == 6 else [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    if connectivity == 18:
        shifts = [s for s in shifts if abs(s[0]) + abs(s[1]) <= 1]
    ny, nx = a.shape
    out = []
    for dy, dx in shifts:
        ya = slice(max(0, -dy), ny - max(0, dy))
        yb = slice(max(0, dy), ny - max(0, -dy))
        xa = slice(max(0, -dx), nx - max(0, dx))
        xb = slice(max(0, dx), nx - max(0, -dx))
        la, lb = a[ya, xa], b[yb, xb]
        sel = (la > 0) & (lb > 0)
        if sel.any():
            out.append(np.stack([la[sel], lb[sel]], 1))
    if not out:
        return np.empty((0, 2), np.int64)
    return np.unique(np.concatenate(out), axis=0)


def label_clusters(source, connectivity: int = 6, slab: int = 64) -> dict:
    nz, ny, nx = source.shape
    st = _structure(connectivity)
    offset = 0
    sizes = [np.zeros(1, np.int64)]  # index 0 = background
    faces = {k: [] for k in ("z0", "z1", "y0", "y1", "x0", "x1")}
    edges = []
    prev_last = None
    for z0 in range(0, nz, slab):
        z1 = min(z0 + slab, nz)
        m = source.slab(z0, z1).astype(bool, copy=False)
        lab, n = ndi.label(m, structure=st, output=np.int32)
        lab = lab.astype(np.int64)
        lab[lab > 0] += offset
        sizes.append(np.bincount(lab.ravel(), minlength=offset + n + 1)[offset + 1 :])
        faces["y0"].append(np.unique(lab[:, 0, :]))
        faces["y1"].append(np.unique(lab[:, -1, :]))
        faces["x0"].append(np.unique(lab[:, :, 0]))
        faces["x1"].append(np.unique(lab[:, :, -1]))
        if z0 == 0:
            faces["z0"].append(np.unique(lab[0]))
        if z1 == nz:
            faces["z1"].append(np.unique(lab[-1]))
        if prev_last is not None:
            edges.append(_boundary_pairs(prev_last, lab[0], connectivity))
        prev_last = lab[-1].copy()
        offset += n
        del lab

    n_labels = offset
    size = np.concatenate(sizes)  # size[i] = voxels of provisional label i
    if n_labels == 0:
        return {"n_clusters": 0, "cluster_volumes": np.array([], np.int64), "pore_voxels": 0,
                "spans": {"z": False, "y": False, "x": False}, "n_voxels": nz * ny * nx}
    e = np.concatenate(edges) if edges else np.empty((0, 2), np.int64)
    g = sparse.coo_matrix((np.ones(len(e)), (e[:, 0] - 1, e[:, 1] - 1)), shape=(n_labels, n_labels))
    n_comp, comp = connected_components(g, directed=False)
    volumes = np.bincount(comp, weights=size[1:], minlength=n_comp).astype(np.int64)

    touch = {}
    for k, lists in faces.items():
        labs = np.unique(np.concatenate(lists)) if lists else np.array([], np.int64)
        labs = labs[labs > 0]
        flag = np.zeros(n_comp, bool)
        flag[comp[labs - 1]] = True
        touch[k] = flag
    spans = {a: touch[a + "0"] & touch[a + "1"] for a in "zyx"}
    return {"n_clusters": int(n_comp), "cluster_volumes": volumes, "spans_by_cluster": spans,
            "touches_boundary": touch["z0"] | touch["z1"] | touch["y0"] | touch["y1"] | touch["x0"] | touch["x1"],
            "pore_voxels": int(volumes.sum()), "n_voxels": int(nz * ny * nx)}


def summarize(res: dict, voxel_mm: float) -> dict:
    vols = res["cluster_volumes"]
    n_vox = res["n_voxels"]
    pore = res["pore_voxels"]
    if res["n_clusters"] == 0:
        return {"n_clusters": 0}
    order = np.argsort(vols)[::-1]
    largest = order[0]
    spans = res["spans_by_cluster"]
    spanning_any = spans["z"] | spans["y"] | spans["x"]
    spanning_all = spans["z"] & spans["y"] & spans["x"]
    isolated = ~res["touches_boundary"]
    domain_mm3 = n_vox * voxel_mm**3
    return {
        "n_clusters": int(res["n_clusters"]),
        "clusters_per_mm3": res["n_clusters"] / domain_mm3,
        "porosity": pore / n_vox,
        "largest_cluster_fraction": float(vols[largest] / pore),
        "largest_spans_z": bool(spans["z"][largest]),
        "largest_spans_y": bool(spans["y"][largest]),
        "largest_spans_x": bool(spans["x"][largest]),
        "percolating_porosity_any": float(vols[spanning_any].sum() / n_vox),
        "percolating_porosity_all": float(vols[spanning_all].sum() / n_vox),
        "percolating_fraction_of_pore_any": float(vols[spanning_any].sum() / pore),
        "isolated_porosity": float(vols[isolated].sum() / n_vox),
        "fraction_clusters_le_8vox": float(np.mean(vols <= 8)),
        "pore_fraction_in_clusters_le_8vox": float(vols[vols <= 8].sum() / pore),
        "median_cluster_volume_vox": float(np.median(vols)),
    }
