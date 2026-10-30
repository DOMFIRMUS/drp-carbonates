"""Validation of every estimator on synthetic volumes with known answers.

Run with:  python -m pytest -q tests
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drp import connectivity as conn  # noqa: E402
from drp import correlation, fractal, histogram as H, poresize, rev, sharpness, stats as st, surface  # noqa: E402
from drp.volume import ArraySource, _weights_1d, downsample  # noqa: E402


def ball_volume(n: int, r: float, center=None) -> np.ndarray:
    c = np.full(3, (n - 1) / 2) if center is None else np.asarray(center, float)
    z, y, x = np.ogrid[:n, :n, :n]
    return (z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2 <= r**2


# ---------------------------------------------------------------------------
# Specific surface
# ---------------------------------------------------------------------------
def test_crofton_weights_sum_and_symmetry():
    w = surface.crofton_weights_13()
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(w[:3], w[0], rtol=1e-3)       # axes
    assert np.allclose(w[3:9], w[3], rtol=1e-3)      # face diagonals
    assert np.allclose(w[9:], w[9], rtol=1e-3)       # body diagonals


@pytest.mark.parametrize("r", [12.3, 25.7])
def test_sv_estimators_on_sphere(r):
    n = 80
    m = ball_volume(n, r)
    true_sv = 4 * np.pi * r**2 / n**3
    res = surface.specific_surface(ArraySource(m), 1.0, slab=7)
    # reference volume of intercept counting = volume covered by voxel pairs
    cnt = surface.transition_counts(ArraySource(m), slab=7)
    ref13 = 4 * np.pi * r**2 * np.sum(surface.crofton_weights_13() / cnt["pairs"])
    assert res["sv_crofton13"] == pytest.approx(ref13, rel=0.01)
    assert res["sv_crofton13"] == pytest.approx(true_sv, rel=0.03)
    # Marching cubes on a *binary* volume keeps part of the voxel staircase:
    # it is biased high (documented here, ~5-10 % for these radii).
    assert 1.0 < res["sv_mc_binary"] / true_sv < 1.12
    assert res["sv_faces"] == pytest.approx(1.5 * true_sv, rel=0.05)


def test_mc_grey_subvoxel_sphere():
    n, r = 64, 20.4
    z, y, x = np.ogrid[:n, :n, :n]
    dist = np.sqrt((z - 31.5) ** 2 + (y - 31.5) ** 2 + (x - 31.5) ** 2)
    grey = np.clip(1000 + 100 * (dist - r), 0, 3000)  # smooth ramp, iso-level 1000 at r
    area = surface.mc_area(ArraySource(grey), 1000.0, 1.0, slab=9)
    assert area == pytest.approx(4 * np.pi * r**2, rel=0.01)


def test_slab_invariance_of_counts_and_mc():
    rng = np.random.default_rng(1)
    m = ndi.gaussian_filter(rng.random((40, 37, 33)), 2) > 0.5
    a = surface.transition_counts(ArraySource(m), slab=40)
    b = surface.transition_counts(ArraySource(m), slab=6)
    assert np.array_equal(a["transitions"], b["transitions"])
    assert np.array_equal(a["pairs"], b["pairs"])
    assert surface.mc_area(ArraySource(m), 0.5, 1.0, slab=40) == pytest.approx(
        surface.mc_area(ArraySource(m), 0.5, 1.0, slab=5), rel=1e-9)


# ---------------------------------------------------------------------------
# Fractal dimension
# ---------------------------------------------------------------------------
def test_box_count_planes_is_two():
    # Stationary layered medium: planar interfaces every 32 voxels. For boxes
    # smaller than the spacing the interface is a set of planes, D = 2.
    x = np.arange(128)
    m = np.broadcast_to(((x + 7) % 64) < 32, (128, 128, 128)).copy()
    sizes = np.array([2, 3, 4, 6, 8, 12, 16])
    bc = fractal.box_count(ArraySource(m), sizes, n_offsets=6)
    fit = fractal.fit_dimension(sizes, bc["count"])
    assert fit["D"] == pytest.approx(2.0, abs=0.03)
    assert fit["r2"] > 0.998


def test_box_count_noise_is_three():
    rng = np.random.default_rng(0)
    m = rng.random((96, 96, 96)) < 0.5
    sizes = np.array([2, 3, 4, 6, 8])
    bc = fractal.box_count(ArraySource(m), sizes, n_offsets=2)
    assert fractal.fit_dimension(sizes, bc["count"])["D"] == pytest.approx(3.0, abs=0.02)


def _porespy_boxcount_reference(im, bins=10):
    """Literal transcription of porespy.metrics.boxcount (2.x) loops."""
    Ds = np.unique(np.logspace(1, np.log10(min(im.shape)), bins).astype(int))
    N = []
    for d in Ds:
        result = 0
        for i in range(0, im.shape[0], d):
            for j in range(0, im.shape[1], d):
                for k in range(0, im.shape[2], d):
                    temp = im[i:i + d, j:j + d, k:k + d]
                    result += np.any(temp)
                    result -= np.all(temp)
        N.append(result)
    return Ds, np.array(N)


def test_legacy_boxcount_matches_porespy_algorithm():
    rng = np.random.default_rng(3)
    m = ndi.gaussian_filter(rng.random((45, 52, 38)), 1.5) > 0.52
    Ds, N = _porespy_boxcount_reference(m)
    res = fractal.legacy_boxcount(m)
    assert np.array_equal(res["size_vox"], Ds)
    assert np.array_equal(res["count"], N)


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("c", [6, 26])
def test_slabwise_labelling_equals_global(c):
    rng = np.random.default_rng(5)
    m = rng.random((50, 30, 30)) < 0.3
    lab_ref, n_ref = ndi.label(m, conn._structure(c))
    ref_sizes = np.sort(np.bincount(lab_ref.ravel())[1:])
    res = conn.label_clusters(ArraySource(m), c, slab=7)
    assert res["n_clusters"] == n_ref
    assert np.array_equal(np.sort(res["cluster_volumes"]), ref_sizes)


def test_spanning_detection():
    m = np.zeros((20, 20, 20), bool)
    m[:, 10, 10] = True  # a straight tube along z
    res = conn.label_clusters(ArraySource(m), 6, slab=6)
    s = conn.summarize(res, 1.0)
    assert s["largest_spans_z"] and not s["largest_spans_x"]
    assert s["percolating_porosity_any"] == pytest.approx(m.mean())


# ---------------------------------------------------------------------------
# Thresholds and grey-level porosity
# ---------------------------------------------------------------------------
def _hist(values):
    v = np.clip(np.rint(values), 0, 65535).astype(np.int64)
    return np.bincount(v, minlength=65536)


def test_thresholds_on_bimodal_mixture():
    rng = np.random.default_rng(0)
    a = rng.normal(20000, 1500, 300_000)
    b = rng.normal(40000, 1500, 700_000)
    h = _hist(np.concatenate([a, b]))
    for T in (H.t_otsu(h), H.t_kittler_illingworth(h), H.t_li(h)):
        assert 25000 < T < 35000
        assert H.porosity_at(h, T) == pytest.approx(0.3, abs=0.01)


def test_mixing_porosity_unbiased_under_noise():
    # 20 % pore at I=10000, 80 % solid at I=50000, plus partial-volume voxels
    rng = np.random.default_rng(2)
    frac = np.concatenate([np.ones(200_000), np.zeros(700_000), rng.random(100_000)])
    I = 50000 - frac * 40000 + rng.normal(0, 2000, frac.size)
    h = _hist(I)
    assert H.mixing_porosity(h, 50000, 10000) == pytest.approx(frac.mean(), abs=0.002)


def test_quantile_threshold_reproduces_target():
    rng = np.random.default_rng(4)
    h = _hist(rng.normal(30000, 5000, 500_000))
    T = H.t_quantile(h, 0.149)
    assert H.porosity_at(h, T) == pytest.approx(0.149, abs=1e-4)


# ---------------------------------------------------------------------------
# REV, correlation, PSD, ESF, downsampling
# ---------------------------------------------------------------------------
def test_integral_range_of_white_noise_is_one_voxel():
    rng = np.random.default_rng(0)
    phi = 0.2
    counts = rng.binomial(8, phi, size=(60, 60, 60)).astype(np.int32)  # 2^3 blocks of iid voxels
    rows = rev.variance_curve(counts, 2, 1.0, [4, 6, 8, 12, 16, 24], 3000, seed=1)
    fit = rev.integral_range(rows, phi, 4, 0.05, 120.0**3)
    assert fit["alpha"] == pytest.approx(1.0, abs=0.08)
    assert fit["A3_mm3_if_alpha1"] == pytest.approx(1.0, rel=0.15)
    assert fit["porosity_se_total_powerlaw"] == pytest.approx(fit["porosity_se_total_if_alpha1"], rel=0.25)


def test_two_point_white_noise():
    rng = np.random.default_rng(0)
    m = rng.random((40, 40, 40)) < 0.25
    s2 = correlation.two_point(ArraySource(m), 10, slab=7)
    for a in "zyx":
        assert s2[a]["s2"][0] == pytest.approx(m.mean(), abs=1e-6)
        assert np.all(np.abs(s2[a]["c"][1:]) < 0.02)


def test_psd_single_sphere():
    m = ball_volume(60, 15.0)
    res = poresize.continuous_psd(m, [2, 5, 10, 14, 16, 20])
    f = res["fraction_ge"]
    assert f[0] == pytest.approx(1.0, abs=0.02)
    assert f[3] > 0.9 and f[4] < 0.05


def test_esf_recovers_blur():
    n, r, sigma = 72, 22.0, 2.0
    m = ball_volume(n, r)
    grey = np.where(m, 10000.0, 50000.0)
    grey = ndi.gaussian_filter(grey, sigma) + np.random.default_rng(0).normal(0, 300, grey.shape)
    s, prof = sharpness.edge_profiles(grey, 30000.0, 2000.0, 800, 12, seed=0)
    res = sharpness.fit_profiles(s, prof, 40000.0, 0.6, 0.97)
    assert res["n_kept"] > 50
    assert res["sigma_median"] == pytest.approx(sigma, rel=0.12)


def test_downsample_preserves_mean_and_block_average():
    rng = np.random.default_rng(0)
    g = rng.random((24, 18, 30)).astype(np.float32)
    out = downsample(ArraySource(g), 2.0)
    ref = g.reshape(12, 2, 9, 2, 15, 2).mean(axis=(1, 3, 5))
    assert np.allclose(out, ref, atol=1e-6)
    out2 = downsample(ArraySource(g), 1.5)
    assert out2.mean() == pytest.approx(g[:24, :18, :30].mean(), rel=1e-3)
    assert np.allclose(np.asarray(_weights_1d(10, 2.5).sum(axis=1)).ravel(), 1.0)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def test_effective_n_ar1():
    rng = np.random.default_rng(0)
    rho, n = 0.8, 20000
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    expected = n * (1 - rho) / (1 + rho)
    assert st.effective_n(x)["n_eff"] == pytest.approx(expected, rel=0.2)


def test_hedges_g_known():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 1.0, 4000)
    b = rng.normal(0.0, 1.0, 4000)
    assert st.hedges_g(a, b) == pytest.approx(1.0, abs=0.06)


# ---------------------------------------------------------------------------
# Added with the October 2026 review: matched box sizes, small-n_eff inference,
# trend calibration, layer-block resampling, REV fits and nulls, plateaus,
# known-answer fractal and edge-spread controls
# ---------------------------------------------------------------------------
def test_matched_box_sizes_share_the_fitting_window():
    for v in (28.84, 45.26, 67.89, 181.04):
        s = fractal.matched_box_sizes(v, 0.1, 2.5, 10, 10_000)
        mm = s * v / 1000
        assert mm.min() >= 0.1 - 1e-9 and mm.max() <= 2.5 + 1e-9
        assert mm.min() < 0.1 + 2 * v / 1000 and mm.max() > 2.5 - v / 1000
        fit = fractal.fit_dimension(mm, mm ** -2.3, mm.min(), mm.max())
        assert fit["n_points"] == len(s)


def _ar1(n, rho, rng):
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1 - rho**2)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def test_mean_ci_neff_covers_on_slab_series_and_refuses_unit_roots():
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(300):
        ci = st.mean_ci_neff(_ar1(69, 0.8, rng))          # ~1-mm slab series
        hits += ci["ci_low"] <= 0 <= ci["ci_high"]
    assert hits / 300 >= 0.85
    ci = st.mean_ci_neff(_ar1(1516, 0.995, rng))           # per-slice series: too persistent
    assert not ci["neff_estimable"] and np.isnan(ci["ci_low"])


def test_trend_test_size_under_persistent_noise():
    rng = np.random.default_rng(2)
    rej = 0
    for k in range(150):
        y = _ar1(500, 0.98, rng)
        rej += st.trend_test(np.arange(500.0), y, 300, seed=k)["p_value"] < 0.05
    assert 0.005 <= rej / 150 <= 0.11


def test_layer_blocks_keep_layers_whole_and_widen_ci():
    rng = np.random.default_rng(3)
    layers = np.repeat(np.arange(12), 20)
    effect = np.repeat(_ar1(12, 0.8, rng), 20)
    x = effect + rng.normal(0, 0.3, len(layers))
    lb = st.LayerBlocks(layers, x)
    idx = lb.draw(rng)
    assert len(idx) == len(x) and set(np.bincount(layers[idx])) <= {0, 20, 40, 60, 80}
    blk = st.block_bootstrap_mean(x, layers, 2000)
    iid = st.bootstrap_ci(x, n_boot=2000)
    assert blk["ci_high"] - blk["ci_low"] > 1.5 * (iid[1] - iid[0])


def test_finite_domain_fit_recovers_exponent():
    V = np.geomspace(8, 8000, 10)
    Vd = 40000.0
    var = 0.01 * (V ** -0.4 - Vd ** -0.4)
    fp = rev.fit_powerlaw(V, var, Vd)
    assert fp["alpha_fd"] == pytest.approx(0.4, abs=1e-3)
    assert fp["alpha"] > 0.4          # the straight line overstates alpha in a finite domain


def test_rev_alpha_of_stationary_field_is_near_one():
    # 300 fields give mean 0.95 +/- 0.16 (Table S1); 20 fields: mean within ~3 SE
    rng = np.random.default_rng(4)
    al = []
    for k in range(20):
        g = rev.gaussian_null_field((150, 70, 70), 1.0 / 0.18, rng)
        c = (g < np.quantile(g, 0.08)).astype(np.int32) * 64
        rows = rev.variance_curve(c, 4, 0.045, [2, 3, 4, 5, 6, 8, 10, 12], 2000, seed=k)
        al.append(rev.integral_range(rows, 0.08, 2.0, 0.05, float(np.prod(c.shape)) * 0.18**3)["alpha"])
    assert 0.83 <= np.mean(al) <= 1.07


def test_plateau_found_only_for_power_law():
    s = np.geomspace(0.1, 2.0, 10)
    n_pl = 1e4 * s ** -2.5
    p = fractal.plateau(s, n_pl, volume=1e9, s_min=0.1, max_occupancy=1.0, tol=0.05)
    assert p["npts"] == 10 and p["mean_slope"] == pytest.approx(2.5, abs=1e-6)
    n_curved = 1e4 * np.exp(-1.5 * np.log(s) - 0.8 * np.log(s) ** 2)
    q = fractal.plateau(s, n_curved, volume=1e9, s_min=0.1, max_occupancy=1.0, tol=0.05)
    assert q["npts"] < 4


def test_box_count_menger_sponge():
    from drp.supplement import menger_sponge

    pore = ~menger_sponge(5)                       # 243^3, as in Table S1
    s = np.array([3, 9, 27, 81])
    d = fractal.fit_dimension(s, fractal.box_count(ArraySource(pore), s, 1)["count"])["D"]
    assert d == pytest.approx(np.log(20) / np.log(3), rel=0.045)


def test_boolean_spheres_have_no_scaling_regime():
    from drp.supplement import boolean_spheres

    b = boolean_spheres(160, 6, 0.08)
    s = np.array([2, 3, 4, 6, 8, 11, 16, 22, 32])
    bc = fractal.box_count(ArraySource(b), s, 4)
    pl = fractal.plateau(s.astype(float), bc["count"], float(b.size), 2, 0.5, 0.05)
    assert pl["decades"] < 0.8                     # quasi-plateau at most, never a scaling regime
    assert fractal.slope_trend(s.astype(float), bc["count"], float(b.size))["rho"] > 0.5


@pytest.mark.parametrize("width,lo,hi", [(16, -0.12, -0.03), (40, -0.02, 0.02)])
def test_esf_bias_for_narrow_pores(width, lo, hi):
    n, sig = 110, 2.5
    g = np.full((n, n, n), 50000.0)
    g[:, :, 35:35 + width] = 10000.0
    g = ndi.gaussian_filter(g, sig) + np.random.default_rng(0).normal(0, 300, g.shape)
    sp, prof = sharpness.edge_profiles(g, 30000.0, 4000.0, 600, 14, seed=0)
    est = sharpness.fit_profiles(sp, prof, 40000.0, 0.6, 0.97)["sigma_median"]
    assert lo <= est / sig - 1 <= hi


def test_layer_mean_inference_is_roughly_calibrated():
    rng = np.random.default_rng(6)
    la, lb = np.repeat(np.arange(13), 20), np.repeat(np.arange(8), 20)
    cov = rej = n_est = 0
    for _ in range(200):
        a = np.repeat(_ar1(13, 0.5, rng), 20) + rng.normal(0, 0.4, len(la))
        b = np.repeat(_ar1(8, 0.5, rng), 20) + rng.normal(0, 0.4, len(lb))
        sm = st.layer_mean_summary(a, la)
        if sm["estimable"]:
            n_est += 1
            cov += sm["ci_low"] <= 0 <= sm["ci_high"]
        c = st.layer_mean_contrast(a, b, la, lb)
        rej += bool(c["estimable"] and c["p_approx"] < 0.05)
    assert n_est >= 140 and cov / n_est >= 0.85 and rej / 200 <= 0.11     # ~80 % estimable at rho = 0.5
    trend = np.repeat(np.linspace(0, 3, 13), 20) + rng.normal(0, 0.1, len(la))   # monotone z-gradient
    assert not st.layer_mean_summary(trend, la)["estimable"]


def test_kendall_rho_flags_unit_root():
    rng = np.random.default_rng(7)
    rc, ok = st.kendall_rho(_ar1(60, 0.3, rng))
    assert ok and rc < 0.95
    flags = [st.kendall_rho(np.cumsum(rng.normal(size=600)))[1] for _ in range(20)]   # random walks
    assert sum(flags) <= 2


def test_association_test_does_not_fire_on_independent_persistent_series():
    rng = np.random.default_rng(8)
    fired = sum(st.association_test(_ar1(400, 0.99, rng), _ar1(400, 0.99, rng), 300, seed=k)["p_surrogate"] < 0.05
                for k in range(60))
    assert fired / 60 <= 0.15


def test_provenance_ignores_comments_and_submission_but_not_parameters(tmp_path):
    import json
    from drp import provenance as prv

    cfg = tmp_path / "config.toml"
    cfg.write_text('[a]\nx = 1\n[submission]\ntitle = "t"\n', encoding="utf-8")
    prov = {"config_sha256": "raw", "analysis_config_sha256": prv.analysis_config_sha256(cfg),
            "code_files": prv.code_hashes(), "python": "3", "quick_mode": False}
    prv.record_step(tmp_path, "tables", prov)
    data = json.loads((tmp_path / "provenance_steps.json").read_text(encoding="utf-8"))
    data["export"] = {"analysis_config_sha256": "old"}           # stale downstream record
    (tmp_path / "provenance_steps.json").write_text(json.dumps(data), encoding="utf-8")
    cfg.write_text('# a comment\n[a]\nx = 1\n[submission]\ntitle = "final"\n', encoding="utf-8")
    assert prv.stale_steps(tmp_path, cfg, ["tables"]) == {}
    cfg.write_text('[a]\nx = 2\n', encoding="utf-8")
    assert "tables" in prv.stale_steps(tmp_path, cfg, ["tables"])
    assert "compare" in prv.stale_steps(tmp_path, cfg, ["compare"])     # never run


def test_latex_tables_escape_comparison_signs():
    from drp.tables import _tex

    t = _tex("p < 0.0001 and r > 2 voxels, α ≤ 1, 50 %")
    assert "<" not in t and ">" not in t and "\\textless{}" in t and "\\%" in t


def test_z_detrending_removes_a_linear_trend_and_raises_alpha():
    """A linear trend along z flattens the variance-volume decay; removing it restores alpha ~ 1."""
    from drp.sensitivity import detrend_z

    rng = np.random.default_rng(21)
    field = (rng.random((240, 60, 60)) < 0.2).astype(np.float64)       # white noise, alpha = 1
    trended = field + np.linspace(-0.15, 0.15, 240)[:, None, None]
    d = detrend_z(trended)
    assert abs(d.mean() - trended.mean()) < 1e-12
    assert abs(np.polyfit(np.arange(240), d.mean(axis=(1, 2)), 1)[0]) < 1e-12
    edges = [2, 3, 4, 6, 8, 10, 14, 18, 24, 30]
    a_tr, _ = rev._alphas_of(trended, 1, 1.0, edges, 400, 0, 2, None)
    a_dt, _ = rev._alphas_of(d, 1, 1.0, edges, 400, 0, 2, None)
    assert a_tr < 0.8 and 0.85 < a_dt < 1.15


def test_solid_endmember_that_closes_the_budget():
    """I_s* = (mean - phi_lab I_p) / (1 - phi_lab) gives a mass balance equal to phi_lab."""
    I_p, mean, lab = 5563.7, 11499.6, 0.149
    I_s = (mean - lab * I_p) / (1 - lab)
    assert abs((I_s - mean) / (I_s - I_p) - lab) < 1e-12
