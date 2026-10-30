"""Statistical tools used across the pipeline.

Design principle: every sample here is ONE specimen per lithology, so
between-sample tests describe the two imaged specimens, not the rock types.
Sub-volumes of one specimen are spatially autocorrelated; wherever they are
used as replicates, uncertainty is obtained with block/cluster resampling or
effective sample sizes rather than with i.i.d. formulas.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Descriptive statistics with bootstrap confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_ci(x, statistic=np.mean, n_boot: int = 5000, confidence: float = 0.95,
                 seed: int = 0, method: str = "BCa") -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return (np.nan, np.nan)
    res = stats.bootstrap((x,), statistic, n_resamples=n_boot, confidence_level=confidence,
                          method=method, random_state=np.random.default_rng(seed), vectorized=False)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def describe(x, n_boot: int = 5000, confidence: float = 0.95, seed: int = 0) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    lo, hi = bootstrap_ci(x, np.mean, n_boot, confidence, seed)
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "cv": float(x.std(ddof=1) / x.mean()) if x.mean() != 0 else np.nan,
            "median": float(med), "q1": float(q1), "q3": float(q3),
            "min": float(x.min()), "max": float(x.max()),
            "mean_ci_low": lo, "mean_ci_high": hi}


# ---------------------------------------------------------------------------
# Serial dependence: effective sample size and HAC trend test
# ---------------------------------------------------------------------------
def acf(x, max_lag: int | None = None) -> np.ndarray:
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    max_lag = n - 1 if max_lag is None else min(max_lag, n - 1)
    f = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(f * np.conj(f))[: max_lag + 1]
    return ac / ac[0]


def _ips(r: np.ndarray) -> tuple[float, int]:
    """Initial-positive-sequence sum of an ACF and the lag where it stops."""
    s, k_end = 0.0, 1
    for k in range(1, len(r)):
        if r[k] <= 0:
            break
        s += r[k]
        k_end = k
    return s, k_end


def _ar1_neff(n: int, rho: float) -> float:
    """n / (1 + 2 sum_{k<n} (1 - k/n) rho^k): effective size of the mean of an AR(1)."""
    if not np.isfinite(rho) or rho <= 0:
        return float(n)
    k = np.arange(1, n)
    return float(n / (1 + 2 * np.sum((1 - k / n) * rho**k)))


RHO_CAP = 0.95


def kendall_rho(x, cap: float = RHO_CAP) -> tuple[float, bool]:
    """Lag-1 autocorrelation with Kendall's bias correction, rho + (1 + 3 rho)/n.

    Returns (rho_c, estimable). Above `cap` the series is too persistent for
    its length to tell an AR(1) from a trend or a unit root; rho_c is then
    clipped and `estimable` is False, so callers can report "not estimable"
    instead of a number set by the cap.
    """
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return np.nan, False
    r1 = float(acf(x, 1)[1])
    rc = r1 + (1 + 3 * r1) / n if r1 > 0 else r1
    estimable = bool(rc < cap and n * (1 - r1) >= 2)
    return float(min(rc, cap)), estimable


def effective_n(x, cap: float = RHO_CAP) -> dict:
    """Effective sample size of the mean of a serially dependent series.

    The initial-positive-sequence estimator (sum of the sample ACF up to its
    first non-positive lag; Thiebaux & Zwiers 1984) is biased high when n_eff
    is small, because the sample ACF of a strongly persistent series decays
    too fast. The primary value therefore uses an AR(1) model whose lag-1
    coefficient is bias-corrected (Kendall 1954) and capped at `cap`
    (`neff_estimable` is False when the cap binds); the IPS value is kept as
    `n_eff_ips`, and its stopping lag is used as the decorrelation length
    elsewhere.
    """
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return {"n": n, "n_eff": float(n), "n_eff_ips": float(n), "rho1": np.nan, "decorrelation_lag": 1,
                "neff_estimable": False}
    r = acf(x, n // 2)
    s, k_end = _ips(r)
    n_ips = float(min(n, max(1.0, n / (1 + 2 * s))))
    rho_c, ok = kendall_rho(x, cap)
    n_ar = float(min(n, max(1.0, _ar1_neff(n, rho_c))))
    return {"n": n, "n_eff": n_ar, "n_eff_ips": n_ips, "rho1": float(r[1]), "rho1_corrected": rho_c,
            "decorrelation_lag": int(k_end), "neff_estimable": ok}


def mean_ci_neff(x, confidence: float = 0.95) -> dict:
    """t interval for the mean with the effective sample size.

    Under serial dependence the sample variance is biased low by the factor
    (1 - 1/n_eff) n/(n - 1); it is corrected before computing the SE.
    """
    x = np.asarray(x, float)
    ne = effective_n(x)
    n, k = len(x), ne["n_eff"]
    s2 = x.var(ddof=1)
    if not ne["neff_estimable"] or k < 3:
        # too persistent for its length: no defensible interval
        return {**ne, "mean": float(x.mean()), "ci_low": np.nan, "ci_high": np.nan}
    sigma2 = s2 * (n - 1) / (n * (1 - 1 / k))
    t = stats.t.ppf(0.5 + confidence / 2, max(1.0, k - 1))
    half = t * np.sqrt(sigma2 / k)
    return {**ne, "mean": float(x.mean()), "ci_low": float(x.mean() - half), "ci_high": float(x.mean() + half)}


def _nw_slope(xc: np.ndarray, sxx: float, e: np.ndarray, maxlags: int) -> float:
    """Newey-West (Bartlett) variance of an OLS slope; residuals e may be 2-D (rows = series)."""
    h = np.atleast_2d(e) * xc
    n = h.shape[1]
    f = np.fft.rfft(h, n=2 * n, axis=1)
    g = np.fft.irfft(f * np.conj(f), axis=1)[:, : maxlags + 1]
    w = 1 - np.arange(maxlags + 1) / (maxlags + 1)
    s = g[:, 0] + 2 * (w[1:] * g[:, 1:]).sum(axis=1)
    return s / sxx**2


def _phase_surrogates(r: np.ndarray, n_surr: int, rng) -> np.ndarray:
    """Stationary Gaussian surrogates with the periodogram of r.

    The series is mirrored (no wrap-around jump), its Fourier phases are
    randomised, and a random window of the original length is kept.
    """
    n = len(r)
    e = np.concatenate([r, r[::-1]])
    m = len(e)
    amp = np.abs(np.fft.rfft(e))
    nf = len(amp)
    ph = rng.uniform(0, 2 * np.pi, (n_surr, nf))
    ph[:, 0] = 0.0
    if m % 2 == 0:
        ph[:, -1] = 0.0
    sur = np.fft.irfft(amp * np.exp(1j * ph), n=m, axis=1)
    starts = rng.integers(0, m - n + 1, n_surr)
    idx = starts[:, None] + np.arange(n)[None, :]
    return np.take_along_axis(sur, idx, axis=1)


def _ar1_series(n: int, rho: float, sigma: float, n_rep: int, rng) -> np.ndarray:
    e = rng.normal(0.0, sigma, (n_rep, n))
    out = np.empty((n_rep, n))
    out[:, 0] = e[:, 0] / np.sqrt(max(1e-12, 1 - rho**2))
    for i in range(1, n):
        out[:, i] = rho * out[:, i - 1] + e[:, i]
    return out


def _t_slope(xc, sxx, ys, maxlags):
    ys = np.atleast_2d(ys)
    ys = ys - ys.mean(axis=1, keepdims=True)
    b = (ys * xc).sum(axis=1) / sxx
    res = ys - b[:, None] * xc
    return np.abs(b / np.sqrt(_nw_slope(xc, sxx, res, maxlags)))


def trend_test(position, y, n_surr: int = 2000, confidence: float = 0.95, seed: int = 0) -> dict:
    """Linear trend with a calibrated Newey-West t statistic.

    With strongly persistent series (lag-1 correlation > 0.97, a handful of
    effective observations) normal-reference HAC p-values over-reject badly.
    The bandwidth is the decorrelation lag of the OLS residuals, and the null
    distribution of |t| is simulated:
    * primary: parametric bootstrap from a stationary AR(1) whose lag-1
      coefficient is estimated on the demeaned series (no detrending, which
      would remove the very persistence that creates spurious trends) and
      bias-corrected (Kendall). In simulations of AR(1) series with
      rho = 0.98-0.995 its size was 2.5-7 % at the 5 % level. If the profile
      has longer memory than an AR(1), this p is optimistic;
    * conservative bound: phase-randomised surrogates of the mirrored series
      (size below 1 % in the same simulations).
    p = (#{|t*| >= |t|} + 1)/(B + 1); the CI is slope +/- q(|t*|) SE_NW with
    the AR(1) quantile.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(position, float)
    y = np.asarray(y, float)
    n = len(y)
    xc = x - x.mean()
    sxx = float(np.sum(xc**2))
    slope = float(np.sum(xc * (y - y.mean())) / sxx)
    r = y - y.mean() - slope * xc
    maxlags = int(min(n - 2, max(int(np.ceil(4 * (n / 100) ** (2 / 9))), effective_n(r)["decorrelation_lag"])))
    se = float(np.sqrt(_nw_slope(xc, sxx, r, maxlags)[0]))
    t_obs = abs(slope / se) if se > 0 else np.nan
    yd = y - y.mean()
    r1 = float(acf(yd, 1)[1])
    rho = float(min(r1 + (1 + 3 * r1) / n, 0.9995)) if r1 > 0 else r1
    sigma = float(np.std(yd[1:] - rho * yd[:-1]))
    t_ar = _t_slope(xc, sxx, _ar1_series(n, rho, sigma, n_surr, rng), maxlags)
    t_ph = _t_slope(xc, sxx, _phase_surrogates(yd, n_surr, rng), maxlags)
    p = float((np.sum(t_ar >= t_obs) + 1) / (n_surr + 1))
    p_cons = float((np.sum(t_ph >= t_obs) + 1) / (n_surr + 1))
    q = float(np.quantile(t_ar, confidence))
    return {"slope": slope, "slope_se": se, "t": float(np.sign(slope) * t_obs), "maxlags": maxlags,
            "p_value": p, "p_conservative": p_cons, "ar1_rho": rho,
            "t_crit": q, "ci_low": slope - q * se, "ci_high": slope + q * se, "n_surrogates": n_surr,
            "method": "OLS slope, Newey-West SE; |t| calibrated by AR(1) parametric bootstrap "
                      "(conservative bound: phase-randomised surrogates)"}


def circular_block_bootstrap(x, statistic, block: int, n_boot: int, seed: int = 0) -> np.ndarray:
    """Circular block bootstrap (Politis & Romano 1992) of a serial statistic."""
    x = np.asarray(x, float)
    n = len(x)
    rng = np.random.default_rng(seed)
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    xx = np.concatenate([x, x[:block]])
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        out[b] = statistic(xx[idx])
    return out


def cv(x) -> float:
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / x.mean())


# ---------------------------------------------------------------------------
# Two-sample comparisons (effect sizes first, p-values second)
# ---------------------------------------------------------------------------
def hedges_g(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = (a.mean() - b.mean()) / sp
    return float(d * (1 - 3 / (4 * (na + nb) - 9)))


def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2 * u / (len(a) * len(b)) - 1)


def boot_p(draws) -> float:
    """Two-sided bootstrap p-value for H0: parameter = 0, (k + 1)/(B + 1) form.

    Never returns 0: with B resamples the smallest attainable p is 2/(B + 1).
    """
    x = np.asarray(draws, float)
    x = x[np.isfinite(x)]
    b = len(x)
    lo = (np.sum(x <= 0) + 1) / (b + 1)
    hi = (np.sum(x >= 0) + 1) / (b + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def layer_mean_summary(x, layers, confidence: float = 0.95) -> dict:
    """Mean of sub-volume values with an interval from the series of z-layer means.

    Sub-volumes of one z-layer are correlated, and layer means are strongly
    correlated along z (lag-1 0.6-0.9 here). Resampling blocks of 2-4 of the
    8-13 layers under-covers badly (simulated coverage 0.4-0.6 at rho 0.7-0.85),
    so the interval is built on the G layer means with an AR(1) effective
    size (Kendall-corrected rho, capped at 0.95; n_eff >= 1.5) and the
    variance corrected for serial dependence; t with n_eff - 1 dof. When
    the layer means are too persistent for an AR(1) to be estimated (the
    corrected rho reaches the cap, typically a monotone z-gradient), no
    interval is returned (`estimable` False) rather than one set by the cap;
    the same holds when n_eff < 3 (a t interval with < 2 dof is uninformative).
    """
    x = np.asarray(x, float)
    lay = np.asarray(layers)
    ok = np.isfinite(x)
    x, lay = x[ok], lay[ok]
    ids = np.unique(lay)
    m = np.array([x[lay == g].mean() for g in ids])
    G = len(m)
    rho, est = kendall_rho(m - m.mean()) if G >= 3 else (np.nan, False)
    k = max(1.5, _ar1_neff(G, rho if np.isfinite(rho) else 0.0)) if G >= 3 else float(G)
    s2 = m.var(ddof=1) if G > 1 else np.nan
    sigma2 = s2 * (G - 1) / (G * (1 - 1 / k)) if G > 1 else np.nan
    se = float(np.sqrt(sigma2 / k)) if G > 1 else np.nan
    t = stats.t.ppf(0.5 + confidence / 2, max(0.5, k - 1))
    mean = float(x.mean())
    ok = bool(est and G >= 5 and k >= 3)        # same rule as mean_ci_neff
    if not ok:
        se = np.nan
    return {"mean": mean, "se": se, "ci_low": mean - t * se, "ci_high": mean + t * se, "n_layers": G,
            "n_eff_layers": float(k), "rho_layers": rho, "rho_estimable": est, "estimable": ok,
            "layer_mean_min": float(m.min()), "layer_mean_max": float(m.max()), "dof": float(k - 1),
            "method": "z-layer means, AR(1) effective size (Kendall-corrected rho capped at 0.95), t interval"}


def layer_mean_contrast(a, b, layers_a, layers_b, confidence: float = 0.95) -> dict:
    """Difference of specimen means a - b from layer-mean summaries (Welch-Satterthwaite).

    Approximate: in simulations with 8-13 layers its size was 0.02-0.07 up to
    rho = 0.5 and 0.11-0.17 at rho = 0.7-0.85, so p-values are indicative.
    `ranges_disjoint` reports whether the ranges of the layer means of the two
    specimens overlap at all, a descriptive statement free of any model.
    """
    sa = layer_mean_summary(a, layers_a, confidence)
    sb = layer_mean_summary(b, layers_b, confidence)
    d = sa["mean"] - sb["mean"]
    if not (sa["estimable"] and sb["estimable"]):
        disjoint = bool(sa["layer_mean_max"] < sb["layer_mean_min"] or sb["layer_mean_max"] < sa["layer_mean_min"])
        return {"difference": float(d), "se": np.nan, "df": np.nan, "diff_ci_low": np.nan, "diff_ci_high": np.nan,
                "p_approx": np.nan, "estimable": False,
                "ratio": float(sa["mean"] / sb["mean"]) if sb["mean"] != 0 else np.nan,
                "ranges_disjoint": disjoint, "a": sa, "b": sb,
                "method": "not estimable: layer means too persistent (trend) for an AR(1) effective size"}
    se = float(np.hypot(sa["se"], sb["se"]))
    na, nb = sa["n_eff_layers"], sb["n_eff_layers"]
    den = sa["se"] ** 4 / max(na - 1, 0.5) + sb["se"] ** 4 / max(nb - 1, 0.5)
    df = float(se**4 / den) if den > 0 else np.nan
    t = stats.t.ppf(0.5 + confidence / 2, df) if np.isfinite(df) else np.nan
    p = float(2 * stats.t.sf(abs(d / se), df)) if se > 0 and np.isfinite(df) else np.nan
    disjoint = bool(sa["layer_mean_max"] < sb["layer_mean_min"] or sb["layer_mean_max"] < sa["layer_mean_min"])
    return {"difference": float(d), "se": se, "df": df, "diff_ci_low": float(d - t * se),
            "diff_ci_high": float(d + t * se), "p_approx": p, "estimable": True, "ratio": float(sa["mean"] / sb["mean"])
            if sb["mean"] != 0 else np.nan, "ranges_disjoint": disjoint, "a": sa, "b": sb,
            "method": "Welch-Satterthwaite on z-layer means with AR(1) effective sizes (approximate)"}


def ancova_descriptive(df, response: str, covariate: str, levels) -> dict:
    """Porosity-adjusted contrast a - b of `response` (descriptive, no p-value).

    OLS of response on sample, covariate and their interaction over all
    sub-volumes, covariate centred at its pooled mean; the contrast is also
    given at the 5th percentile, middle and 95th percentile of the covariate
    range common to both specimens, which matters when the slopes differ.
    With only a few effectively independent layers per specimen, no valid
    sampling interval is available, so none is reported.
    """
    d = df[[response, covariate, "sample"]].dropna()
    a_lv, b_lv = levels
    y = d[response].to_numpy(float)
    cov = d[covariate].to_numpy(float)
    dummy = (d["sample"] == a_lv).to_numpy(float)
    centre = float(cov.mean())
    c = cov - centre
    X = np.column_stack([np.ones_like(c), dummy, c, dummy * c])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    r2 = float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    ca, cb = cov[dummy == 1], cov[dummy == 0]
    lo, hi = max(np.quantile(ca, 0.05), np.quantile(cb, 0.05)), min(np.quantile(ca, 0.95), np.quantile(cb, 0.95))
    at = [float(v) for v in (lo, 0.5 * (lo + hi), hi)] if lo < hi else []
    slope_a, slope_b = beta[2] + beta[3], beta[2]
    return {"adjusted_difference": float(beta[1]), "at_covariate": centre, "slope_difference": float(beta[3]),
            "slope_a": float(slope_a), "slope_b": float(slope_b), "r2": r2,
            "contrast_over_common_support": [{"covariate": x0, "difference": float(beta[1] + beta[3] * (x0 - centre))}
                                             for x0 in at],
            "common_support": [float(lo), float(hi)] if lo < hi else None,
            "levels": [str(a_lv), str(b_lv)], "contrast": f"{a_lv} − {b_lv}",
            "n": {str(a_lv): int(dummy.sum()), str(b_lv): int((1 - dummy).sum())},
            "method": "OLS over sub-volumes; descriptive (no sampling interval)"}


def association_test(x, y, n_surr: int = 2000, seed: int = 0) -> dict:
    """Spearman rho(x, y) with a surrogate p-value for strongly autocorrelated series.

    x is replaced by phase-randomised surrogates of its mirrored series (same
    power spectrum, no relation to y) while y is held fixed;
    p = (#{|rho*| >= |rho|} + 1)/(B + 1). The class is "negative" or
    "positive" when p < 0.05 and "inconclusive" otherwise.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rho = float(stats.spearmanr(x, y).statistic)
    sur = _phase_surrogates(x - x.mean(), n_surr, rng)
    ry = stats.rankdata(y)
    ry = (ry - ry.mean()) / ry.std()
    rs = np.apply_along_axis(stats.rankdata, 1, sur)
    rs = (rs - rs.mean(axis=1, keepdims=True)) / rs.std(axis=1, keepdims=True)
    rho_null = (rs * ry).mean(axis=1)
    p = float((np.sum(np.abs(rho_null) >= abs(rho)) + 1) / (n_surr + 1))
    cls = "inconclusive" if p >= 0.05 else ("negative" if rho < 0 else "positive")
    return {"rho": rho, "p_surrogate": p, "association_class": cls, "n_surrogates": n_surr,
            "null_q025": float(np.quantile(rho_null, 0.025)), "null_q975": float(np.quantile(rho_null, 0.975))}


def p_floor(n_boot: int) -> float:
    """Smallest two-sided p that boot_p can return with n_boot resamples."""
    return 2.0 / (n_boot + 1)


# ---------------------------------------------------------------------------
# Resampling of spatially ordered clusters (z-layers of sub-volumes)
# ---------------------------------------------------------------------------
class LayerBlocks:
    """Moving-block resampler over z-ordered layers of sub-volumes.

    Sub-volumes of one layer are correlated with each other, and layer means
    are correlated along z (gradients, lag-1 correlation 0.6-0.9 here), so
    drawing single layers i.i.d. under-covers. Blocks of `block` consecutive
    layers are drawn with wrap-around (circular moving blocks; Politis &
    Romano 1992) until as many layers as observed are collected. The block
    length follows the decorrelation lag of the layer means, clipped to 2-4
    layers. With 8-13 layers per specimen the intervals remain approximate.
    """

    def __init__(self, layers, x=None, block: int | None = None):
        layers = np.asarray(layers)
        self.ids = np.unique(layers)
        self.members = [np.nonzero(layers == g)[0] for g in self.ids]
        self.G = len(self.ids)
        if block is None:
            if x is not None and self.G >= 4:
                xm = np.asarray(x, float)
                means = np.array([np.nanmean(xm[m]) for m in self.members])
                lag = effective_n(means)["decorrelation_lag"]
            else:
                lag = 2
            block = int(np.clip(max(2, lag), 2, max(2, min(4, self.G // 2))))
        self.block = int(max(1, min(block, self.G)))

    def draw(self, rng) -> np.ndarray:
        n_blocks = int(np.ceil(self.G / self.block))
        starts = rng.integers(0, self.G, n_blocks)
        pos = ((starts[:, None] + np.arange(self.block)[None, :]) % self.G).ravel()[: self.G]
        return np.concatenate([self.members[p] for p in pos])


def block_bootstrap_mean(x, layers, n_boot: int = 5000, confidence: float = 0.95, seed: int = 0,
                         block: int | None = None) -> dict:
    """Percentile CI of the mean of sub-volume values, resampling layer blocks."""
    x = np.asarray(x, float)
    lb = LayerBlocks(layers, x, block)
    rng = np.random.default_rng(seed)
    bs = np.array([x[lb.draw(rng)].mean() for _ in range(n_boot)])
    a = (1 - confidence) / 2
    return {"mean": float(x.mean()), "ci_low": float(np.quantile(bs, a)), "ci_high": float(np.quantile(bs, 1 - a)),
            "se_boot": float(bs.std(ddof=1)), "n_layers": lb.G, "block_layers": lb.block,
            "method": "moving-block bootstrap of consecutive z-layers"}


def _mad(v):
    return float(np.mean(np.abs(v - np.median(v))))


def cluster_bootstrap_diff(a, b, groups_a, groups_b, statistic=np.mean, n_boot: int = 5000,
                           confidence: float = 0.95, seed: int = 0) -> dict:
    """Bootstrap CI of statistic(a) - statistic(b) resampling blocks of z-layers.

    Each specimen is resampled on its own with `LayerBlocks`. The same draws
    give a test of equal spread, log(MAD_a / MAD_b) with MAD the mean absolute
    deviation from the median.
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    la, lb = LayerBlocks(groups_a, a), LayerBlocks(groups_b, b)
    diffs, ratios, spread = np.empty(n_boot), np.empty(n_boot), np.empty(n_boot)
    for k in range(n_boot):
        sa, sb = a[la.draw(rng)], b[lb.draw(rng)]
        ta, tb = statistic(sa), statistic(sb)
        diffs[k] = ta - tb
        ratios[k] = ta / tb if tb != 0 else np.nan
        ma, mb = _mad(sa), _mad(sb)
        spread[k] = np.log(ma / mb) if ma > 0 and mb > 0 else np.nan
    alpha = (1 - confidence) / 2
    est = statistic(a) - statistic(b)
    return {"difference": float(est),
            "diff_ci_low": float(np.quantile(diffs, alpha)), "diff_ci_high": float(np.quantile(diffs, 1 - alpha)),
            "ratio": float(statistic(a) / statistic(b)),
            "ratio_ci_low": float(np.nanquantile(ratios, alpha)), "ratio_ci_high": float(np.nanquantile(ratios, 1 - alpha)),
            "p_boot_two_sided": boot_p(diffs), "p_floor": p_floor(n_boot),
            "log_mad_ratio": float(np.log(_mad(a) / _mad(b))) if _mad(a) > 0 and _mad(b) > 0 else np.nan,
            "p_spread_boot": boot_p(spread),
            "n_clusters_a": la.G, "n_clusters_b": lb.G, "block_layers_a": la.block, "block_layers_b": lb.block,
            "method": "moving-block bootstrap of consecutive z-layers within each specimen"}


def compare_two(a, b, groups_a=None, groups_b=None, n_boot: int = 5000, confidence: float = 0.95,
                seed: int = 0) -> dict:
    """Effect sizes (descriptive) and layer-block bootstrap contrast of a - b.

    The i.i.d. tests (Welch, Brunner-Munzel, Brown-Forsythe) treat the
    sub-volumes as independent; they are kept only for reference under
    `iid_reference_anticonservative` and are not reported.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    fa, fb = np.isfinite(a), np.isfinite(b)
    a, b = a[fa], b[fb]
    out = {
        "n_a": int(len(a)), "n_b": int(len(b)),
        "hedges_g": hedges_g(a, b),
        "cliffs_delta": cliffs_delta(a, b),
        "iid_reference_anticonservative": {
            "welch_p": float(stats.ttest_ind(a, b, equal_var=False).pvalue),
            "brunner_munzel_p": float(stats.brunnermunzel(a, b).pvalue),
            "brown_forsythe_p": float(stats.levene(a, b, center="median").pvalue)},
    }
    if groups_a is not None:
        out.update(cluster_bootstrap_diff(a, b, np.asarray(groups_a)[fa], np.asarray(groups_b)[fb],
                                          n_boot=n_boot, confidence=confidence, seed=seed))
    return out


def _ancova_coefs(y, dummy, cov, centre):
    """OLS of y on [1, dummy, cov_c, dummy*cov_c] with cov_c = cov - centre.

    Returns (difference at the covariate value `centre`, slope difference).
    """
    c = cov - centre
    X = np.column_stack([np.ones_like(c), dummy, c, dummy * c])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta[1], beta[3]


def ancova_sample_effect(df, response: str, covariate: str, cluster: str, levels, n_boot: int = 2000,
                         confidence: float = 0.95, seed: int = 0) -> dict:
    """Difference levels[0] - levels[1] in `response` adjusted for `covariate`.

    Model: response ~ sample * covariate, with the covariate centred at its
    pooled mean, so the sample coefficient is the contrast at the mean
    covariate. Uncertainty: moving-block bootstrap of consecutive z-layers
    within each specimen. When the slopes differ, a single adjusted
    difference hides how the contrast changes with the covariate, so the
    contrast is also evaluated at the 5th percentile, the median and the 95th
    percentile of the covariate range common to both specimens (never
    outside it). The cluster-robust (sandwich) p-value is kept for reference;
    with ~20 clusters it is anti-conservative.
    """
    import statsmodels.formula.api as smf

    d = df[[response, covariate, "sample", cluster]].dropna().copy()
    d["cov_c"] = d[covariate] - d[covariate].mean()
    groups = d["sample"].astype(str) + "_" + d[cluster].astype(str)
    fit = smf.ols(f"{response} ~ C(sample) * cov_c", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": groups.astype("category").cat.codes})
    name = [p for p in fit.params.index if p.startswith("C(sample)") and ":" not in p][0]
    inter = [p for p in fit.params.index if ":" in p][0]
    a_lv, b_lv = levels
    y = d[response].to_numpy(float)
    cov = d[covariate].to_numpy(float)
    dummy = (d["sample"] == a_lv).to_numpy(float)          # coefficient = a - b
    centre = float(cov.mean())
    # common support of the covariate
    ca, cb = cov[dummy == 1], cov[dummy == 0]
    lo, hi = max(np.quantile(ca, 0.05), np.quantile(cb, 0.05)), min(np.quantile(ca, 0.95), np.quantile(cb, 0.95))
    at = [float(v) for v in (lo, 0.5 * (lo + hi), hi)] if lo < hi else []
    est, est_slope = _ancova_coefs(y, dummy, cov, centre)
    rng = np.random.default_rng(seed)
    blk = {lv: LayerBlocks(d.loc[d["sample"] == lv, cluster].to_numpy(), d.loc[d["sample"] == lv, response].to_numpy())
           for lv in (a_lv, b_lv)}
    pos = {lv: np.nonzero((d["sample"] == lv).to_numpy())[0] for lv in (a_lv, b_lv)}
    bd, bs = np.empty(n_boot), np.empty(n_boot)
    for k in range(n_boot):
        sel = np.concatenate([pos[lv][blk[lv].draw(rng)] for lv in (a_lv, b_lv)])
        bd[k], bs[k] = _ancova_coefs(y[sel], dummy[sel], cov[sel], centre)
    al = (1 - confidence) / 2
    at_rows = []
    for x0 in at:
        draws = bd + bs * (x0 - centre)
        at_rows.append({"covariate": x0, "difference": float(est + est_slope * (x0 - centre)),
                        "ci_low": float(np.quantile(draws, al)), "ci_high": float(np.quantile(draws, 1 - al)),
                        "p_boot": boot_p(draws)})
    slope_ci = [float(np.quantile(bs, al)), float(np.quantile(bs, 1 - al))]
    return {"adjusted_difference": float(est), "at_covariate": centre, "ci_low": float(np.quantile(bd, al)),
            "ci_high": float(np.quantile(bd, 1 - al)), "p_boot": boot_p(bd), "p_floor": p_floor(n_boot),
            "slope_difference": float(est_slope), "slope_ci": slope_ci,
            "slope_difference_p_boot": boot_p(bs),
            "interaction": bool(slope_ci[0] > 0 or slope_ci[1] < 0),
            "contrast_over_common_support": at_rows,
            "p_cluster_robust": float(fit.pvalues[name]), "slope_p_cluster_robust": float(fit.pvalues[inter]),
            "r2": float(fit.rsquared), "levels": [str(a_lv), str(b_lv)], "contrast": f"{a_lv} − {b_lv}",
            "n_clusters": {lv: blk[lv].G for lv in (a_lv, b_lv)},
            "block_layers": {lv: blk[lv].block for lv in (a_lv, b_lv)}, "n_boot": n_boot,
            "n": {lv: int(len(pos[lv])) for lv in (a_lv, b_lv)}}


def spearman_ci(x, y, n_boot: int = 2000, confidence: float = 0.95, seed: int = 0, groups=None,
                block: int | None = None) -> dict:
    """Spearman rho with a bootstrap CI.

    `groups` (z-layer of each sub-volume) switches to the layer-block
    resampler; `block` alone (no groups) resamples a series with circular
    blocks of that many consecutive points.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho = stats.spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    n = len(x)
    if groups is not None:
        lb = LayerBlocks(np.asarray(groups)[ok], y)
        draw = lambda: lb.draw(rng)  # noqa: E731
    elif block:
        L = int(max(1, min(block, n)))
        nb = int(np.ceil(n / L))
        draw = lambda: ((rng.integers(0, n, nb)[:, None] + np.arange(L)[None, :]) % n).ravel()[:n]  # noqa: E731
    else:
        draw = lambda: rng.integers(0, n, n)  # noqa: E731
    bs = []
    for _ in range(n_boot):
        i = draw()
        bs.append(stats.spearmanr(x[i], y[i]).statistic)
    a = (1 - confidence) / 2
    return {"rho": float(rho), "ci_low": float(np.nanquantile(bs, a)), "ci_high": float(np.nanquantile(bs, 1 - a)),
            "resampling": "layer blocks" if groups is not None else (f"circular blocks of {block}" if block else "iid")}
