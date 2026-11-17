import math

import numpy as np
import pytest
from scipy.stats import norm

from overfit import synthetic as syn
from overfit.cscv import cscv, cscv_slow, split_matrix
from overfit.haircut import adjust, haircut_sharpe, haircut_table, pvalues
from overfit.sharpe import (PERIODS_PER_YEAR, dsr, expected_max_sr, min_track_record_length, moments, psr, sharpe,
                            sr_std_error)
from overfit.spa import spa_tests, stationary_bootstrap_counts, stationary_bootstrap_indices
from overfit.trials import cluster_effective_n, eigen_effective_n, correlation

SEEDS = [0, 1, 2, 3, 4]


# ---- CSCV -------------------------------------------------------------------------------------
def test_split_count():
    assert split_matrix(16).shape == (12870, 16)
    assert split_matrix(8).shape == (70, 8)
    assert (split_matrix(16).sum(1) == 8).all()


@pytest.mark.parametrize("seed", SEEDS)
def test_cscv_matches_slow_loop_exactly(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.002, 0.04, (73, 12))  # T not divisible by S: leading remainder is dropped
    fast, slow = cscv(x, s=8), cscv_slow(x, s=8)
    np.testing.assert_allclose(fast.omega, slow["omega"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(fast.logit, slow["logit"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(fast.is_sr, slow["is_sr"], rtol=1e-10)
    np.testing.assert_allclose(fast.oos_sr, slow["oos_sr"], rtol=1e-10)
    assert fast.pbo == slow["pbo"]


def test_cscv_joint_density_normalised():
    x = np.random.default_rng(0).normal(0, 0.04, (96, 30))
    r = cscv(x, s=8, joint_bins=10)
    assert r.joint.shape == (10, 10) and abs(r.joint.sum() - 1) < 1e-12


@pytest.mark.parametrize("seed", SEEDS)
def test_noise_pbo_about_half_and_dsr_rejects(seed):
    rng = np.random.default_rng(seed)
    x = syn.noise(249, 300, rng)
    res = cscv(x, s=16)
    assert res.pbo >= 0.4
    sr = x.mean(0) / x.std(0, ddof=1)
    best = int(sr.argmax())
    assert dsr(x[:, best], sr)[0] < 0.95


@pytest.mark.parametrize("seed", SEEDS)
def test_planted_signal_detected(seed):
    # With 249 monthly periods and 300 trials a planted annual Sharpe of 2.0 is reliably identified.
    # (At Sharpe 1.0 the best noise trial still beats the planted one in-sample in roughly half of the splits.)
    rng = np.random.default_rng(seed)
    x = syn.noise_plus_signal(249, 300, 2.0, rng)
    res = cscv(x, s=16)
    assert res.pbo < 0.1
    sr = x.mean(0) / x.std(0, ddof=1)
    assert int(sr.argmax()) == 0
    assert dsr(x[:, 0], sr)[0] > 0.95


def test_pbo_decreases_with_planted_sharpe():
    mean_pbo = []
    for sr in (0.0, 1.0, 2.0):
        vals = [cscv(syn.noise_plus_signal(249, 300, sr, np.random.default_rng(s)), s=16).pbo for s in SEEDS]
        mean_pbo.append(np.mean(vals))
    assert mean_pbo[0] > 0.4 and mean_pbo[0] > mean_pbo[1] > mean_pbo[2]
    assert mean_pbo[1] < 0.35


# ---- PSR / DSR --------------------------------------------------------------------------------
def test_psr_closed_form():
    # sr=0.2, benchmark 0.1, n=101, normal: standard error = sqrt((1 + 0.5*0.04)/100)
    se = math.sqrt((1 + 0.5 * 0.04) / 100)
    assert sr_std_error(0.2, 101) == pytest.approx(se, abs=1e-12)
    assert psr(0.2, 0.1, 101) == pytest.approx(norm.cdf(0.1 / se), abs=1e-12)
    assert psr(0.1, 0.1, 50) == pytest.approx(0.5)


def test_psr_skew_kurtosis_adjustment_direction():
    base = psr(0.3, 0.0, 60)
    assert psr(0.3, 0.0, 60, skew_=-1.0, kurt=6.0) < base  # negative skew and fat tails lower confidence


def test_expected_max_matches_monte_carlo():
    rng = np.random.default_rng(0)
    for n in (10, 100, 1000):
        mc = rng.standard_normal((4000, n)).max(1).mean()
        assert expected_max_sr(n, 1.0) == pytest.approx(mc, abs=0.06)


def test_expected_max_scaling():
    assert expected_max_sr(100, 4.0) == pytest.approx(2 * expected_max_sr(100, 1.0))
    assert expected_max_sr(1, 1.0) == 0.0


def test_dsr_matches_independent_reference():
    # Reference transcribed from the research repository's deflated_sharpe(): same formula, different code path.
    from scipy.stats import kurtosis, skew
    rng = np.random.default_rng(3)
    x = rng.normal(0.01, 0.05, (249, 40))
    srs = x.mean(0) / x.std(0, ddof=1)
    r = x[:, int(srs.argmax())]
    n, sr = len(srs), r.mean() / r.std(ddof=1)
    g = 0.5772156649
    sr0 = math.sqrt(np.var(srs, ddof=1)) * ((1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * math.e)))
    den = math.sqrt(1 - skew(r) * sr + (kurtosis(r, fisher=False) - 1) / 4 * sr**2)
    ref = norm.cdf((sr - sr0) * math.sqrt(len(r) - 1) / den)
    got, sr0_got = dsr(r, srs)
    assert got == pytest.approx(ref, abs=1e-9) and sr0_got == pytest.approx(sr0, abs=1e-9)


def test_dsr_decreases_with_more_trials():
    rng = np.random.default_rng(1)
    x = rng.normal(0.01, 0.05, (249, 50))
    srs = x.mean(0) / x.std(0, ddof=1)
    r = x[:, int(srs.argmax())]
    assert dsr(r, srs, 10)[0] > dsr(r, srs, 50)[0] > dsr(r, srs, 5000)[0]


def test_min_track_record_length():
    # normal returns, sr=0.1 per period vs 0: n* = 1 + (1 + 0.005) * (1.6449 / 0.1)^2
    assert min_track_record_length(0.1) == pytest.approx(1 + 1.005 * (norm.ppf(0.95) / 0.1) ** 2)
    assert math.isinf(min_track_record_length(-0.1))
    n = min_track_record_length(0.2, 0.0)
    assert psr(0.2, 0.0, int(math.ceil(n))) >= 0.95 > psr(0.2, 0.0, int(n) - 2)


def test_annualised_sharpe():
    r = np.array([0.01, -0.02, 0.03, 0.0, 0.02])
    assert sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR))
    assert moments(r)[0] == pytest.approx(r.mean() / r.std(ddof=1))


# ---- haircuts ---------------------------------------------------------------------------------
def test_adjusted_pvalues_small_case():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(adjust(p, "bonferroni"), [0.04, 0.16, 0.12, 0.02])
    np.testing.assert_allclose(adjust(p, "holm"), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(adjust(p, "bh"), [0.02, 0.04, 0.04, 0.02])
    c = 1 + 1 / 2 + 1 / 3 + 1 / 4
    np.testing.assert_allclose(adjust(p, "bhy"), np.array([0.02, 0.04, 0.04, 0.02]) * c)


def test_adjusted_pvalues_ordering_and_caps():
    p = np.random.default_rng(0).uniform(0, 0.2, 200)
    b, h, y = adjust(p, "bonferroni"), adjust(p, "holm"), adjust(p, "bhy")
    assert (h <= b + 1e-12).all() and (adjust(p, "bh") <= y + 1e-12).all()
    assert max(b.max(), h.max(), y.max()) <= 1.0


def test_haircut_sharpe_and_table():
    sr, hc = haircut_sharpe(1.0, 249, 0.5, PERIODS_PER_YEAR)
    assert 0 < sr < 1.0 and hc == pytest.approx(1 - sr)
    assert haircut_sharpe(1.0, 249, 1e-9, PERIODS_PER_YEAR)[1] == pytest.approx(0.0)  # capped at no haircut
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.04, (249, 500))
    srp = x.mean(0) / x.std(0, ddof=1)
    tab = haircut_table(srp, 249, PERIODS_PER_YEAR, m_eff=50)
    assert tab["nominal_bonferroni"]["haircut"] >= tab["effective_bonferroni"]["haircut"]
    assert tab["nominal_bhy"]["haircut"] >= tab["nominal_bh"]["haircut"] if "nominal_bh" in tab else True
    assert tab["nominal_bonferroni"]["p_adj"] == pytest.approx(min(1.0, 500 * pvalues(srp, 249)[srp.argmax()]))


# ---- bootstrap / SPA ----------------------------------------------------------------------------
def test_bootstrap_shapes_and_counts():
    w = stationary_bootstrap_counts(60, 6.0, 500, np.random.default_rng(0))
    assert w.shape == (500, 60)
    assert (w.sum(1) == 60).all()
    assert abs(w.mean() - 1.0) < 1e-6


@pytest.mark.parametrize("block", [2.0, 5.0, 12.0])
def test_bootstrap_mean_block_length(block):
    t = 500
    idx = stationary_bootstrap_indices(t, block, 400, np.random.default_rng(2))
    cont = (idx[:, 1:] == (idx[:, :-1] + 1) % t).mean()
    assert cont == pytest.approx(1 - 1 / block + 1 / (block * t), abs=0.01)
    assert idx.min() >= 0 and idx.max() < t


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_spa_noise_not_rejected_signal_rejected(seed):
    rng = np.random.default_rng(seed)
    noise = syn.noise(249, 100, rng)
    r = spa_tests(noise, n_boot=4000, seed=seed)
    assert r.p_spa_consistent > 0.05 and r.p_rc > 0.05
    sig = syn.noise_plus_signal(249, 100, 2.0, rng)
    r2 = spa_tests(sig, n_boot=4000, seed=seed)
    assert r2.p_spa_consistent < 0.05 and r2.p_rc < 0.05 and r2.best_index == 0
    assert r2.p_spa_lower <= r2.p_spa_consistent + 0.01 <= r2.p_spa_upper + 0.02


def test_spa_pvalues_ordering_and_uniformity_under_null():
    ps = [spa_tests(syn.noise(120, 20, np.random.default_rng(s)), n_boot=1000, seed=s).p_rc for s in range(60)]
    assert 0.35 < np.mean(ps) < 0.75  # roughly uniform/conservative under the null


# ---- trial-set structure ---------------------------------------------------------------------------
def test_effective_n_recovers_families():
    rng = np.random.default_rng(0)
    x = syn.correlated_families(400, n_families=10, per_family=20, rho=0.9, rng=rng)
    corr = correlation(x)
    cl = cluster_effective_n(corr, thresholds=(0.5,))
    assert 8 <= cl["clusters_rho_0.5"] <= 14
    pr = eigen_effective_n(corr)["participation_ratio"]
    assert 5 <= pr <= 25
    ind = eigen_effective_n(correlation(syn.noise(400, 50, rng)))["participation_ratio"]
    assert ind > 30


def test_calibrated_effective_n():
    from overfit.trials import calibrated_effective_n
    rng = np.random.default_rng(0)
    ind = calibrated_effective_n(syn.noise(249, 200, rng), reps=300, seed=0)
    assert 100 <= ind["n_eff"] <= 200
    common = syn.correlated_families(249, 1, 200, 0.995, rng)  # one family, almost identical trials
    assert calibrated_effective_n(common, reps=300, seed=0)["n_eff"] < 60


def test_spa_close_to_arch_reference():
    arch = pytest.importorskip("arch.bootstrap")
    for sr, seed in ((0.0, 1), (0.5, 2), (1.0, 3)):
        rng = np.random.default_rng(seed)
        x = syn.noise(249, 60, rng) if sr == 0 else syn.noise_plus_signal(249, 60, sr, rng)
        ref = arch.SPA(np.zeros(249), -x, block_size=2, reps=5000, bootstrap="stationary", seed=0, studentize=True)
        ref.compute()
        mine = spa_tests(x, n_boot=20000, mean_block=2, seed=0)
        assert abs(mine.p_spa_consistent - ref.pvalues["consistent"]) < 0.06
        assert abs(mine.p_spa_lower - ref.pvalues["lower"]) < 0.06
