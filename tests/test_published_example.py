"""Deflated Sharpe ratio: numerical example in Bailey and Lopez de Prado (2014), The Deflated Sharpe Ratio, JPM.
Annualised SR 2.5 over T = 1,250 daily observations, skewness -3, kurtosis 10, N = 100 trials with annualised
cross-trial Sharpe variance 0.5. The paper reports DSR = 0.9004."""
import math

from overfit.sharpe import expected_max_sr, psr


def test_dsr_matches_published_example():
    sr = 2.5 / math.sqrt(250)
    sr0 = expected_max_sr(100, 0.5 / 250)
    assert round(psr(sr, sr0, 1250, -3, 10), 4) == 0.9004
