"""Integration tests for pls1_confirmatory_test(ci=True)."""
import numpy as np
import pytest

from plskit import (
    CIScalar,
    ConfirmatoryCI,
    PlsKitError,
    pls1_confirmatory_test,
)


def _synth(n=100, d=6, snr=4.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    beta = np.array([1.0] * 2 + [0.0] * (d - 2))
    y = x @ beta * snr + rng.standard_normal(n)
    return x, y


def test_ci_false_returns_no_ci_field():
    x, y = _synth()
    r = pls1_confirmatory_test(x, y, k=2, test_method="split_nb", seed=7)
    assert r.ci is None


def test_ci_none_defaults_are_recorded_as_resolved_values():
    """None (the public default for n_boot/m_rate/level) must resolve to the
    engine's own default and be recorded on result.ci as that resolved
    value, never as None itself."""
    x, y = _synth()
    r = pls1_confirmatory_test(x, y, k=2, test_method="split_nb", ci=True, seed=7)
    assert r.ci.n_boot == 1000
    assert r.ci.m_rate == pytest.approx(0.7)
    assert r.ci.level == pytest.approx(0.95)
    assert isinstance(r.ci, ConfirmatoryCI)
    assert isinstance(r.ci.holdout_corr, CIScalar)


@pytest.mark.parametrize("bad_m_rate", [0.4, 0.5, 0.95, 1.0])
def test_ci_rejects_out_of_range_m_rate(bad_m_rate):
    x, y = _synth()
    with pytest.raises(PlsKitError) as excinfo:
        pls1_confirmatory_test(x, y, k=2, test_method="split_nb",
                               ci=True, n_boot=200, m_rate=bad_m_rate, seed=7)
    assert excinfo.value.code == "invalid_argument"


@pytest.mark.parametrize("bad_level", [0.49, 0.991, 1.0])
def test_ci_rejects_out_of_range_level(bad_level):
    x, y = _synth()
    with pytest.raises(PlsKitError) as excinfo:
        pls1_confirmatory_test(x, y, k=2, test_method="split_nb",
                               ci=True, n_boot=200, level=bad_level, seed=7)
    assert excinfo.value.code == "invalid_argument"


def test_ci_rejects_m_less_than_k_plus_2():
    # n=20, k=4, m_rate=0.51 → m = ceil(20^0.51) = 5 < k+2 = 6
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 6))
    y = rng.standard_normal(20)
    with pytest.raises(PlsKitError) as excinfo:
        pls1_confirmatory_test(x, y, k=4, test_method="split_nb",
                               ci=True, n_boot=200, m_rate=0.51, seed=7)
    assert excinfo.value.code == "invalid_argument"


def test_ci_max_failure_rate_validated():
    """Range check: max_failure_rate ∈ [0.0, 1.0]."""
    x, y = _synth()
    for bad in (-0.01, 1.01, 2.0):
        with pytest.raises(PlsKitError) as excinfo:
            pls1_confirmatory_test(
                x, y, k=2, test_method="split_nb",
                ci=True, n_boot=200, seed=7,
                max_failure_rate=bad,
            )
        assert excinfo.value.code == "invalid_argument", (
            f"max_failure_rate={bad}: expected invalid_argument, got {excinfo.value.code}"
        )
