from dataclasses import replace

import numpy as np
import pytest

from alpha_evidence_lab.factor_eval import forward_returns, ic_statistics, rank_ic
from alpha_evidence_lab.panel import Panel


def panel(days=15, stocks=5):
    opening = 10 + np.arange(days)[:, None] * np.arange(1, stocks + 1)[None, :] * 0.1
    shape = opening.shape
    return Panel(
        np.arange("2024-01-01", np.datetime64("2024-01-01") + days, dtype="datetime64[D]"),
        tuple(f"S{i}" for i in range(stocks)),
        {
            **{name: opening.copy() for name in ["open", "high", "low", "close"]},
            "volume": np.ones(shape) * 100,
        },
        np.zeros(shape, bool),
        np.ones(shape, bool),
        np.ones(shape) * 100,
        np.ones(shape),
        np.zeros(shape, bool),
        "fixture",
        "offline fixture",
    )


def test_forward_timing_and_masks():
    p = panel()
    r, valid, count = forward_returns(p, 2)
    assert r[0, 0] == pytest.approx(p.fields["open"][3, 0] / p.fields["open"][1, 0] - 1)
    assert np.isnan(r[-3:]).all()  # entry eligibility survives unknown future exits
    p.up_limit[1, 0] = p.fields["open"][1, 0]
    p.down_limit[3, 1] = p.fields["open"][3, 1]
    p.suspended[1, 2] = True
    p.up_limit[1, 3] = np.nan
    r, valid, count = forward_returns(p, 2)
    assert not valid[0, [0, 2, 3]].any() and valid[0, 1] and valid[0, 4]
    assert r[0, 1] == pytest.approx(p.fields["open"][4, 1] / p.fields["open"][1, 1] - 1)
    assert count["entry_limit_up"] > 0
    assert count["entry_unobserved"] > 0 and count["unknown_limits"] > 0
    p.limit_exempt[:, :] = True
    assert forward_returns(p, 2)[1][0, 0]


def test_rank_ties_and_nonoverlap():
    x = np.array([[1.0, 1.0, 3.0, 4.0], [1, 2, 3, 4]])
    y = np.array([[4.0, 3.0, 2.0, 1.0], [1, 2, 3, 4]])
    r = rank_ic(x, y, np.ones(x.shape, bool), minimum=3)
    assert r[0] == pytest.approx(np.corrcoef([1.5, 1.5, 3, 4], [4, 3, 2, 1])[0, 1])
    assert r[1] == pytest.approx(1)
    assert np.isnan(rank_ic(x, y, np.ones(x.shape, bool))[0])
    stats = ic_statistics(np.array([1.0, 999.0, -1.0, 999.0, 0.0]), 0, 5, 2)
    assert stats["ic_observations"] == 3 and stats["ic_mean"] == 0 and stats["ic_std"] == 1
    assert stats["icir"] == 0
    assert ic_statistics(np.ones(5), 0, 5, 2)["icir"] is None


def test_future_prices_do_not_change_matured_signal():
    p = panel()
    before = forward_returns(p, 2)[0][0].copy()
    changed = {name: values.copy() for name, values in p.fields.items()}
    changed["open"][4:] *= 2
    after = forward_returns(replace(p, fields=changed), 2)[0][0]
    np.testing.assert_equal(before, after)
    with pytest.raises(ValueError):
        forward_returns(p, 0)


def test_long_short_costs_tax_and_open_marked_drawdown():
    from datetime import date

    from alpha_evidence_lab.factor_eval import portfolio_metrics

    p = panel(days=10, stocks=5)
    p.fields["open"][:] = 10
    p.fields["open"][2, 4] = 9  # interim loss of the long leg
    p.fields["open"][3, 4] = 11
    p.fields["open"][3, 0] = 9
    scores = np.broadcast_to(np.arange(5.0), (10, 5))
    eligible = np.zeros((10, 5), bool)
    eligible[0] = True
    result = portfolio_metrics(
        p, scores, eligible, 0, 1, 2, 10, ((date(2000, 1, 1), 5.0),), minimum=3
    )
    # long +10%, short +10%; costs .002 entry +.002 exit; taxes .0005+.00055
    assert result["return"] == pytest.approx(0.2 - 0.004 - 0.00105)
    assert result["drawdown"] == pytest.approx(0.1025)
    assert result["portfolio_periods"] == 1


def test_assessment_no_test_leak_and_tail_is_fenced():
    from alpha_evidence_lab.factor_dsl import parse
    from alpha_evidence_lab.factor_eval import evaluate_factor

    p = panel(days=25, stocks=5)
    factor = parse("cs_rank(pct_change(close, 2))")
    first = evaluate_factor(p, factor, 3, 13, horizon=2, label_stop=16, minimum=3)
    fields = {key: value.copy() for key, value in p.fields.items()}
    for name in ["open", "close", "high", "low"]:
        fields[name][16:] *= 3
    second = evaluate_factor(
        replace(p, fields=fields), factor, 3, 13, horizon=2, label_stop=16, minimum=3
    )
    assert first == second
    assert first["portfolio_periods"] == 5
    with pytest.raises(ValueError, match="coverage"):
        evaluate_factor(p, parse("ts_mean(close, 120)"), 3, 13, horizon=2, minimum=3)


def test_delayed_long_short_and_unresolved_losses():
    from alpha_evidence_lab.factor_eval import exit_labels, portfolio_metrics

    p = panel(days=20, stocks=5)
    for v in p.fields.values():
        v[:] = 10
    p.fields["open"][3, 4] = 8
    p.down_limit[3, 4] = 8
    p.fields["open"][4:, 4] = 7
    labels = exit_labels(p, 2, max_exit_delay=2)
    assert labels["eligible"][0, 4]
    assert labels["exit_day"][0, 4] == 4
    assert labels["returns"][0, 4] == pytest.approx(-0.3)
    p.fields["open"][3, 0] = 12
    p.up_limit[3, 0] = 12
    p.fields["open"][4:, 0] = 13
    short = exit_labels(p, 2, max_exit_delay=2, short=True)
    assert short["exit_day"][0, 0] == 4
    values = np.broadcast_to(np.arange(5.0), (20, 5))
    result = portfolio_metrics(
        p, values, labels["eligible"], 0, 1, 2, 0, minimum=3, max_exit_delay=2
    )
    assert result["return"] == pytest.approx(-0.6)
    assert result["delayed_positions"] == 2
    # All ten permitted delayed sessions blocked: the loss is marked, not deleted.
    p.down_limit[3:14, 4] = p.fields["open"][3:14, 4]
    p.fields["close"][13, 4] = 6
    labels = exit_labels(p, 2, max_exit_delay=10)
    assert labels["exit_unresolved"][0, 4]
    assert labels["returns"][0, 4] == pytest.approx(-0.4)
    result = portfolio_metrics(p, values, labels["eligible"], 0, 1, 2, 0, minimum=3)
    assert result["return"] == pytest.approx(-0.7)
    assert result["exit_unresolved"] == 1


def test_delayed_boundary_unknown_not_dropped_and_future_fenced():
    from alpha_evidence_lab.factor_eval import exit_labels, portfolio_metrics

    p = panel(days=20, stocks=5)
    p.suspended[3:8, 4] = True
    labels = exit_labels(p, 2, boundary=8)
    assert labels["eligible"][0, 4] and labels["boundary_unknown"][0, 4]
    values = np.broadcast_to(np.arange(5.0), (20, 5))
    first = portfolio_metrics(p, values, labels["eligible"], 0, 1, 2, 0, minimum=3, boundary=8)
    assert first["return"] is None and first["boundary_unknown_positions"] == 1
    for field in p.fields.values():
        field[8:] *= 99
    after = exit_labels(p, 2, boundary=8)
    for k in labels:
        np.testing.assert_equal(labels[k][:8], after[k][:8])
    assert first == portfolio_metrics(
        p, values, after["eligible"], 0, 1, 2, 0, minimum=3, boundary=8
    )


def test_scoring_counters_exclude_warmup_and_missing_ic_is_invalid():
    from alpha_evidence_lab.factor_dsl import parse
    from alpha_evidence_lab.factor_eval import evaluate_factor

    p = panel(days=30, stocks=5)
    p.up_limit[1, 0] = p.fields["open"][1, 0]  # warmup only
    result = evaluate_factor(p, parse("cs_rank(close)"), 5, 15, horizon=2, minimum=3)
    assert result["exclusions"]["entry_limit_up"] == 0
    assert result["counter_interval"]["scope"] == "scoring_period_only"
    assert result["ic_observations"] == 5 and result["ic_valid_days"] == 10
    assert ic_statistics(np.full(10, np.nan), 0, 10, 2)["ic_mean"] is None
    with pytest.raises(ValueError, match="No valid.*IC"):
        evaluate_factor(panel(days=30, stocks=1), parse("close"), 5, 15, minimum=3)
    p.suspended[:] = True
    with pytest.raises(ValueError, match="coverage"):
        evaluate_factor(p, parse("close"), 5, 15, minimum=3)


def test_precomputed_exit_labels_match_uncached_and_actual_exit_tax():
    from datetime import date

    from alpha_evidence_lab.factor_dsl import parse
    from alpha_evidence_lab.factor_eval import evaluate_factor, exit_labels, portfolio_metrics

    p = panel(days=25, stocks=5)
    labels = tuple(exit_labels(p, 2, short=s) for s in (False, True))
    args = (p, parse("cs_rank(close)"), 5, 15)
    assert evaluate_factor(*args, horizon=2, minimum=3) == evaluate_factor(
        *args, horizon=2, minimum=3, _labels=labels
    )
    for field in p.fields.values():
        field[:] = 10
    p.down_limit[3, 4] = 10
    p.fields["open"][4:, 4] = 8
    values = np.broadcast_to(np.arange(5.0), (25, 5))
    eligible = exit_labels(p, 2)["eligible"]
    result = portfolio_metrics(
        p,
        values,
        eligible,
        0,
        1,
        2,
        0,
        ((date(2024, 1, 1), 10.0), (date(2024, 1, 5), 5.0)),
        minimum=3,
    )
    # Short-entry tax .001; actual long exit Jan5 at .8 notional taxed .0004.
    assert result["return"] == pytest.approx(-0.2 - 0.001 - 0.0004)
