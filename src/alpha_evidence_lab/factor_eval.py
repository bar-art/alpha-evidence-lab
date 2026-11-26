"""Factor diagnostics. Future exit eligibility is a conditional research filter."""

from datetime import date
from typing import Any

import numpy as np

from .factor_dsl import _cs_rank
from .panel import Panel


def forward_returns(panel: Panel, horizon: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """At close t, enter open t+1 and exit open t+1+h. Not an executable portfolio."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    shape = panel.fields["open"].shape
    returns = np.full(shape, np.nan)
    allowed = np.zeros(shape, dtype=bool)
    count = {
        key: 0
        for key in [
            "eligible_decisions",
            "entry_unobserved",
            "exit_unobserved",
            "entry_limit_up",
            "exit_limit_down",
            "unknown_limits",
        ]
    }
    length = len(panel.dates) - horizon - 1
    if length <= 0:
        return returns, allowed, count
    signal = panel.member[:length] & panel.observed[:length]
    entry, exit_ = slice(1, length + 1), slice(horizon + 1, horizon + length + 1)
    opens = panel.fields["open"]
    entry_known = panel.limit_exempt[entry] | (
        np.isfinite(panel.up_limit[entry]) & np.isfinite(panel.down_limit[entry])
    )
    exit_known = panel.limit_exempt[exit_] | (
        np.isfinite(panel.up_limit[exit_]) & np.isfinite(panel.down_limit[exit_])
    )
    up = ~panel.limit_exempt[entry] & (opens[entry] >= panel.up_limit[entry] - 1e-8)
    down = ~panel.limit_exempt[exit_] & (opens[exit_] <= panel.down_limit[exit_] + 1e-8)
    valid = (
        signal
        & panel.observed[entry]
        & panel.observed[exit_]
        & entry_known
        & exit_known
        & ~up
        & ~down
    )
    raw = opens[exit_] / opens[entry] - 1
    allowed[:length] = valid & np.isfinite(raw)
    returns[:length] = np.where(allowed[:length], raw, np.nan)
    count.update(
        eligible_decisions=int(signal.sum()),
        entry_unobserved=int((signal & ~panel.observed[entry]).sum()),
        exit_unobserved=int((signal & ~panel.observed[exit_]).sum()),
        entry_limit_up=int((signal & up).sum()),
        exit_limit_down=int((signal & down).sum()),
        unknown_limits=int((signal & ~(entry_known & exit_known)).sum()),
    )
    return returns, allowed, count


def rank_ic(
    values: np.ndarray, returns: np.ndarray, eligible: np.ndarray, minimum: int = 30
) -> np.ndarray:
    if values.shape != returns.shape or eligible.shape != values.shape or minimum < 2:
        raise ValueError("Invalid Rank IC input")
    mask = eligible & np.isfinite(values) & np.isfinite(returns)
    x, y = _cs_rank(values, mask), _cs_rank(returns, mask)
    result = np.full(len(values), np.nan)
    for t in range(len(values)):
        valid = mask[t]
        if int(valid.sum()) < minimum:
            continue
        a, b = x[t, valid], y[t, valid]
        a, b = a - a.mean(), b - b.mean()
        denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
        if denominator > 1e-12:
            result[t] = float((a * b).sum() / denominator)
    return result


def ic_statistics(ic: np.ndarray, start: int, stop: int, horizon: int) -> dict[str, Any]:
    if not 0 <= start < stop <= len(ic) or horizon < 1:
        raise ValueError("Invalid IC sampling interval")
    sampled = ic[np.arange(start, stop, horizon)]
    sampled = sampled[np.isfinite(sampled)]
    mean = float(sampled.mean()) if len(sampled) else None
    std = float(sampled.std(ddof=1)) if len(sampled) > 1 else None
    return {
        "ic_mean": mean,
        "ic_std": std,
        "icir": mean / std if mean is not None and std is not None and std > 1e-12 else None,
        "ic_observations": len(sampled),
    }


def portfolio_metrics(
    panel: Panel,
    values: np.ndarray,
    eligible: np.ndarray,
    start: int,
    stop: int,
    horizon: int,
    cost_bps: float,
    stamp_duty: tuple[tuple[date, float], ...] = (),
    minimum: int = 30,
) -> dict[str, Any]:
    """Unit long + unit short per reset NAV, marked at opens; four cash-flow costs.

    Future exit filtering makes this conditional diagnostics, not executable P&L.
    """
    from .evaluator import stamp_duty_rate

    cost = cost_bps / 10000
    nav, net, exclusions = [1.0], [], 0
    for t in range(start, stop, horizon):
        valid = eligible[t] & np.isfinite(values[t])
        names = np.flatnonzero(valid)
        if len(names) < minimum:
            exclusions += 1
            continue
        order = names[np.argsort(values[t, names], kind="stable")]
        size = max(1, int(np.floor(len(order) * 0.2)))
        if size * 2 > len(order):
            exclusions += 1
            continue
        shorts, longs = order[:size], order[-size:]
        entry, exit_ = t + 1, t + 1 + horizon
        if exit_ >= len(panel.dates):
            exclusions += 1
            continue
        opening = panel.fields["open"]
        entry_tax = stamp_duty_rate(stamp_duty, panel.dates[entry].astype(object))
        exit_tax = stamp_duty_rate(stamp_duty, panel.dates[exit_].astype(object))
        base = nav[-1]
        entry_fee = 2 * cost + entry_tax
        nav.append(base * (1 - entry_fee))
        before = len(nav)
        for day in range(entry + 1, exit_ + 1):
            long_ratio = float(np.mean(opening[day, longs] / opening[entry, longs]))
            short_ratio = float(np.mean(opening[day, shorts] / opening[entry, shorts]))
            value = 1 + long_ratio - short_ratio - entry_fee
            if day == exit_:
                value -= (cost + exit_tax) * long_ratio + cost * short_ratio
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Missing valuation or nonpositive long-short NAV")
            nav.append(base * value)
        assert len(nav) > before
        net.append(nav[-1] / base - 1)
    if not net:
        raise ValueError("No complete tradable non-overlapping baskets")
    curve = np.asarray(nav)
    return {
        "return": float(curve[-1] - 1),
        "drawdown": float(np.max(1 - curve / np.maximum.accumulate(curve))),
        "nav": nav,
        "period_returns": net,
        "portfolio_periods": len(net),
        "excluded_rebalance_days": exclusions,
    }


def evaluate_factor(
    panel: Panel,
    factor: Any,
    start: int,
    stop: int,
    *,
    horizon: int = 5,
    label_stop: int | None = None,
    cost_bps: float = 7,
    stamp_duty_schedule: tuple[tuple[date, float], ...] = (),
    minimum: int = 30,
) -> dict[str, Any]:
    from .factor_dsl import evaluate

    if (
        not 0 <= start < stop <= len(panel.dates)
        or not np.isfinite(cost_bps)
        or not 0 <= cost_bps <= 100
    ):
        raise ValueError("Invalid factor assessment interval or cost")
    universe = panel.member & panel.observed
    values = evaluate(factor, panel.factor_inputs(), universe)
    denominator = int(universe[start:stop].sum())
    coverage = (
        float((universe[start:stop] & np.isfinite(values[start:stop])).sum() / denominator)
        if denominator
        else 0.0
    )
    if coverage < 0.7:
        raise ValueError(f"Factor coverage below 70%: {coverage:.4f}")
    returns, eligible, exclusions = forward_returns(panel, horizon)
    boundary = min(label_stop if label_stop is not None else len(panel.dates), len(panel.dates))
    eligible[np.arange(len(panel.dates)) + horizon + 1 >= boundary] = False
    ic = rank_ic(values, returns, eligible, minimum)
    metrics = portfolio_metrics(
        panel, values, eligible, start, stop, horizon, cost_bps, stamp_duty_schedule, minimum
    )
    return {
        **metrics,
        **ic_statistics(ic, start, stop, horizon),
        "coverage": coverage,
        "daily_ic": [float(x) if np.isfinite(x) else None for x in ic[start:stop]],
        "exclusions": exclusions,
        "label_stop_exclusive": boundary,
        "metric_kind": "conditional_long_short_gross_2_open_marked_research",
    }
