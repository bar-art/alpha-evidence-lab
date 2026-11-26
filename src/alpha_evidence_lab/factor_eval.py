"""Fenced factor evaluation with side-specific delayed exits."""

from datetime import date
from typing import Any

import numpy as np

from .factor_dsl import _cs_rank
from .panel import Panel


class SettlementError(ValueError):
    """A v3 date fence failed to contain an entered position; stop the study."""


def exit_labels(
    panel: Panel,
    horizon: int,
    *,
    max_exit_delay: int = 10,
    boundary: int | None = None,
    short: bool = False,
    entry_stop: int | None = None,
) -> dict[str, np.ndarray]:
    """Resolve without crossing the exclusive fence; never filter entry by future exits.

    A complete delay budget exhausted inside the fence is marked to last observed close.
    A truncated delay budget is unknown, not a zero return or a removed loss.
    """
    if horizon < 1 or max_exit_delay < 0:
        raise ValueError("Invalid holding period or exit delay")
    limit = min(boundary if boundary is not None else len(panel.dates), len(panel.dates))
    opening = panel.fields["open"]
    shape = opening.shape
    t = np.arange(len(panel.dates))[:, None]
    entry = t + 1
    nominal = entry + horizon
    entry_inside = entry < min(entry_stop if entry_stop is not None else limit, limit)
    entry_index = np.minimum(entry, max(0, limit - 1))[:, 0]
    known = panel.limit_exempt | (np.isfinite(panel.up_limit) & np.isfinite(panel.down_limit))
    up = ~panel.limit_exempt & (opening >= panel.up_limit - 1e-8)
    down = ~panel.limit_exempt & (opening <= panel.down_limit + 1e-8)
    # Retain the registered common entry convention: observed and not limit-up.
    allowed = (
        panel.member
        & panel.observed
        & entry_inside
        & panel.observed[entry_index]
        & known[entry_index]
        & ~up[entry_index]
    )
    price = np.full(shape, np.nan)
    day = np.full(shape, -1, dtype=int)
    unresolved = np.zeros(shape, bool)
    for delay in range(max_exit_delay + 1):
        target = nominal + delay
        idx = np.minimum(target, max(0, limit - 1))[:, 0]
        can = (target < limit) & panel.observed[idx] & known[idx] & ~(up if short else down)[idx]
        take = allowed & (day < 0) & can
        price[take] = np.broadcast_to(opening[idx], shape)[take]
        day[take] = np.broadcast_to(target, shape)[take]
    exhausted = allowed & (day < 0) & (nominal + max_exit_delay < limit)
    # Carry the last observed close, as of the last permitted delay day only.
    last_close = np.full(shape[1], np.nan)
    carried = np.full(shape, np.nan)
    for i in range(limit):
        last_close = np.where(panel.observed[i], panel.fields["close"][i], last_close)
        carried[i] = last_close
    target = np.minimum(nominal + max_exit_delay, max(0, limit - 1))[:, 0]
    fallback = carried[target]
    take = exhausted & np.isfinite(fallback)
    price[take] = fallback[take]
    day[take] = np.broadcast_to(nominal + max_exit_delay, shape)[take]
    unresolved[take] = True
    boundary_unknown = allowed & (day < 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        returns = price / opening[entry_index] - 1
    return {
        "returns": returns,
        "eligible": allowed,
        "exit_day": day,
        "exit_price": price,
        "exit_unresolved": unresolved,
        "boundary_unknown": boundary_unknown,
        "nominal_inside": np.broadcast_to(nominal < limit, shape),
        "entry_limit_up": (panel.member & panel.observed & entry_inside & up[entry_index]),
        "entry_unobserved": (
            panel.member & panel.observed & entry_inside & ~panel.observed[entry_index]
        ),
        "unknown_limits": (panel.member & panel.observed & entry_inside & ~known[entry_index]),
    }


def forward_returns(
    panel: Panel, horizon: int, *, max_exit_delay: int = 10, boundary: int | None = None
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    labels = exit_labels(panel, horizon, max_exit_delay=max_exit_delay, boundary=boundary)
    counters = {
        key: int(labels[key].sum())
        for key in [
            "entry_limit_up",
            "entry_unobserved",
            "unknown_limits",
            "exit_unresolved",
            "boundary_unknown",
        ]
    }
    counters["eligible_decisions"] = int(labels["eligible"].sum())
    return labels["returns"], labels["eligible"], counters


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
    *,
    max_exit_delay: int = 10,
    boundary: int | None = None,
    _labels: tuple[dict[str, np.ndarray], ...] | None = None,
) -> dict[str, Any]:
    """Unit long + unit short, actual per-security exits; no new basket while capital is held.

    Boundary-unknown selected positions invalidate the aggregate, never remove that stock.
    Exhausted exits are marked, remain unresolved, and incur no fictional closing fee.
    """
    from .evaluator import stamp_duty_rate

    limit = min(boundary if boundary is not None else len(panel.dates), len(panel.dates))
    sides = _labels or tuple(
        [
            exit_labels(panel, horizon, max_exit_delay=max_exit_delay, boundary=limit, short=x)
            for x in (False, True)
        ]
    )
    cost = cost_bps / 10000
    nav, net = [1.0], []
    exclusions = delayed = unresolved = unknown = 0
    available = start + 1
    opening, close = panel.fields["open"], panel.fields["close"]
    tax_rates = np.array([stamp_duty_rate(stamp_duty, d.astype(object)) for d in panel.dates])
    entry_days = []
    for t in range(start, stop, horizon):
        entry, nominal = t + 1, t + 1 + horizon
        if nominal >= limit or entry < available:
            exclusions += 1
            continue
        names = np.flatnonzero(eligible[t] & np.isfinite(values[t]))
        if len(names) < minimum:
            exclusions += 1
            continue
        order = names[np.argsort(values[t, names], kind="stable")]
        size = max(1, int(np.floor(len(order) * 0.2)))
        longs, shorts = order[-size:], order[:size]
        selected = [(sides[0], longs, 1), (sides[1], shorts, -1)]
        unknown += sum(
            int(labels["boundary_unknown"][t, names].sum()) for labels, names, _ in selected
        )
        if unknown:
            break
        entry_days.append(entry)
        last = max(int(labels["exit_day"][t, names].max()) for labels, names, _ in selected)
        delayed += sum(
            int((labels["exit_day"][t, names] > nominal).sum()) for labels, names, _ in selected
        )
        unresolved += sum(
            int(labels["exit_unresolved"][t, names].sum()) for labels, names, _ in selected
        )
        base = nav[-1]
        entry_fee = 2 * cost + stamp_duty_rate(stamp_duty, panel.dates[entry].astype(object))
        nav.append(base * (1 - entry_fee))
        carried = {sign: opening[entry, names].copy() for _, names, sign in selected}
        rates_by_side = {
            sign: tax_rates[labels["exit_day"][t, names]] if sign == 1 else np.zeros(len(names))
            for labels, names, sign in selected
        }
        for day in range(entry + 1, last + 1):
            value = 1 - entry_fee
            for labels, names, sign in selected:
                exits = labels["exit_day"][t, names]
                observed = panel.observed[day, names]
                mark = np.where(observed, opening[day, names], carried[sign])
                carried[sign] = np.where(observed, close[day, names], carried[sign])
                mark = np.where(exits <= day, labels["exit_price"][t, names], mark)
                ratio = mark / opening[entry, names]
                value += sign * float(np.mean(ratio - 1))
                closed = (exits <= day) & ~labels["exit_unresolved"][t, names]
                rates = rates_by_side[sign]
                value -= float(np.mean(np.where(closed, ratio * (cost + rates), 0)))
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Missing valuation or nonpositive long-short NAV")
            nav.append(base * value)
        net.append(nav[-1] / base - 1)
        available = last
    curve = np.asarray(nav)
    return {
        "return": float(curve[-1] - 1) if net and not unknown else None,
        "drawdown": float(np.max(1 - curve / np.maximum.accumulate(curve)))
        if net and not unknown
        else None,
        "nav": nav,
        "period_returns": net,
        "entry_days": entry_days,
        "portfolio_periods": len(net),
        "excluded_rebalance_days": exclusions,
        "delayed_positions": delayed,
        "exit_unresolved": unresolved,
        "boundary_unknown_positions": unknown,
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
    max_exit_delay: int = 10,
    _labels: tuple[dict[str, np.ndarray], ...] | None = None,
    settlement: bool = False,
) -> dict[str, Any]:
    from .factor_dsl import evaluate

    if (
        not 0 <= start < stop <= len(panel.dates)
        or not np.isfinite(cost_bps)
        or not 0 <= cost_bps <= 100
    ):
        raise ValueError("Invalid factor assessment interval or cost")
    # Factor inputs end at the scoring fence. Settlement/gap prices are exit-only:
    # even unused factor values must not be computed in these periods.
    scoring = panel.window(0, stop) if settlement else panel
    universe = panel.member & panel.observed
    values = np.full(panel.fields["open"].shape, np.nan)
    values[: len(scoring.dates)] = evaluate(
        factor, scoring.factor_inputs(), universe[: len(scoring.dates)]
    )
    denominator = int(universe[start:stop].sum())
    coverage = (
        float((universe[start:stop] & np.isfinite(values[start:stop])).sum() / denominator)
        if denominator
        else 0.0
    )
    if coverage < 0.7:
        raise ValueError(f"Factor coverage below 70%: {coverage:.4f}")
    boundary = min(label_stop if label_stop is not None else len(panel.dates), len(panel.dates))
    sides = _labels or tuple(
        exit_labels(
            panel,
            horizon,
            max_exit_delay=max_exit_delay,
            boundary=boundary,
            short=x,
            entry_stop=stop if settlement else None,
        )
        for x in (False, True)
    )
    if settlement:
        if boundary < stop + horizon + max_exit_delay:
            raise SettlementError("Full exit-only settlement period required")
        if any(s["boundary_unknown"][start:stop].any() for s in sides):
            raise SettlementError("Unexpected unknown exit inside complete settlement fence")
    labels = sides[0]
    eligible = labels["eligible"]
    ic = rank_ic(values, labels["returns"], eligible, minimum)
    # Unknown labels must not remove individual losses from a day's correlation.
    ic[np.any(eligible & labels["boundary_unknown"], axis=1)] = np.nan
    statistics = ic_statistics(ic, start, stop, horizon)
    if statistics["ic_observations"] == 0:
        raise ValueError("No valid non-overlapping IC observations")
    metrics = portfolio_metrics(
        panel,
        values,
        eligible,
        start,
        stop,
        horizon,
        cost_bps,
        stamp_duty_schedule,
        minimum,
        max_exit_delay=max_exit_delay,
        boundary=boundary,
        _labels=sides,
    )
    exclusions = {
        key: int(labels[key][start:stop].sum())
        for key in [
            "entry_limit_up",
            "entry_unobserved",
            "unknown_limits",
            "exit_unresolved",
            "boundary_unknown",
        ]
    }
    return {
        **metrics,
        **statistics,
        "ic_valid_days": int(np.isfinite(ic[start:stop]).sum()),
        "counter_interval": {
            "start_inclusive": str(panel.dates[start]),
            "stop_exclusive_index": stop,
            "start_index": start,
            "scope": "scoring_period_only",
        },
        "coverage": coverage,
        "daily_ic": [float(x) if np.isfinite(x) else None for x in ic[start:stop]],
        "exclusions": exclusions,
        "label_stop_exclusive": boundary,
        "metric_kind": "settlement_long_short_gross_2_open_marked_research"
        if settlement
        else "delayed_exit_long_short_gross_2_open_marked_research",
        "settlement_sessions": horizon + max_exit_delay if settlement else 0,
        "entry_stop_exclusive": stop if settlement else None,
    }
