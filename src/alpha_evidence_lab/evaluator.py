"""统一收益/成本时点：前一收盘信号，下一开盘成交，当日收盘估值；可选卖出分段印花税。"""

import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from typing import Any

import numpy as np

from .data import TZ, Bar
from .models import Candidate


@dataclass(frozen=True)
class Evaluation:
    total_return: float
    max_drawdown: float
    sharpe: float | None
    costs: float
    trades: int
    nav: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stamp_duty_rate(schedule: tuple[tuple[date, float], ...], day: date) -> float:
    """卖出印花税率（分段表按生效日升序）；空表为0；早于首个生效日拒绝，不猜测。"""
    if not schedule:
        return 0.0
    applicable = [bps for effective, bps in schedule if effective <= day]
    if not applicable:
        raise ValueError(f"{day} 早于印花税分段表的首个生效日，税率未知")
    return applicable[-1] / 10000


def evaluate(
    bars: tuple[Bar, ...],
    candidate: Candidate,
    start: int,
    stop: int,
    cost_bps: float,
    stamp_duty: tuple[tuple[date, float], ...] = (),
) -> Evaluation:
    if not candidate.slow <= start < stop <= len(bars):
        raise ValueError("区间无效或缺少完整暖机历史")
    if not math.isfinite(cost_bps) or not 0 <= cost_bps <= 100:
        raise ValueError("成本参数非法")
    # 每个测试段从现金1开始；暖机不带入此前持仓或收益。
    cash, quantity, costs, trades = 1.0, 0.0, 0.0, 0
    nav = [1.0]
    closes = [bar.closing for bar in bars]
    for i in range(start, stop):
        if any(
            bar.available_at > datetime.combine(bars[i].day, time(9, 30), TZ)
            for bar in bars[i - candidate.slow : i]
        ):
            raise ValueError("前一日完整行情在执行时点尚未可得")
        history = closes[i - candidate.slow : i]
        signal = sum(history[-candidate.fast :]) / candidate.fast > sum(history) / candidate.slow
        opening = bars[i].opening
        bar = bars[i]
        known_limits = bar.price_limit_exempt or (
            bar.up_limit is not None and bar.down_limit is not None
        )
        can_buy = (
            known_limits
            and not bar.is_suspended
            and (bar.price_limit_exempt or opening < float(bar.up_limit or 0))
        )
        can_sell = (
            known_limits
            and not bar.is_suspended
            and (bar.price_limit_exempt or opening > float(bar.down_limit or 0))
        )
        if signal and quantity == 0 and can_buy:
            quantity = cash / (opening * (1 + cost_bps / 10000))
            fee = quantity * opening * cost_bps / 10000
            costs += fee
            cash = 0.0
            trades += 1
        elif not signal and quantity > 0 and can_sell:
            rate = cost_bps / 10000 + stamp_duty_rate(stamp_duty, bar.day)
            fee = quantity * opening * rate
            cash += quantity * opening - fee
            quantity = 0.0
            costs += fee
            trades += 1
        nav.append(cash + quantity * bars[i].closing)
    values = np.asarray(nav)
    returns = values[1:] / values[:-1] - 1
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(np.mean(returns) / std * np.sqrt(252)) if std > 1e-12 else None
    return Evaluation(
        float(values[-1] - 1),
        float(np.max(1 - values / np.maximum.accumulate(values))),
        sharpe,
        costs,
        trades,
        tuple(nav),
    )


def selection_score(result: Evaluation) -> float:
    # 预注册的探索性准则；不是统计显著性。
    return result.total_return - 2 * result.max_drawdown
