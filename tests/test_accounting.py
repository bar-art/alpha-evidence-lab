import math
from datetime import date, datetime, time, timedelta

import pytest

from alpha_evidence_lab.data import TZ, Bar, PricePath
from alpha_evidence_lab.evaluator import evaluate
from alpha_evidence_lab.models import Candidate


def bars(closes, opens=None):
    start = date(2024, 1, 1)
    return tuple(
        Bar(
            start + timedelta(days=i),
            (opens or closes)[i],
            close,
            datetime.combine(start + timedelta(days=i), time(15), TZ),
        )
        for i, close in enumerate(closes)
    )


def test_next_open_and_fee_hand_calculation():
    # 前三日已知收盘10,10,12；下一开盘12入场，1%手续费，当日收盘13。
    data = bars([10.0, 10.0, 12.0, 13.0], [10.0, 10.0, 12.0, 12.0])
    result = evaluate(data, Candidate(id="c", fast=2, slow=3), 3, 4, 100)
    assert result.nav[0] == 1
    assert result.nav[-1] == pytest.approx(13 / (12 * 1.01))
    assert result.costs == pytest.approx(1 - 1 / 1.01)
    assert result.trades == 1
    assert result.total_return == pytest.approx(result.nav[-1] - 1)


def test_current_close_cannot_change_current_open_trade():
    first = evaluate(bars([10.0, 10.0, 10.0, 100.0]), Candidate(id="c", fast=2, slow=3), 3, 4, 0)
    assert first.trades == 0
    assert first.total_return == 0


def test_initial_loss_is_included_in_drawdown_and_return():
    result = evaluate(
        bars([10.0, 10.0, 12.0, 6.0], [10.0, 10.0, 12.0, 12.0]),
        Candidate(id="c", fast=2, slow=3),
        3,
        4,
        0,
    )
    assert result.total_return == -0.5
    assert result.max_drawdown == 0.5


def test_sell_cost_and_overnight_gap():
    data = bars([10.0, 10.0, 12.0, 6.0, 5.0], [10.0, 10.0, 12.0, 12.0, 8.0])
    result = evaluate(data, Candidate(id="c", fast=2, slow=3), 3, 5, 100)
    assert result.trades == 2
    assert result.nav[-1] == pytest.approx(8 * 0.99 / (12 * 1.01))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -1.0])
def test_bad_prices(value):
    data = list(bars([10.0, 11.0, 12.0]))
    data[0] = Bar(data[0].day, value, 10.0, data[0].available_at)
    with pytest.raises(ValueError):
        PricePath("x", tuple(data))


def test_date_clock_and_warmup_validation():
    data = bars([10.0, 11.0, 12.0, 13.0])
    with pytest.raises(ValueError):
        PricePath("x", data + data[:1])
    with pytest.raises(ValueError):
        PricePath("x", (Bar(data[0].day, 10.0, 10.0, data[0].available_at.replace(tzinfo=None)),))
    with pytest.raises(ValueError):
        PricePath("x", (Bar(data[0].day, 10.0, 10.0, data[0].available_at.replace(hour=9)),))
    with pytest.raises(ValueError):
        evaluate(data, Candidate(id="c", fast=2, slow=3), 2, 4, 7)
    with pytest.raises(ValueError):
        evaluate(data, Candidate(id="c", fast=2, slow=3), 3, 4, math.inf)


def test_delayed_data_is_not_made_available_early(paths):
    path = paths[0]
    index = 8
    data = list(path.bars)
    row = data[index]
    data[index] = Bar(row.day, row.opening, row.closing, row.available_at + timedelta(days=5))
    changed = PricePath("x", tuple(data))
    with pytest.raises(ValueError):
        changed.known(index)
    with pytest.raises(ValueError):
        changed.known(-1)


def test_older_late_publication_cannot_leak_through_rolling_window():
    data = list(bars([10.0, 11.0, 12.0, 13.0]))
    old = data[0]
    data[0] = Bar(old.day, old.opening, old.closing, old.available_at + timedelta(days=10))
    with pytest.raises(ValueError):
        evaluate(tuple(data), Candidate(id="c", fast=2, slow=3), 3, 4, 0)


def test_dated_stamp_duty_on_sells_only():
    from pathlib import Path

    from alpha_evidence_lab.evaluator import stamp_duty_rate
    from alpha_evidence_lab.models import StudyPlan

    schedule = ((date(2008, 9, 19), 10.0), (date(2024, 1, 5), 5.0))
    data = bars([10.0, 10.0, 12.0, 6.0, 5.0], [10.0, 10.0, 12.0, 12.0, 8.0])
    # 01-04 买入只付 1% 佣金；01-05 卖出付 1% 佣金 + 当日生效的 5bps 印花税。
    result = evaluate(data, Candidate(id="c", fast=2, slow=3), 3, 5, 100, schedule)
    assert result.nav[-1] == pytest.approx(8 * (1 - 0.01 - 0.0005) / (12 * 1.01))
    assert stamp_duty_rate(schedule, date(2024, 1, 4)) == 0.001
    assert stamp_duty_rate((), date(2000, 1, 4)) == 0
    with pytest.raises(ValueError, match="税率未知"):
        stamp_duty_rate(schedule, date(2008, 9, 18))
    plan = StudyPlan.model_validate_json(
        (Path(__file__).parents[1] / "config" / "ashare.json").read_text()
    )
    assert plan.stamp_duty_schedule[-1] == (date(2023, 8, 28), 5.0)
    with pytest.raises(ValueError):
        StudyPlan(stamp_duty_schedule=((date(2023, 8, 28), 5.0), (date(2008, 9, 19), 10.0)))
