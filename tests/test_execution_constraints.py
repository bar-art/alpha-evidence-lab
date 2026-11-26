from dataclasses import replace
from datetime import date, datetime, time, timedelta

import pytest

from alpha_evidence_lab.data import TZ, Bar
from alpha_evidence_lab.evaluator import evaluate
from alpha_evidence_lab.models import Candidate


def sample():
    start = date(2024, 1, 1)
    return tuple(
        Bar(
            start + timedelta(days=i),
            10.0,
            close,
            datetime.combine(start + timedelta(days=i), time(15), TZ),
            up_limit=11.0,
            down_limit=9.0,
            price_limit_exempt=False,
        )
        for i, close in enumerate([1.0, 2.0, 3.0, 1.0, 5.0])
    )


@pytest.mark.parametrize(
    "changes", [dict(is_suspended=True), dict(up_limit=10.0), dict(up_limit=None)]
)
def test_cannot_buy_suspended_limit_up_or_unknown_limit(changes):
    bars = list(sample())
    bars[3] = replace(bars[3], **changes)
    assert evaluate(tuple(bars), Candidate(id="test", fast=2, slow=3), 3, 4, 0).trades == 0


def test_limit_down_blocks_exit_and_retains_marked_position():
    bars = list(sample())
    candidate = Candidate(id="test", fast=2, slow=3)
    assert evaluate(tuple(bars), candidate, 3, 5, 0).trades == 2
    bars[4] = replace(bars[4], down_limit=10.0)
    blocked = evaluate(tuple(bars), candidate, 3, 5, 0)
    assert blocked.trades == 1
    assert blocked.nav[-1] == pytest.approx(0.5)
