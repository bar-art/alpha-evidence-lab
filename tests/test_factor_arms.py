import json

import numpy as np
import pytest
from test_factor_eval import panel

from alpha_evidence_lab.factor_arms import CLASSIC_EXPRESSIONS, normalize, random_factor, search_arm
from alpha_evidence_lab.ledger import Ledger
from alpha_evidence_lab.llm import LlmBudgetError, OpenRouterClient
from alpha_evidence_lab.models import LlmSettings


def test_classic_order_and_random_reproducibility():
    assert len(CLASSIC_EXPRESSIONS) == 10
    assert CLASSIC_EXPRESSIONS[0] == "cs_rank(-pct_change(close, 5))"
    assert normalize("cs_rank(close)+volume").expression.startswith("cs_rank(cs_rank")
    assert all(normalize(x).lookback <= 250 for x in CLASSIC_EXPRESSIONS)
    a, b = np.random.default_rng(9), np.random.default_rng(9)
    for _ in range(500):
        first, second = random_factor(a), random_factor(b)
        assert first == second and first.lookback <= 250 and first.depth <= 4


def test_llm_failures_consume_trials_but_json_repairs_do_not(tmp_path, monkeypatch):
    from datetime import date

    calls = []
    expressions = iter(["close", "unknown(close)", "close", "volume", "ret"])

    def transport(body, key):
        calls.append(body)
        content = (
            "{bad}"
            if len(calls) == 1
            else json.dumps(
                {
                    "hypothesis": "Price-volume hypothesis",
                    "expression": next(expressions),
                    "failure_conditions": "Regime changes",
                    "evidence_ids": ["e1"],
                }
            )
        )
        return {"choices": [{"message": {"content": content}}], "usage": {"cost": 0.0}}

    def assess(p, factor, *args, **kwargs):
        if factor.expression == "cs_rank(volume)":
            raise ValueError("Factor coverage below 70%")
        return {"icir": 1.0, "ic_mean": 0.02, "coverage": 1.0}

    monkeypatch.setattr("alpha_evidence_lab.factor_arms.evaluate_factor", assess)
    client = OpenRouterClient(
        LlmSettings(
            model="fixture/model",
            release_date=date(2020, 1, 1),
            release_source="https://example.invalid",
        ),
        tmp_path / "cache",
        1,
        api_key="fixture",
        transport=transport,
        pricing={"prompt": 0.0, "completion": 0.0},
    )
    ledger = Ledger(tmp_path / "audit", "p", "d", "c", budget=5, arms=("llm",))
    ledger.start_batch("fixture")
    winner = search_arm(
        "llm",
        panel(),
        3,
        10,
        ledger=ledger,
        batch="fixture",
        budget=5,
        horizon=2,
        cost_bps=7,
        stamp_duty_schedule=(),
        seed=1,
        memories=[{"expression": "cs_rank(close)", "status": "ok", "score": 0.2}],
        client=client,
        minimum=3,
    )
    rows = ledger.trials("fixture", "llm")
    assert len(rows) == 5 and sum(r["status"] == "failed" for r in rows) == 3
    assert client.calls == 6 and client.repairs == 1 and winner["score"] == 1
    for body in calls:
        text = json.dumps(body)
        assert "600000.SH" not in text and "2024-01" not in text and "holdout" not in text


def test_paid_budget_exhaustion_remains_incomplete(tmp_path):
    from datetime import date

    client = OpenRouterClient(
        LlmSettings(
            model="fixture/model",
            release_date=date(2020, 1, 1),
            release_source="https://example.invalid",
            max_cost_usd=0.01,
        ),
        tmp_path / "cache",
        1,
        api_key="fixture",
        pricing={"prompt": 1.0, "completion": 1.0},
        transport=lambda *args: {},
    )
    ledger = Ledger(tmp_path / "audit", "p", "d", "c", budget=1, arms=("llm",))
    ledger.start_batch("fixture")
    with pytest.raises(LlmBudgetError):
        search_arm(
            "llm",
            panel(),
            3,
            10,
            ledger=ledger,
            batch="fixture",
            budget=1,
            horizon=2,
            cost_bps=7,
            stamp_duty_schedule=(),
            seed=1,
            memories=[],
            client=client,
            minimum=3,
        )
    assert not ledger.trials("fixture", "llm") and client.calls == 0
