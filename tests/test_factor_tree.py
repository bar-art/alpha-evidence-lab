import json
from datetime import date

import numpy as np
import pytest
from test_factor_eval import panel

from alpha_evidence_lab.factor_arms import (
    CLASSIC_EXPRESSIONS,
    normalize,
    proposal_schema,
    random_factor,
    search_arm,
)
from alpha_evidence_lab.factor_tree import expression_tree, schema_issue, serialize_tree
from alpha_evidence_lab.ledger import Ledger
from alpha_evidence_lab.llm import OpenRouterClient, schema_problem
from alpha_evidence_lab.models import LlmSettings


@pytest.mark.parametrize("expression", CLASSIC_EXPRESSIONS)
def test_ten_handwritten_expression_tree_roundtrips(expression):
    factor = normalize(expression)
    tree = expression_tree(expression)
    assert normalize(serialize_tree(tree)) == factor
    assert not schema_issue(tree, {"$ref": "#/$defs/node"}, proposal_schema([], "v2-tree"))


def test_random_roundtrips_and_historical_invalid_strings():
    rng = np.random.default_rng(1)
    for _ in range(100):
        f = random_factor(rng)
        assert normalize(serialize_tree(expression_tree(f.expression))) == f
    # Exact five expressions retained from the v1 trial audit, without its credentials/cache.
    historical = [
        "(ts_delta(close, 5) > 0) and (ts_delta(volume, 5) > 0)",
        "ts_delta(close, 5) > 0 and ts_delta(volume, 5) > 0",
        "ts_delta(close,5) > 0 and ts_delta(volume,5) > 0",
        "(ts_delta(close,5) > 0) and (ts_delta(volume,5) > 0)",
        "ts_delta(close,5) > 0 and ts_delta(volume,5) > 0",
    ]
    for expression in historical:
        assert schema_problem(
            {
                "hypothesis": "h",
                "expression": expression,
                "failure_conditions": "f",
                "evidence_ids": ["none"],
            },
            proposal_schema([], "v2-tree"),
        )
    assert schema_problem(
        {
            "hypothesis": "h",
            "tree": {"fn": "ts_delta", "args": []},
            "failure_conditions": "f",
            "evidence_ids": ["none"],
        },
        proposal_schema([], "v2-tree"),
    )


def test_tree_format_repairs_vs_semantic_trials_and_feedback(tmp_path, monkeypatch):
    leaf = {"field": "close"}
    deep = leaf
    for _ in range(9):
        deep = {"fn": "abs", "args": [deep]}
    lookback = leaf
    for _ in range(3):
        lookback = {"fn": "delay", "args": [lookback], "window": 120}
    trees = iter(
        [
            {"op": "and", "args": []},  # format repair, not a trial
            leaf,
            {"fn": "ts_mean", "args": [], "window": 5},  # arity
            {"fn": "ts_std", "args": [leaf], "window": 1},  # minimum
            deep,
            lookback,
            leaf,  # duplicate
            {"field": "volume"},
        ]
    )
    calls = []

    def transport(body, key):
        calls.append(body)
        idea = {
            "hypothesis": "h",
            "tree": next(trees),
            "failure_conditions": "f",
            "evidence_ids": ["none"],
        }
        return {"choices": [{"message": {"content": json.dumps(idea)}}], "usage": {"cost": 0}}

    monkeypatch.setattr(
        "alpha_evidence_lab.factor_arms.evaluate_factor",
        lambda *a, **kw: {"icir": 1.0, "return": 0.1},
    )
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
    ledger = Ledger(tmp_path / "ledger", "p", "d", "c", budget=7, arms=("llm",))
    ledger.start_batch("b")
    search_arm(
        "llm",
        panel(),
        3,
        10,
        ledger=ledger,
        batch="b",
        budget=7,
        horizon=2,
        cost_bps=0,
        stamp_duty_schedule=(),
        seed=1,
        memories=[],
        client=client,
        minimum=3,
        prompt_version="v2-tree",
    )
    rows = ledger.trials("b", "llm")
    assert len(rows) == 7 and sum(r["status"] == "failed" for r in rows) == 5
    assert client.calls == 8 and client.repairs == 1
    final = json.loads(calls[-1]["messages"][1]["content"])
    assert any(r["failure"] for r in final["attempts_in_this_round"])
    for request in calls:
        text = json.dumps(request)
        assert "600000.SH" not in text and "2024-01" not in text and "holdout" not in text


def test_shared_limits_ignore_only_outer_rank_depth():
    from alpha_evidence_lab.factor_arms import bounded_factor
    from alpha_evidence_lab.factor_dsl import FactorError

    leaf = "close"
    for _ in range(3):
        leaf = f"abs({leaf})"
    assert bounded_factor(leaf).depth == 5
    assert bounded_factor(f"cs_rank({leaf})") == bounded_factor(leaf)
    with pytest.raises(FactorError, match="Shared expression limits"):
        bounded_factor(f"abs({leaf})")
    with pytest.raises(FactorError, match="Shared expression limits"):
        bounded_factor("close+volume", max_nodes=3)
    with pytest.raises(FactorError, match="Shared expression limits"):
        bounded_factor("delay(close,120)", max_lookback=100)
    # Random and model tree adapter share the exact same acceptance function.
    rng = np.random.default_rng(42)
    for _ in range(500):
        f = random_factor(rng)
        assert f == bounded_factor(serialize_tree(expression_tree(f.expression)))


def test_ten_independently_written_trees_match_classics():
    def f(name, field, window):
        return {"fn": name, "args": [{"field": field}], "window": window}

    def neg(tree):
        return {"op": "neg", "args": [tree]}

    trees = [
        neg(f("pct_change", "close", 5)),
        neg(f("ts_std", "ret", 20)),
        {
            "op": "sub",
            "args": [
                {"op": "div", "args": [f("delay", "close", 20), f("delay", "close", 120)]},
                {"const": 1},
            ],
        },
        {"op": "div", "args": [neg(f("ts_mean", "volume", 20)), f("ts_mean", "volume", 120)]},
        neg({"fn": "ts_corr", "args": [{"field": "ret"}, {"field": "volume"}], "window": 20}),
        neg(
            {
                "fn": "ts_mean",
                "args": [
                    {
                        "op": "div",
                        "args": [
                            {"op": "sub", "args": [{"field": "high"}, {"field": "low"}]},
                            {"field": "close"},
                        ],
                    }
                ],
                "window": 20,
            }
        ),
        neg(f("ts_max", "ret", 20)),
        {"op": "div", "args": [{"field": "close"}, f("ts_max", "high", 120)]},
        f("pct_change", "close", 20),
        neg(f("ts_std", "ret", 60)),
    ]
    for tree, expected in zip(trees, CLASSIC_EXPRESSIONS, strict=True):
        assert normalize(serialize_tree(tree)) == normalize(expected)
