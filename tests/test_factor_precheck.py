import json
from dataclasses import replace
from datetime import date

from test_factor_study import random_panel

from alpha_evidence_lab.factor_arms import normalize, random_precheck, search_arm
from alpha_evidence_lab.ledger import Ledger
from alpha_evidence_lab.llm import OpenRouterClient
from alpha_evidence_lab.models import LlmSettings


def test_precheck_is_feature_only_and_rejects_constant_and_low_coverage(monkeypatch):
    p = random_panel(days=80)
    monkeypatch.setattr(
        "alpha_evidence_lab.factor_arms.exit_labels",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("No labels allowed")),
    )
    assert "constant" in random_precheck(p, normalize("1"), 20, 40)
    assert "coverage" in random_precheck(p, normalize("ts_mean(close,120)"), 20, 40)
    assert random_precheck(p, normalize("close"), 20, 40) == ""
    fields = {k: v.copy() for k, v in p.fields.items()}
    for v in fields.values():
        v[40:] *= 100
    assert random_precheck(replace(p, fields=fields), normalize("close"), 20, 40) == ""


def test_redraw_accounting_and_candidates_ignore_future_returns(tmp_path, monkeypatch):
    p = random_panel(days=80)
    fields = {k: v.copy() for k, v in p.fields.items()}
    for v in fields.values():
        v[40:] *= 1.1
    outputs = []
    evaluated = []

    def assess(panel, factor, *args, **kwargs):
        evaluated.append(factor.expression)
        return {"icir": 1.0, "return": 0.1}

    monkeypatch.setattr("alpha_evidence_lab.factor_arms.evaluate_factor", assess)
    for index, data in enumerate([p, replace(p, fields=fields)]):
        expressions = iter(["1", "ts_mean(close,120)", "close", "close", "volume"])
        monkeypatch.setattr(
            "alpha_evidence_lab.factor_arms.random_factor",
            lambda *a, expressions=expressions: normalize(next(expressions)),
        )
        ledger = Ledger(tmp_path / f"ledger{index}", "p", "d", "c", budget=2, arms=("random",))
        ledger.start_batch("b")
        search_arm(
            "random",
            data,
            20,
            40,
            ledger=ledger,
            batch="b",
            budget=2,
            horizon=5,
            cost_bps=0,
            stamp_duty_schedule=(),
            seed=42,
            memories=[],
            precheck_random=True,
        )
        rows = ledger.trials("b", "random")
        assert len(rows) == 2 and [r["payload"]["precheck"]["redraws"] for r in rows] == [2, 1]
        assert all(not r["payload"]["precheck"]["uses_labels"] for r in rows)
        outputs.append([(r["candidate"], r["payload"]["precheck"]) for r in rows])
    assert outputs[0] == outputs[1]
    assert evaluated == ["cs_rank(close)", "cs_rank(volume)"] * 2


def test_actual_seed_is_reproducible_and_exhaustion_is_bounded(tmp_path, monkeypatch):
    p = random_panel(days=80)
    monkeypatch.setattr(
        "alpha_evidence_lab.factor_arms.evaluate_factor",
        lambda *a, **k: {"icir": 1.0, "return": 0.1},
    )
    outputs = []
    for i in range(2):
        ledger = Ledger(tmp_path / f"seed{i}", "p", "d", "c", budget=5, arms=("random",))
        ledger.start_batch("b")
        search_arm(
            "random",
            p,
            20,
            40,
            ledger=ledger,
            batch="b",
            budget=5,
            horizon=5,
            cost_bps=0,
            stamp_duty_schedule=(),
            seed=42,
            memories=[],
            precheck_random=True,
        )
        outputs.append(
            [(r["candidate"], r["payload"]["precheck"]) for r in ledger.trials("b", "random")]
        )
    assert outputs[0] == outputs[1]
    draws = []

    def constant(*args):
        draws.append(1)
        return normalize("1")

    monkeypatch.setattr("alpha_evidence_lab.factor_arms.random_factor", constant)
    ledger = Ledger(tmp_path / "exhausted", "p", "d", "c", budget=1, arms=("random",))
    ledger.start_batch("b")
    try:
        search_arm(
            "random",
            p,
            20,
            40,
            ledger=ledger,
            batch="b",
            budget=1,
            horizon=5,
            cost_bps=0,
            stamp_duty_schedule=(),
            seed=42,
            memories=[],
            precheck_random=True,
        )
    except ValueError:
        pass
    rows = ledger.trials("b", "random")
    assert len(draws) == 51 and len(rows) == 1 and rows[0]["status"] == "failed"
    assert "exhausted 50" in rows[0]["payload"]["error"]


def test_llm_degenerate_proposal_is_not_prechecked_and_consumes_budget(tmp_path, monkeypatch):
    p = random_panel(days=80)
    ideas = iter([{"const": 1}, {"field": "close"}])

    def transport(body, key):
        idea = {
            "hypothesis": "h",
            "tree": next(ideas),
            "failure_conditions": "f",
            "evidence_ids": ["none"],
        }
        return {"choices": [{"message": {"content": json.dumps(idea)}}], "usage": {"cost": 0}}

    monkeypatch.setattr(
        "alpha_evidence_lab.factor_arms.random_precheck",
        lambda *a: (_ for _ in ()).throw(AssertionError("No LLM precheck")),
    )
    model = LlmSettings(
        model="fixture/model",
        release_date=date(2020, 1, 1),
        release_source="https://example.invalid",
    )
    client = OpenRouterClient(
        model,
        tmp_path / "cache",
        1,
        api_key="fixture",
        transport=transport,
        pricing={"prompt": 0.0, "completion": 0.0},
    )
    ledger = Ledger(tmp_path / "ledger", "p", "d", "c", budget=2, arms=("llm",))
    ledger.start_batch("b")
    search_arm(
        "llm",
        p,
        20,
        40,
        ledger=ledger,
        batch="b",
        budget=2,
        horizon=5,
        cost_bps=0,
        stamp_duty_schedule=(),
        seed=42,
        memories=[],
        client=client,
        prompt_version="v2-tree",
        precheck_random=True,
    )
    rows = ledger.trials("b", "llm")
    assert len(rows) == 2 and rows[0]["status"] == "failed" and rows[1]["status"] == "ok"
    assert "No valid" in rows[0]["payload"]["error"]
