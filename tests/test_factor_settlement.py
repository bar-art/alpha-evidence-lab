import copy
import json
from dataclasses import replace
from datetime import date

import numpy as np
import pytest
from test_factor_eval import panel
from test_factor_study import random_panel

from alpha_evidence_lab.factor_arms import CLASSIC_EXPRESSIONS
from alpha_evidence_lab.factor_dsl import parse
from alpha_evidence_lab.factor_eval import (
    SettlementError,
    evaluate_factor,
    exit_labels,
    portfolio_metrics,
)
from alpha_evidence_lab.factor_study import episode_calendar, episodes, run_factor_study
from alpha_evidence_lab.factor_tree import expression_tree
from alpha_evidence_lab.ledger import Ledger
from alpha_evidence_lab.llm import OpenRouterClient
from alpha_evidence_lab.models import FactorPlan, LlmSettings


def test_late_long_and_short_settle_in_eighth_day_and_exhausted_mark():
    p = panel(days=40, stocks=5)
    for value in p.fields.values():
        value[:] = 10
    p.fields["open"][21:, 4] = 7
    p.down_limit[21:27, 4] = 7
    p.fields["open"][21:, 0] = 12
    p.up_limit[21:26, 0] = 12
    labels = exit_labels(p, 5, boundary=35, entry_stop=20)
    assert labels["exit_day"][15, 4] == 27  # settlement day 8
    assert not labels["eligible"][19:].any()  # no settlement entries, including final close
    scores = np.broadcast_to(np.arange(5.0), (40, 5))
    m = portfolio_metrics(p, scores, labels["eligible"], 0, 20, 5, 0, minimum=3, boundary=35)
    assert m["entry_days"] == [1, 6, 11, 16]
    assert m["return"] == pytest.approx(-0.5)
    assert m["drawdown"] == pytest.approx(0.5)
    assert m["delayed_positions"] == 2 and m["boundary_unknown_positions"] == 0
    for v in p.fields.values():
        v[35:] *= 99
    assert m == portfolio_metrics(
        p, scores, labels["eligible"], 0, 20, 5, 0, minimum=3, boundary=35
    )
    p.down_limit[21:32, 4] = 7
    p.fields["close"][31, 4] = 6
    m = portfolio_metrics(p, scores, labels["eligible"], 0, 20, 5, 0, minimum=3, boundary=35)
    assert m["exit_unresolved"] == 1 and m["return"] == pytest.approx(-0.6)


def test_v3_factor_inputs_end_at_score_stop_and_bad_fence_raises(monkeypatch):
    import alpha_evidence_lab.factor_dsl as dsl

    p = random_panel(days=80)
    actual = dsl.evaluate
    calls = []

    def checked(factor, inputs, universe):
        calls.append(len(inputs["close"]))
        assert len(inputs["close"]) == 50
        return actual(factor, inputs, universe)

    monkeypatch.setattr(dsl, "evaluate", checked)
    before = evaluate_factor(p, parse("cs_rank(close)"), 30, 50, settlement=True)
    fields = {k: v.copy() for k, v in p.fields.items()}
    for v in fields.values():
        v[65:] *= 2
    after = evaluate_factor(
        replace(p, fields=fields), parse("cs_rank(close)"), 30, 50, settlement=True, label_stop=65
    )
    bounded = evaluate_factor(p, parse("cs_rank(close)"), 30, 50, settlement=True, label_stop=65)
    assert after == bounded
    assert before["entry_days"] == after["entry_days"]
    fields = {k: v.copy() for k, v in p.fields.items()}
    for v in fields.values():
        v[50:65] *= 1.1
    changed = evaluate_factor(
        replace(p, fields=fields), parse("cs_rank(close)"), 30, 50, settlement=True, label_stop=65
    )
    assert changed["entry_days"] == bounded["entry_days"]
    assert changed["return"] != bounded["return"]
    with pytest.raises(SettlementError, match="Full exit-only"):
        evaluate_factor(p, parse("cs_rank(close)"), 30, 50, settlement=True, label_stop=64)


def test_v3_grid_keeps_ids_excludes_date_ineligible_blocks():
    p = random_panel(days=460)
    plan = FactorPlan(protocol_version="v3-settlement")
    schedule, excluded = episode_calendar(p, plan)
    assert plan.settlement_sessions == plan.gap_sessions == 15
    assert [r["episode"] for r in schedule] == [1, 2]
    assert schedule[0]["test_start"] == 395 and schedule[0]["dev_stop"] == 380
    assert schedule[0]["settlement_stop"] == 430
    assert [(r["episode"], r["reason"]) for r in excluded] == [
        (0, "insufficient_warmup"),
        (3, "incomplete_settlement"),
    ]
    assert episodes(p, FactorPlan())[1]["test_start"] == schedule[0]["test_start"]


def test_next_search_and_llm_requests_do_not_depend_on_prior_reveal(tmp_path, monkeypatch):
    import alpha_evidence_lab.factor_study as module

    p = random_panel(days=480)
    model = LlmSettings(
        model="fixture/model",
        release_date=date(2020, 1, 1),
        release_source="https://example.invalid",
    )
    plan = FactorPlan(protocol_version="v3-settlement", llm=model, budget=2, max_episodes=2)
    all_requests, all_searches = [], []
    actual_search, actual_reveal = module.search_arm, Ledger.reveal
    deferred = []
    for defer in (False, True):
        requests, searches = [], []

        def transport(body, key, requests=requests):
            req = json.loads(body["messages"][1]["content"])
            requests.append(copy.deepcopy(body))
            aliases = req["prior_development_evidence"]
            idea = {
                "hypothesis": "h",
                "tree": expression_tree(CLASSIC_EXPRESSIONS[2 - req["attempts_remaining"]]),
                "failure_conditions": "f",
                "evidence_ids": [aliases[0]["evidence_id"]] if aliases else ["none"],
            }
            return {"choices": [{"message": {"content": json.dumps(idea)}}], "usage": {"cost": 0}}

        def search(*args, searches=searches, **kwargs):
            searches.append(
                (
                    args[0],
                    args[1].sha256,
                    args[2:4],
                    kwargs["seed"],
                    copy.deepcopy(kwargs["memories"]),
                )
            )
            return actual_search(*args, **kwargs)

        def reveal(self, batch, results):
            deferred.append((self, batch, results))

        monkeypatch.setattr(module, "search_arm", search)
        monkeypatch.setattr(Ledger, "reveal", reveal if defer else actual_reveal)
        client = OpenRouterClient(
            model,
            tmp_path / f"cache{defer}",
            1,
            api_key="fixture",
            transport=transport,
            pricing={"prompt": 0.0, "completion": 0.0},
        )
        result = run_factor_study(plan, p, tmp_path / f"run{defer}", client=client)
        assert len(result["batches"]) == 2
        all_requests.append(requests)
        all_searches.append(searches)
    assert len(deferred) == 2
    assert all_searches[0] == all_searches[1]
    assert all_requests[0] == all_requests[1]
    for self, batch, results in deferred:
        actual_reveal(self, batch, results)


def test_settlement_mutation_does_not_change_development_or_entry_choices(tmp_path, monkeypatch):
    import alpha_evidence_lab.factor_eval as module

    p = random_panel(days=480)
    plan = FactorPlan(protocol_version="v3-settlement", budget=2, max_episodes=1)
    ep = episodes(p, plan)[0]
    modified = {k: v.copy() for k, v in p.fields.items()}
    for v in modified.values():
        v[ep["test_stop"] : ep["settlement_stop"]] *= 1.2
    captures = []
    actual = module.portfolio_metrics

    def capture(panel, values, eligible, start, stop, *args, **kwargs):
        captures.append((values[:stop].copy(), eligible[:stop].copy()))
        return actual(panel, values, eligible, start, stop, *args, **kwargs)

    monkeypatch.setattr(module, "portfolio_metrics", capture)
    a = run_factor_study(plan, p, tmp_path / "a")
    left = captures.copy()
    captures.clear()
    b = run_factor_study(plan, replace(p, fields=modified), tmp_path / "b")
    assert [(r["candidate"], r["score"]) for r in a["trials"]] == [
        (r["candidate"], r["score"]) for r in b["trials"]
    ]
    for first, second in zip(left, captures, strict=True):
        for x, y in zip(first, second, strict=True):
            np.testing.assert_equal(x, y)


def test_v3_unexpected_unknown_is_an_error_not_a_missing_return(monkeypatch):
    import alpha_evidence_lab.factor_eval as module

    p = random_panel(days=80)
    actual = module.exit_labels

    def broken(*args, **kwargs):
        labels = actual(*args, **kwargs)
        labels["boundary_unknown"][30, 0] = True
        return labels

    monkeypatch.setattr(module, "exit_labels", broken)
    with pytest.raises(SettlementError, match="Unexpected unknown"):
        evaluate_factor(p, parse("close"), 30, 50, settlement=True, label_stop=65)
