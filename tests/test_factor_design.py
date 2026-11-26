from datetime import date

import pytest
from test_factor_study import random_panel

from alpha_evidence_lab.factor_design import power_description
from alpha_evidence_lab.factor_study import episode_calendar, factor_gate
from alpha_evidence_lab.models import FactorPlan, LlmSettings


def test_trial_ids_keep_original_dates_and_power_counts_eligible_blocks():
    p = random_panel(days=2889, stocks=3)
    plan = FactorPlan(protocol_version="v3-settlement", confirm_start=p.dates[2594].astype(object))
    schedule, excluded = episode_calendar(p, plan)
    assert len(schedule) == 121
    assert [e["episode"] for e in schedule if e["phase"] == "confirmation"] == list(range(113, 124))
    banned = {e["episode"]: e for e in excluded}
    assert set(banned) == {0, 111, 112, 124}
    assert banned[111]["test_start"] == 2595 and banned[112]["test_start"] == 2615
    assert banned[124]["reason"] == "incomplete_settlement"
    with pytest.raises(ValueError, match="must exclude"):
        FactorPlan(protocol_version="v3-settlement", excluded_episodes=(111,))
    with pytest.raises(ValueError, match="unique"):
        FactorPlan(excluded_episodes=(111, 111))


def test_power_resolution_and_explicit_proxy_assumptions():
    r = power_description(11, 0.08896108640865788)
    assert r["minimum_raw_exact_p_if_all_nonzero"] == 1 / 2048
    assert r["minimum_corrected_exact_p_if_all_nonzero"] == 2 / 2048
    effects = r["effect_sensitivity"]
    assert 0.10 < effects[0]["approx_mean_block_effect_for_80pct_power"] < 0.11
    assert (
        effects[0]["approx_mean_block_effect_for_80pct_power"]
        > effects[-1]["approx_mean_block_effect_for_80pct_power"]
    )
    assert "not exact" in r["power_method"] and "post-hoc" in r["reference"]
    assert power_description(0, None)["minimum_raw_exact_p_if_all_nonzero"] is None
    assert "effect_sensitivity" not in power_description(11, None)


def test_v3_family_keeps_secondary_without_promoting_it_to_primary():
    model = LlmSettings(
        model="fixture/model",
        release_date=date(2020, 1, 1),
        release_source="https://example.invalid",
    )
    plan = FactorPlan(protocol_version="v3-settlement", llm=model)
    rows = []
    for i in range(11):
        for arm, ret in [("library", 0.0), ("random", 0.1), ("llm", 0.2)]:
            rows.append(
                {
                    "path": "fixture",
                    "episode": 113 + i,
                    "arm": arm,
                    "status": "ok",
                    "return": ret,
                    "drawdown": 0.1,
                    "holdout_start": f"2030-{i + 1:02d}-01",
                    "holdout_end": f"2030-{i + 1:02d}-20",
                }
            )
    gate = factor_gate(rows, plan)
    assert gate["family_size"] == gate["planned_family_size"] == 2
    comparisons = {x["arm"]: x for x in gate["comparisons"]}
    assert comparisons["llm"]["comparison_role"] == "primary"
    assert comparisons["llm"]["p_corrected"] == 2 * comparisons["llm"]["p_raw"]
    assert comparisons["random"]["outcome"] == "reference_only"
    assert comparisons["random"]["numeric_gate_outcome"] == "research_threshold_passed"
