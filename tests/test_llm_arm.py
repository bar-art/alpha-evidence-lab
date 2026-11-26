import json
from datetime import date

import pytest

from alpha_evidence_lab.cli import main
from alpha_evidence_lab.llm import LlmBudgetError, OpenRouterClient
from alpha_evidence_lab.models import LlmSettings, StudyPlan
from alpha_evidence_lab.study import run_study

SETTINGS = LlmSettings(
    model="openai/gpt-oss-120b",
    release_date=date(2025, 8, 5),
    release_source="https://openai.com/index/introducing-gpt-oss/",
    max_calls=500,
    max_cost_usd=0.5,
)


def fake_transport(sent):
    """按目录顺序挑第一个本批次未试过的候选；第一次调用故意引用不存在的证据。"""

    def transport(body, api_key):
        assert api_key == "test-key"
        sent.append(body)
        context = json.loads(body["messages"][1]["content"])
        tried = {row["candidate"] for row in context["attempts_in_this_batch"]}
        choice = next(item["id"] for item in context["catalog"] if item["id"] not in tried)
        evidence = [row["evidence_id"] for row in context["prior_validation_evidence"]][:1]
        if len(sent) == 1:
            evidence = ["invented"]
        content = {"candidate_id": choice, "rationale": "test", "evidence_ids": evidence}
        return {
            "model": body["model"],
            "choices": [{"message": {"content": json.dumps(content)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.0001},
        }

    return transport


def plan_with_llm(**overrides):
    return StudyPlan(llm=SETTINGS.model_copy(update=overrides), episodes=3)


def test_study_with_llm_arm_is_audited_and_anonymised(tmp_path):
    sent: list[dict] = []
    plan = plan_with_llm()
    client = OpenRouterClient(
        plan.llm,
        tmp_path / "cache.jsonl",
        seed=1,
        api_key="test-key",
        transport=fake_transport(sent),
    )
    result = run_study(plan, tmp_path / "out", llm_client=client)
    assert plan.arms == ("ordered", "random", "memory", "llm")
    assert (
        result["registration"]["arms"] == list(plan.arms)
        or result["registration"]["arms"] == plan.arms
    )
    assert result["gate"]["family_size"] == 3
    assert {c["arm"] for c in result["gate"]["comparisons"]} == {"random", "memory", "llm"}
    llm_trials = [t for t in result["trials"] if t["arm"] == "llm"]
    assert len(llm_trials) == plan.budget * plan.episodes * len(plan.data_seeds)
    # 编造的证据引用记为失败尝试，并消耗预算。
    assert (
        llm_trials[0]["status"] == "failed" and "不存在的证据" in llm_trials[0]["payload"]["error"]
    )
    assert result["llm_usage"]["api_calls"] == len(sent)
    assert result["llm_usage"]["cost_usd"] == pytest.approx(0.0001 * len(sent))
    # 发给模型的内容不含路径ID（证券代码）、日期或密钥。
    text = json.dumps(sent, ensure_ascii=False)
    assert "syn_" not in text and "2024" not in text and "test-key" not in text
    assert "authorization" not in text.lower()
    assert sent[0]["reasoning"] == {"effort": "low"}
    assert sent[0]["provider"] == {"require_parameters": True}


def test_provider_pinning_is_registered(tmp_path):
    sent: list[dict] = []
    plan = plan_with_llm(provider="openai/flex", temperature=None, reasoning_effort="minimal")
    client = OpenRouterClient(plan.llm, tmp_path / "p.jsonl", 1, "test-key", fake_transport(sent))
    result = run_study(plan, tmp_path / "out", llm_client=client)
    assert sent[0]["provider"] == {
        "require_parameters": True,
        "only": ["openai/flex"],
        "allow_fallbacks": False,
    }
    assert "temperature" not in sent[0] and sent[0]["reasoning"] == {"effort": "minimal"}
    assert result["llm_usage"]["provider_routing"] == "openai/flex"
    assert result["registration"]["plan"]["llm"]["provider"] == "openai/flex"
    with pytest.raises(ValueError):
        LlmSettings(**{**SETTINGS.model_dump(), "provider": "Bad Provider"})


def test_cache_makes_rerun_free_and_identical(tmp_path):
    sent: list[dict] = []
    cache = tmp_path / "cache.jsonl"
    plan = plan_with_llm()
    first = run_study(
        plan,
        tmp_path / "a",
        llm_client=OpenRouterClient(plan.llm, cache, 1, "test-key", fake_transport(sent)),
    )
    calls = len(sent)
    second = run_study(
        plan,
        tmp_path / "b",
        llm_client=OpenRouterClient(plan.llm, cache, 1, "test-key", fake_transport(sent)),
    )
    assert len(sent) == calls and second["llm_usage"]["api_calls"] == 0
    usage = first["llm_usage"]
    assert usage["api_calls"] == calls
    # 同一次研究中完全相同的请求也走缓存；第二次全部命中。
    assert second["llm_usage"]["cache_hits"] == usage["api_calls"] + usage["cache_hits"]
    assert first["results"] == second["results"]
    assert "test-key" not in cache.read_text()


def test_budget_limits_and_missing_key_stop_the_study(tmp_path):
    sent: list[dict] = []
    plan = plan_with_llm(max_calls=2)
    client = OpenRouterClient(plan.llm, tmp_path / "c.jsonl", 1, "test-key", fake_transport(sent))
    with pytest.raises(LlmBudgetError, match="次数"):
        run_study(plan, tmp_path / "out", llm_client=client)
    costly = OpenRouterClient(
        plan_with_llm(max_cost_usd=0.00005).llm,
        tmp_path / "d.jsonl",
        1,
        "test-key",
        fake_transport([]),
    )
    with pytest.raises(LlmBudgetError, match="费用"):
        run_study(plan_with_llm(max_cost_usd=0.00005), tmp_path / "out2", llm_client=costly)
    keyless = OpenRouterClient(plan.llm, tmp_path / "e.jsonl", 1, api_key="")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        keyless.complete("s", {}, {"type": "object"}, "x")


def test_malformed_model_output_is_rejected(tmp_path):
    def broken(body, api_key):
        return {"choices": [{"message": {"content": "not json"}}], "usage": {}}

    client = OpenRouterClient(SETTINGS, tmp_path / "f.jsonl", 1, "test-key", broken)
    with pytest.raises(ValueError, match="JSON"):
        client.complete("s", {}, {"type": "object"}, "x")


def test_cli_verify_accepts_llm_study(tmp_path, monkeypatch):
    sent: list[dict] = []
    plan = plan_with_llm()
    import alpha_evidence_lab.llm as llm

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(llm, "http_transport", fake_transport(sent))
    monkeypatch.setattr(
        llm.OpenRouterClient.__init__,
        "__defaults__",
        (None, fake_transport(sent)),
    )
    config = tmp_path / "plan.json"
    config.write_text(plan.model_dump_json())
    out = tmp_path / "study"
    assert main(["demo", "--config", str(config), "--output", str(out)]) == 0
    assert main(["verify", "--output", str(out)]) == 0
    assert sent


def test_format_repair_does_not_consume_search_budget(tmp_path):
    sent: list[dict] = []
    good = fake_transport(sent)
    state = {"n": 0}

    def wrong_key_first(body, api_key):
        state["n"] += 1
        if state["n"] == 1:
            sent.append(body)
            bad = {"candidate": "ma_2_8", "evidence_ids": []}
            return {"choices": [{"message": {"content": json.dumps(bad)}}], "usage": {}}
        return good(body, api_key)

    plan = plan_with_llm()
    cache = tmp_path / "r.jsonl"
    client = OpenRouterClient(plan.llm, cache, 1, "test-key", wrong_key_first)
    result = run_study(plan, tmp_path / "out", llm_client=client)
    assert result["llm_usage"]["format_repairs"] == 1
    repair = sent[1]["messages"]
    assert repair[-2]["role"] == "assistant" and "invalid" in repair[-1]["content"]
    llm_trials = [t for t in result["trials"] if t["arm"] == "llm"]
    assert len(llm_trials) == plan.budget * plan.episodes * len(plan.data_seeds)
    # 缓存中只有合规结果；以首次请求为键，重跑全部命中。
    entries = [json.loads(line) for line in cache.read_text().splitlines()]
    assert all("candidate_id" in e["output"] for e in entries)
    rerun = OpenRouterClient(plan.llm, cache, 1, "test-key", fake_transport([]))
    again = run_study(plan, tmp_path / "again", llm_client=rerun)
    assert again["llm_usage"]["api_calls"] == 0 and again["results"] == result["results"]


def test_content_filter_false_positive_is_retried_then_stops(tmp_path):
    from alpha_evidence_lab.llm import ProviderFlagged

    sent: list[dict] = []
    good = fake_transport(sent)
    flags = {"left": 1}

    def flag_once(body, api_key):
        if flags["left"]:
            flags["left"] -= 1
            raise ProviderFlagged("OpenRouter HTTP 400: flagged")
        return good(body, api_key)

    client = OpenRouterClient(SETTINGS, tmp_path / "f.jsonl", 1, "test-key", flag_once)
    context = {
        "catalog": [{"id": "ma_2_8"}],
        "attempts_in_this_batch": [],
        "prior_validation_evidence": [],
    }
    schema = {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string", "enum": ["ma_2_8"]},
            "rationale": {"type": "string"},
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["candidate_id", "rationale", "evidence_ids"],
    }
    assert client.complete("s", context, schema, "p")["candidate_id"] == "ma_2_8"
    assert client.usage()["content_filter_retries"] == 1 and client.calls == 2

    def always_flag(body, api_key):
        raise ProviderFlagged("OpenRouter HTTP 400: flagged")

    stubborn = OpenRouterClient(SETTINGS, tmp_path / "g.jsonl", 1, "test-key", always_flag)
    with pytest.raises(ProviderFlagged):
        stubborn.complete("s", context, schema, "p")
    assert stubborn.calls == 3
