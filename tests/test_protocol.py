import pytest

from alpha_evidence_lab.ledger import Ledger
from alpha_evidence_lab.models import ARMS, Candidate, Hypothesis, StudyPlan
from alpha_evidence_lab.search import EvidenceAgent, Memory, ResearchTools, visible_memory


@pytest.fixture
def ledger(tmp_path, plan):
    return Ledger(tmp_path / "audit.sqlite", "p", "d", "c", budget=plan.budget)


def fill(ledger, budget=2):
    ledger.start_batch("batch")
    for arm in ARMS:
        for i in range(budget):
            ledger.record_attempt("batch", arm, budget, f"c{i}", "ok", float(i), {})


def test_canonical_catalog_rejects_unsupported_params(plan):
    with pytest.raises(ValueError):
        plan.candidate("invented_100_999")
    with pytest.raises(ValueError):
        Candidate(id="x", fast=2.9, slow=3)
    with pytest.raises(ValueError):
        Candidate(id="x", fast=3, slow=3)
    with pytest.raises(ValueError):
        StudyPlan(budget=99)
    with pytest.raises(ValueError):
        StudyPlan(data_seeds=(1, 1))


def test_failed_and_duplicate_attempts_consume_budget(ledger, paths, plan):
    ledger.start_batch("batch")
    tools = ResearchTools(paths[0].known(49), plan, ledger, "batch", "memory")
    assert tools.evaluate(Hypothesis(candidate_id="invented", rationale="x"))["status"] == "failed"
    assert tools.evaluate(Hypothesis(candidate_id="invented", rationale="x"))["status"] == "failed"
    with pytest.raises(ValueError):
        tools.evaluate(Hypothesis(candidate_id=plan.catalog[0].id, rationale="x"))
    assert len(ledger.trials("batch", "memory")) == 2
    assert ledger.verify()


def test_repeat_candidate_is_not_a_free_retry(ledger, paths, plan):
    ledger.start_batch("batch")
    tools = ResearchTools(paths[0].known(49), plan, ledger, "batch", "random")
    proposal = Hypothesis(candidate_id=plan.catalog[0].id, rationale="test")
    assert tools.evaluate(proposal)["status"] == "ok"
    assert tools.evaluate(proposal)["status"] == "failed"


def test_reveal_protocol_and_frozen_selection(ledger):
    ledger.start_batch("batch")
    with pytest.raises(ValueError):
        ledger.frozen_winners("batch")
    with pytest.raises(ValueError):
        ledger.reveal("batch", dict.fromkeys(ARMS, {}))
    with pytest.raises(ValueError):
        ledger.freeze("batch", {"ordered": "c1"}, 2)
    for arm in ARMS:
        for i in range(2):
            ledger.record_attempt("batch", arm, 2, f"c{i}", "ok", float(i), {})
    with pytest.raises(ValueError):
        ledger.freeze("batch", dict.fromkeys(ARMS, "untested"), 2)
    ledger.freeze("batch", dict.fromkeys(ARMS, "c1"), 2)
    assert set(ledger.frozen_winners("batch")) == set(ARMS)
    with pytest.raises(ValueError):
        ledger.record_attempt("batch", "ordered", 2, "c3", "ok", 3.0, {})
    with pytest.raises(ValueError):
        ledger.reveal("batch", {"ordered": {}})
    ledger.reveal("batch", dict.fromkeys(ARMS, {"return": 0.1}))
    with pytest.raises(ValueError):
        ledger.frozen_winners("batch")
    with pytest.raises(ValueError):
        ledger.reveal("batch", dict.fromkeys(ARMS, {}))
    assert ledger.verify()


def test_budget_cannot_be_increased_by_caller(ledger):
    ledger.start_batch("batch")
    with pytest.raises(ValueError):
        ledger.record_attempt("batch", "memory", 100, "c", "ok", 0.0, {})
    with pytest.raises(ValueError):
        ledger.record_attempt("batch", "memory", 2, "c", "ok", float("inf"), {})
    assert ledger.trials("batch", "memory") == []


@pytest.mark.parametrize("target", ["events", "trials", "batches", "identity"])
def test_tamper_detection_including_state_projection(ledger, target):
    fill(ledger)
    with ledger.connection() as db:
        if target == "events":
            db.execute("UPDATE events SET payload='{}' WHERE seq=2")
        elif target == "trials":
            db.execute("UPDATE trials SET score=999 WHERE arm='memory'")
        elif target == "batches":
            db.execute("UPDATE batches SET state='revealed'")
        else:
            db.execute("UPDATE identity SET code='changed'")
    with pytest.raises(ValueError):
        ledger.verify()


def test_identity_and_checkpoint(ledger, plan):
    with pytest.raises(ValueError):
        Ledger(ledger.path, "different", "d", "c", budget=plan.budget)
    fill(ledger)
    head = ledger.verify()
    with pytest.raises(ValueError):
        ledger.verify("wrong")
    assert ledger.verify(head) == head


def test_memory_cannot_see_future_or_other_market(paths):
    day = paths[0].bars[50].day
    memories = [
        Memory("old", "c", 0.1, paths[0].bars[10].day, "seed_11"),
        Memory("future", "c", 99.0, paths[0].bars[80].day, "seed_11"),
        Memory("foreign", "c", 99.0, paths[0].bars[10].day, "seed_23"),
        Memory("same_day", "c", 99.0, day, "seed_11"),
    ]
    assert [row.id for row in visible_memory(memories, day, "seed_11")] == ["old"]


def test_all_failed_agent_does_not_freeze(ledger, paths, plan, monkeypatch):
    ledger.start_batch("batch")
    tools = ResearchTools(paths[0].known(49), plan, ledger, "batch", "ordered")
    monkeypatch.setattr(
        "alpha_evidence_lab.search.evaluate",
        lambda *args: (_ for _ in ()).throw(ValueError("failure")),
    )
    with pytest.raises(ValueError):
        EvidenceAgent(plan, "ordered", 1).search(tools, [])
    assert all(row["status"] == "failed" for row in ledger.trials("batch", "ordered"))
