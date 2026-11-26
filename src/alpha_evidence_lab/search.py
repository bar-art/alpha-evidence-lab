"""有界研究Agent：分析→提议→统一评估→反思；工具只看到开发段。"""

import random
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

from .data import Bar
from .evaluator import evaluate, selection_score
from .ledger import Ledger
from .models import LLM_ARM, Hypothesis, StudyPlan

if TYPE_CHECKING:
    from .llm import OpenRouterClient

NO_EVIDENCE = "none"
LLM_SYSTEM = (
    "You choose which pre-registered dual moving-average strategy to evaluate next. "
    "Each candidate is long when the fast simple moving average of prior closes is above the "
    "slow one, otherwise in cash; trades at the next open with costs. You see anonymised "
    "validation scores from earlier, already-finished periods of the same price series and "
    "the results of attempts already made in this batch. Score = validation total return "
    "minus 2 x max drawdown; higher is better. Every proposal consumes one attempt, including "
    'duplicates. Cite only evidence_ids that appear in the input (use ["none"] when there is '
    "no prior evidence). Return JSON only."
)


@dataclass(frozen=True)
class Memory:
    id: str
    candidate: str
    score: float
    available_on: date
    data_scope: str


def visible_memory(memories: list[Memory], cutoff: date, scope: str) -> list[Memory]:
    return [item for item in memories if item.available_on < cutoff and item.data_scope == scope]


class ResearchTools:
    def __init__(
        self, development: tuple[Bar, ...], plan: StudyPlan, ledger: Ledger, batch: str, arm: str
    ) -> None:
        self.development, self.plan, self.ledger = development, plan, ledger
        self.batch, self.arm = batch, arm
        self.seen: set[str] = set()

    def evaluate(self, proposal: Hypothesis, refusal: str | None = None) -> dict[str, Any]:
        # 在实际计算前检查预算；所有无效提议均留下失败记录并消耗一次预算。
        if len(self.ledger.trials(self.batch, self.arm)) >= self.plan.budget:
            raise ValueError("本方法已用尽预算")
        payload: dict[str, Any]
        try:
            if refusal:
                raise ValueError(refusal)
            candidate = self.plan.candidate(proposal.candidate_id)
            if candidate.id in self.seen:
                raise ValueError("重复候选不获得免费重试")
            self.seen.add(candidate.id)
            train = evaluate(
                self.development,
                candidate,
                max(c.slow for c in self.plan.catalog),
                self.plan.train_sessions,
                self.plan.cost_bps,
                self.plan.stamp_duty_schedule,
            )
            validation = evaluate(
                self.development,
                candidate,
                self.plan.train_sessions,
                len(self.development),
                self.plan.cost_bps,
                self.plan.stamp_duty_schedule,
            )
            score = selection_score(validation)
            payload = {
                "proposal": proposal.model_dump(),
                "train": train.to_dict(),
                "validation": validation.to_dict(),
            }
            status = "ok"
        except ValueError as error:
            score, status, payload = None, "failed", {"error": str(error)}
        self.ledger.record_attempt(
            self.batch, self.arm, self.plan.budget, proposal.candidate_id, status, score, payload
        )
        return {
            "candidate": proposal.candidate_id,
            "status": status,
            "score": score,
            "payload": payload,
        }


class EvidenceAgent:
    def __init__(
        self, plan: StudyPlan, arm: str, seed: int, client: "OpenRouterClient | None" = None
    ) -> None:
        self.plan, self.arm, self.rng, self.client = plan, arm, random.Random(seed), client
        if arm not in plan.arms:
            raise ValueError("未知搜索策略")
        if arm == LLM_ARM and client is None:
            raise ValueError("大模型搜索方法需要模型客户端")

    def _llm_search(self, tools: ResearchTools, memories: list[Memory]) -> None:
        assert self.client is not None
        # 证据ID匿名化：原ID含证券代码和批次，不发给模型，避免它凭记忆识别真实行情。
        aliases = {f"e{number}": item for number, item in enumerate(memories, 1)}
        catalog = [
            {"id": item.id, "fast": item.fast, "slow": item.slow} for item in self.plan.catalog
        ]
        schema = {
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string", "enum": [item["id"] for item in catalog]},
                "rationale": {"type": "string"},
                # 引用只能取自输入中的匿名证据编号；没有证据时只能写占位值 none（随后剔除）。
                "evidence_ids": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(aliases) or [NO_EVIDENCE]},
                },
            },
            "required": ["candidate_id", "rationale", "evidence_ids"],
            "additionalProperties": False,
        }
        for remaining in range(self.plan.budget, 0, -1):
            attempts = [
                {"candidate": row["candidate"], "status": row["status"], "score": row["score"]}
                for row in tools.ledger.trials(tools.batch, tools.arm)
            ]
            context = {
                "catalog": catalog,
                "prior_validation_evidence": [
                    {"evidence_id": alias, "candidate": item.candidate, "score": item.score}
                    for alias, item in aliases.items()
                ],
                "attempts_in_this_batch": attempts,
                "attempts_remaining": remaining,
            }
            output = self.client.complete(LLM_SYSTEM, context, schema, "proposal")
            cited = [str(item) for item in output.get("evidence_ids") or [] if item != NO_EVIDENCE]
            unknown = [item for item in cited if item not in aliases]
            rationale = str(output.get("rationale") or "").strip()[:500] or "（模型未给出理由）"
            tools.evaluate(
                Hypothesis(
                    candidate_id=str(output.get("candidate_id", ""))[:40] or "missing",
                    rationale=rationale,
                    evidence_ids=tuple(aliases[item].id for item in cited if item in aliases),
                ),
                refusal=f"引用了输入中不存在的证据：{unknown[:3]}" if unknown else None,
            )

    def search(self, tools: ResearchTools, memories: list[Memory]) -> str:
        if self.arm == LLM_ARM:
            self._llm_search(tools, memories)
            return self._winner(tools)
        catalog = list(self.plan.catalog)
        evidence: dict[str, list[Memory]] = {}
        if self.arm == "random":
            self.rng.shuffle(catalog)
        elif self.arm == "memory":
            for memory in memories:
                evidence.setdefault(memory.candidate, []).append(memory)

            # 先试历史验证表现较好或尚未覆盖的候选；不是LLM或在线训练。
            def priority(candidate_id: str) -> float:
                rows = evidence.get(candidate_id, [])
                return sum(row.score for row in rows) / len(rows) if rows else 0.0

            catalog.sort(key=lambda candidate: (-priority(candidate.id), candidate.id))
        for candidate in catalog[: self.plan.budget]:
            citations = tuple(item.id for item in evidence.get(candidate.id, []))
            tools.evaluate(
                Hypothesis(
                    candidate_id=candidate.id,
                    rationale=f"{self.arm}策略在预注册目录内选择；共用相同评估预算。",
                    evidence_ids=citations,
                )
            )
        return self._winner(tools)

    @staticmethod
    def _winner(tools: ResearchTools) -> str:
        valid = [
            row for row in tools.ledger.trials(tools.batch, tools.arm) if row["status"] == "ok"
        ]
        if not valid:
            raise ValueError("预算耗尽且没有有效候选，禁止揭晓测试集")
        return str(max(valid, key=lambda row: (row["score"], row["candidate"]))["candidate"])
