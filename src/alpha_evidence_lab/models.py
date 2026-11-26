"""预注册配置和候选目录；未知字段、浮点窗口和不完整区间直接拒绝。"""

import hashlib
import json
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Candidate(Contract):
    id: str = Field(min_length=1, max_length=40, pattern=r"^[a-zA-Z0-9_-]+$")
    fast: int = Field(ge=2, strict=True)
    slow: int = Field(ge=3, strict=True)

    @model_validator(mode="after")
    def ordered(self) -> "Candidate":
        if self.fast >= self.slow:
            raise ValueError("快窗口必须小于慢窗口")
        return self


CATALOG = tuple(
    Candidate(id=f"ma_{fast}_{slow}", fast=fast, slow=slow)
    for fast, slow in [(2, 8), (3, 10), (4, 12), (5, 16), (6, 20), (8, 24)]
)
ARMS = ("ordered", "random", "memory")
LLM_ARM = "llm"


class LlmSettings(Contract):
    """可选的大模型搜索方法；只看匿名化的候选、历史验证得分和本批次尝试结果。"""

    model: str = Field(min_length=3, max_length=100, pattern=r"^[a-z0-9._/:-]+$")
    # 模型发布日期（一手来源），写入报告，用于判断前视偏差边界。
    release_date: date
    release_source: str = Field(min_length=8, max_length=300)
    max_calls: int = Field(default=2000, ge=1, le=20000)
    max_cost_usd: float = Field(default=1.0, gt=0, le=5)
    # 推理型模型（如 gpt-5-nano）不接受 temperature；设为 null 时不发送。
    temperature: float | None = Field(default=0.0, ge=0, le=1)
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = "low"
    # 服务商或端点标签（如 openai/flex）：固定单一端点且禁止回退；为空则由 OpenRouter 路由。
    provider: str | None = Field(default=None, pattern=r"^[a-z0-9._-]+(/[a-z0-9._-]+)?$")


class StudyPlan(Contract):
    question: str = "历史失败记忆是否帮助搜索器在相同尝试预算下选择更好的双均线策略？"
    data_seeds: tuple[int, ...] = (11, 23, 37, 53, 71, 89)
    episodes: int = Field(default=7, ge=2, le=50)
    train_sessions: int = Field(default=50, ge=30)
    validation_sessions: int = Field(default=20, ge=5)
    holdout_sessions: int = Field(default=20, ge=5)
    budget: int = Field(default=3, ge=1)
    cost_bps: float = Field(default=7, ge=0, le=100)
    # 卖出分段印花税（生效日, bps），叠加在 cost_bps 之上；A股示例见 config/ashare.json。
    stamp_duty_schedule: tuple[tuple[date, float], ...] = ()
    seed: int = Field(default=20261124, ge=0)
    bootstrap_samples: int = Field(default=1999, ge=199, le=100000)
    alpha: float = Field(default=0.05, gt=0, lt=0.5)
    max_holdout_drawdown: float = Field(default=0.25, gt=0, le=1)
    catalog: tuple[Candidate, ...] = CATALOG
    warmup_episodes: int = Field(default=1, ge=0)
    cluster_mode: Literal["auto", "paths", "time_blocks"] = "auto"
    metric: Literal["net_total_return"] = "net_total_return"
    llm: LlmSettings | None = None

    @property
    def arms(self) -> tuple[str, ...]:
        return (*ARMS, LLM_ARM) if self.llm else ARMS

    @model_validator(mode="after")
    def consistency(self) -> "StudyPlan":
        if self.warmup_episodes >= self.episodes:
            raise ValueError("暖机段不能覆盖全部episode")
        if not self.data_seeds or len(set(self.data_seeds)) != len(self.data_seeds):
            raise ValueError("数据种子必须非空且唯一")
        if any(seed < 0 for seed in self.data_seeds):
            raise ValueError("数据种子必须非负")
        if not self.catalog or self.budget > len(self.catalog):
            raise ValueError("预算不能超过候选目录大小")
        if len({item.id for item in self.catalog}) != len(self.catalog):
            raise ValueError("候选ID重复")
        if len({(item.fast, item.slow) for item in self.catalog}) != len(self.catalog):
            raise ValueError("候选参数重复")
        days = [day for day, _ in self.stamp_duty_schedule]
        if days != sorted(set(days)) or any(
            not 0 <= bps <= 100 for _, bps in self.stamp_duty_schedule
        ):
            raise ValueError("印花税分段生效日须严格递增，bps 在 0 到 100 之间")
        if max(item.slow for item in self.catalog) >= self.train_sessions:
            raise ValueError("训练历史不足以暖机最大窗口")
        return self

    def candidate(self, candidate_id: str) -> Candidate:
        for item in self.catalog:
            if item.id == candidate_id:
                return item
        raise ValueError("提议不在预注册候选目录中")


class Hypothesis(Contract):
    candidate_id: str
    rationale: str = Field(min_length=1, max_length=500)
    evidence_ids: tuple[str, ...] = ()


class FactorPlan(Contract):
    max_exit_delay: int = Field(default=10, ge=0, le=30)
    question: str = (
        "At equal trial budgets, do LLM price-volume factors "
        "outperform fixed-library and random proposals out of sample?"
    )
    budget: int = Field(default=5, ge=1, le=10)
    horizon: int = Field(default=5, ge=1, le=20)
    dev_sessions: int = Field(default=120, ge=20)
    holdout_sessions: int = Field(default=20, ge=5)
    confirm_start: date = date(2025, 8, 8)
    cost_bps: float = Field(default=7, ge=0, le=100)
    stamp_duty_schedule: tuple[tuple[date, float], ...] = ()
    seed: int = Field(default=20261126, ge=0)
    alpha: float = Field(default=0.05, gt=0, lt=0.5)
    max_holdout_drawdown: float = Field(default=0.25, gt=0, le=1)
    minimum_paths: int = Field(default=6, ge=6)
    warmup_episodes: int = Field(default=0, ge=0)
    minimum_symbols: int = Field(default=30, ge=2)
    # Same maximum history for every method; DSL lookback is capped at 250.
    warmup_sessions: int = Field(default=250, ge=250, le=250)
    confirmation_only: bool = False
    max_episodes: int | None = Field(default=None, ge=1)
    llm: LlmSettings | None = None

    @property
    def arms(self) -> tuple[str, ...]:
        return ("library", "random", "llm") if self.llm else ("library", "random")

    @model_validator(mode="after")
    def factor_consistency(self) -> "FactorPlan":
        if self.holdout_sessions <= self.horizon + 1:
            raise ValueError("Test block must contain a completed forward holding period")
        days = [day for day, _ in self.stamp_duty_schedule]
        if days != sorted(set(days)) or any(
            not 0 <= bps <= 100 for _, bps in self.stamp_duty_schedule
        ):
            raise ValueError("Invalid factor stamp duty schedule")
        if self.llm and self.confirm_start <= self.llm.release_date:
            raise ValueError("Confirmation boundary must be after the declared model release")
        return self
