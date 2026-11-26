"""预注册→等预算搜索→批量冻结→一次揭晓；完整保存每个尝试。"""

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .data import PricePath, synthetic_paths
from .evaluator import evaluate
from .inference import EXACT_MAX, MC_DRAWS, paired_gate
from .ledger import Ledger
from .llm import OpenRouterClient
from .models import LLM_ARM, StudyPlan, digest
from .search import EvidenceAgent, Memory, ResearchTools, visible_memory


def source_digest() -> str:
    root = Path(__file__).parent
    return digest(
        {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}
    )


def run_study(
    plan: StudyPlan,
    output: Path,
    paths: tuple[PricePath, ...] | None = None,
    llm_client: OpenRouterClient | None = None,
    llm_cache: Path | None = None,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ValueError("输出目录非空，请为新实验指定新目录")
    paths = paths if paths is not None else synthetic_paths(plan)
    if len({path.id for path in paths}) != len(paths) or not paths:
        raise ValueError("路径ID重复或数据为空")
    episode_length = plan.train_sessions + plan.validation_sessions + plan.holdout_sessions
    if any(len(path.bars) < plan.episodes * episode_length for path in paths):
        raise ValueError("数据长度不足；不自动缩短或取消holdout")
    fingerprint = digest({path.id: path.sha256 for path in paths})
    code_hash = source_digest()
    cluster_unit = (
        ("paths" if all(p.kind == "synthetic" for p in paths) else "time_blocks")
        if plan.cluster_mode == "auto"
        else plan.cluster_mode
    )
    if cluster_unit == "paths" and any(p.kind != "synthetic" for p in paths):
        intervals = sorted((p.bars[0].day, p.bars[-1].day) for p in paths)
        if any(a[1] >= b[0] for a, b in zip(intervals, intervals[1:], strict=False)):
            raise ValueError("真实路径时间窗口重叠，不得按股票独立聚类；请使用time_blocks")
    registration = {
        "plan": plan.model_dump(mode="json"),
        "data_hash": fingerprint,
        "code_hash": code_hash,
        "arms": plan.arms,
        "metric": "net_total_return",
        "gate": "cluster_sign_flip_bonferroni_one_sided",
        "sign_flip": {
            "exact_max_nonzero_clusters": EXACT_MAX,
            "monte_carlo_draws": MC_DRAWS,
            "monte_carlo_seed": plan.seed,
        },
        "cluster_unit": cluster_unit,
        "warmup_episodes": plan.warmup_episodes,
        "minimum_clusters": 6,
        "kind": "local_preregistration_not_public_timestamp",
    }
    study_id = digest(registration)[:20]
    output.mkdir(parents=True, exist_ok=True)
    from .models import canonical

    (output / "registration.json").write_text(canonical(registration), encoding="utf-8")
    ledger = Ledger(
        output / "audit.sqlite",
        digest(plan.model_dump(mode="json")),
        fingerprint,
        code_hash,
        budget=plan.budget,
        arms=plan.arms,
    )
    if plan.llm and llm_client is None:
        # 缓存放在输出目录之外时，可在新目录中免费重跑同一研究。
        llm_client = OpenRouterClient(
            plan.llm, llm_cache or output / "llm_cache.jsonl", seed=plan.seed
        )
    rows, trials, batches = [], [], []
    memories: list[Memory] = []
    for path in paths:
        for episode in range(plan.episodes):
            start = episode * episode_length
            train_end = start + plan.train_sessions
            validation_end = train_end + plan.validation_sessions
            holdout_end = validation_end + plan.holdout_sessions
            batch = f"{path.id}:ep{episode}"
            ledger.start_batch(batch)
            development = path.known(validation_end - 1)[start:validation_end]
            prior = visible_memory(memories, development[0].day, path.id)
            winners = {}
            for arm in plan.arms:
                tools = ResearchTools(development, plan, ledger, batch, arm)
                seed = int(digest({"seed": plan.seed, "path": path.id, "ep": episode})[:8], 16)
                client = llm_client if arm == LLM_ARM else None
                winners[arm] = EvidenceAgent(plan, arm, seed, client).search(tools, prior)
                for record in ledger.trials(batch, arm):
                    trials.append({"path": path.id, "episode": episode, "arm": arm, **record})
            commitment = ledger.freeze(batch, winners, plan.budget)
            frozen = ledger.frozen_winners(batch)
            # 第一次把当前holdout数据传给评估器发生在全部方法冻结以后。
            known = path.known(holdout_end - 1)
            reveals = {}
            for arm, candidate_id in frozen.items():
                candidate = plan.candidate(candidate_id)
                result = evaluate(
                    known,
                    candidate,
                    validation_end,
                    holdout_end,
                    plan.cost_bps,
                    plan.stamp_duty_schedule,
                )
                reveals[arm] = {"candidate": candidate_id, "result": result.to_dict()}
                rows.append(
                    {
                        "path": path.id,
                        "episode": episode,
                        "arm": arm,
                        "status": "ok",
                        "candidate": candidate_id,
                        "return": result.total_return,
                        "drawdown": result.max_drawdown,
                        "costs": result.costs,
                        "holdout_start": str(path.bars[validation_end].day),
                        "holdout_end": str(path.bars[holdout_end - 1].day),
                    }
                )
            ledger.reveal(batch, reveals)
            batches.append(
                {
                    "batch": batch,
                    "commitment": commitment,
                    "winners": winners,
                    "visible_memory": [item.id for item in prior],
                }
            )
            # 只用此前验证集证据更新下一episode的搜索优先级；holdout从不进入搜索记忆。
            for record in ledger.trials(batch, "memory"):
                if record["status"] == "ok":
                    memories.append(
                        Memory(
                            f"{batch}:{record['candidate']}",
                            record["candidate"],
                            record["score"],
                            path.bars[holdout_end - 1].day,
                            path.id,
                        )
                    )
    inference = paired_gate(
        rows,
        plan.arms,
        plan.bootstrap_samples,
        plan.seed,
        plan.alpha,
        plan.max_holdout_drawdown,
        cluster_unit=cluster_unit,
        warmup_episodes=plan.warmup_episodes,
    )
    head = ledger.verify()
    return {
        "study_id": study_id,
        "registration": registration,
        "data_manifest": [
            {"id": path.id, "sha256": path.sha256, "kind": path.kind, "source": path.source}
            for path in paths
        ],
        "batches": batches,
        "trials": trials,
        "results": rows,
        "gate": inference,
        "memory": [{**asdict(item), "available_on": str(item.available_on)} for item in memories],
        "ledger_head": head,
        "llm_usage": llm_client.usage() if llm_client else None,
        "scope": "研究搜索器比较；无实盘、无LLM训练；不是完整A股撮合",
        "limitations": [
            "默认为人工行情；阈值通过不能外推为市场盈利。",
            "单资产、碎股、开盘成交；无停牌、涨跌停、税费历史或公司行动模型。",
            "本地注册和哈希不是第三方时间戳或不可篡改证明。",
            "封存是可信进程中的API隔离，不是防本机恶意操作者的安全沙箱。",
            "精确符号翻转依赖独立簇与零假设符号对称；时间分组本身不证明这些假设。",
            "多重校正仅覆盖预注册的方法比较，不校正跨项目反复选实验的所有研究自由度。",
            "memory是确定性的历史验证证据优先级，不是LLM自我改进。",
            *(
                [
                    "llm方法只看到匿名化的候选与历史验证得分，不含证券代码和日期；"
                    "模型仍可能从得分模式中利用训练语料里的市场知识，"
                    f"发布日期 {plan.llm.release_date} 之前的测试段不能排除这种前视。",
                    "模型输出在temperature=0时仍可能不确定；复现依赖保存的响应缓存。",
                ]
                if plan.llm
                else []
            ),
        ],
        "disclaimer": "仅供研究，不构成投资建议",
    }
