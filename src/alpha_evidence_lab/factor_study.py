"""Development-only search, joint freeze, then one reveal for each sealed time block."""

from pathlib import Path
from typing import Any

import numpy as np

from .factor_arms import CLASSIC_EXPRESSIONS, normalize, search_arm
from .factor_dsl import FIELDS, WINDOWS
from .factor_eval import evaluate_factor
from .factor_tree import FUNCTIONS
from .inference import paired_gate
from .ledger import Ledger
from .llm import MAX_FLAG_RETRIES, MAX_REPAIRS, LlmBudgetError, OpenRouterClient, fetch_pricing
from .models import FactorPlan, canonical, digest
from .panel import Panel
from .study import source_digest


def episodes(panel: Panel, plan: FactorPlan) -> list[dict[str, Any]]:
    first = plan.warmup_sessions + plan.dev_sessions + plan.horizon
    result: list[dict[str, Any]] = []
    for test_start in range(
        first, len(panel.dates) - plan.holdout_sessions + 1, plan.holdout_sessions
    ):
        phase = (
            "confirmation"
            if panel.dates[test_start] > np.datetime64(plan.confirm_start)
            else "exploration"
        )
        if plan.confirmation_only and phase != "confirmation":
            continue
        dev_stop = test_start - plan.horizon
        result.append(
            {
                "episode": (test_start - first) // plan.holdout_sessions,
                "warm_start": dev_stop - plan.dev_sessions - plan.warmup_sessions,
                "dev_start": dev_stop - plan.dev_sessions,
                "dev_stop": dev_stop,
                "test_start": test_start,
                "test_stop": test_start + plan.holdout_sessions,
                "phase": phase,
            }
        )
        if plan.max_episodes is not None and len(result) >= plan.max_episodes:
            break
    if not result:
        raise ValueError("No complete factor episodes in the requested sample")
    return result


def factor_gate(rows: list[dict[str, Any]], plan: FactorPlan) -> dict[str, Any]:
    return paired_gate(
        rows,
        plan.arms,
        1999,
        plan.seed,
        plan.alpha,
        plan.max_holdout_drawdown,
        plan.minimum_paths,
        cluster_unit="time_blocks",
        warmup_episodes=plan.warmup_episodes,
        baseline="library",
    )


def run_factor_study(
    plan: FactorPlan,
    panel: Panel,
    output: Path,
    *,
    client: OpenRouterClient | None = None,
    llm_cache: Path | None = None,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be a new directory")
    schedule = episodes(panel, plan)
    logical = len(schedule) * plan.budget if plan.llm else 0
    if plan.llm and client is None:
        pricing = fetch_pricing(plan.llm.model, plan.llm.provider)
        if pricing is None or any(
            not np.isfinite(value) or value < 0 for value in pricing.values()
        ):
            raise LlmBudgetError("Trusted token pricing unavailable: no paid factor request sent")
        client = OpenRouterClient(
            plan.llm, llm_cache or output / "llm_cache.jsonl", plan.seed, pricing=pricing
        )
    registration: dict[str, Any] = {
        "mode": "factor",
        "protocol_version": "v2-delayed-exit",
        "prompt_version": plan.prompt_version,
        "plan": plan.model_dump(mode="json"),
        "data_hash": panel.sha256,
        "code_hash": source_digest(),
        "arms": plan.arms,
        "episodes": schedule,
        "classic_library": CLASSIC_EXPRESSIONS,
        "selection_score": "development ICIR; max(score,id) tie break; "
        "unknown basket returns remain missing, do not invalidate finite ICIR",
        "random_grammar": {
            "depth_max": plan.max_depth,
            "nodes_max": plan.max_nodes,
            "lookback_max": plan.max_lookback,
            "outer_rank_depth_exempt": True,
            "shared_with_llm": True,
            "functions": FUNCTIONS,
            "fields": FIELDS,
            "windows": WINDOWS,
            "operators": ["add", "sub", "mul", "div", "neg"],
            "random_constants": [-2, -1, 1, 2],
            "constant_root": False,
            "duplicate_redraws": 50,
            "seed_derivation": "sha256(seed,episode,arm)",
        },
        "call_plan": {
            "logical_proposals": logical,
            "worst_case_calls": logical * (1 + MAX_REPAIRS) * (1 + MAX_FLAG_RETRIES),
            "max_cost_usd": plan.llm.max_cost_usd if plan.llm else 0,
            "pricing_usd_per_token": client.pricing if client else None,
        },
        "confirm_rule": (
            "test_start strictly after confirm_start; primary gate uses confirmation only"
        ),
        "boundary_rule": (
            "exits/marks strictly before label_stop; "
            "truncated delay budget unknown; no outside reads"
        ),
        "portfolio": ("long 1 + short 1, gross 2, open marks, side-specific delayed exits"),
        "cost_rule": "entry and exit notional costs on both legs; tax on short entry and long exit",
        "kind": "local_preregistration_not_public_timestamp",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "registration.json").write_text(canonical(registration))
    ledger = Ledger(
        output / "audit.sqlite",
        digest(registration["plan"]),
        panel.sha256,
        registration["code_hash"],
        budget=plan.budget,
        arms=plan.arms,
    )
    rows: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    memories: dict[str, list[dict[str, Any]]] = {arm: [] for arm in plan.arms}
    try:
        for ep in schedule:
            number = ep["episode"]
            batch = f"factor:ep{number}"
            ledger.start_batch(batch)
            # No test field, factor value or label is evaluated until freeze completes.
            development = panel.window(ep["warm_start"], ep["test_start"])
            winners = {}
            for arm in plan.arms:
                seed = int(digest({"seed": plan.seed, "episode": number, "arm": arm})[:8], 16)
                winner = search_arm(
                    arm,
                    development,
                    ep["dev_start"] - ep["warm_start"],
                    ep["dev_stop"] - ep["warm_start"],
                    ledger=ledger,
                    batch=batch,
                    budget=plan.budget,
                    horizon=plan.horizon,
                    cost_bps=plan.cost_bps,
                    stamp_duty_schedule=plan.stamp_duty_schedule,
                    seed=seed,
                    memories=memories[arm],
                    client=client if arm == "llm" else None,
                    minimum=plan.minimum_symbols,
                    max_exit_delay=plan.max_exit_delay,
                    prompt_version=plan.prompt_version,
                    max_depth=plan.max_depth,
                    max_nodes=plan.max_nodes,
                    max_lookback=plan.max_lookback,
                )
                winners[arm] = winner["candidate"]
                trials.extend(
                    {"episode": number, "arm": arm, **record}
                    for record in ledger.trials(batch, arm)
                )
            commitment = ledger.freeze(batch, winners, plan.budget)
            frozen = ledger.frozen_winners(batch)
            test = panel.window(ep["test_start"] - plan.warmup_sessions, ep["test_stop"])
            reveals = {}
            for arm, candidate in frozen.items():
                winner = next(
                    r
                    for r in ledger.trials(batch, arm)
                    if r["candidate"] == candidate and r["status"] == "ok"
                )
                expression = winner["payload"]["expression"]
                try:
                    metrics = evaluate_factor(
                        test,
                        normalize(expression),
                        plan.warmup_sessions,
                        len(test.dates),
                        horizon=plan.horizon,
                        cost_bps=plan.cost_bps,
                        stamp_duty_schedule=plan.stamp_duty_schedule,
                        minimum=plan.minimum_symbols,
                        max_exit_delay=plan.max_exit_delay,
                    )
                    status = "ok" if metrics["return"] is not None else "failed"
                except ValueError as error:
                    metrics = {"return": None, "drawdown": None, "error": str(error)[:200]}
                    status = "failed"
                row = {
                    "arm": arm,
                    "path": "market_panel",
                    "episode": number,
                    "phase": ep["phase"],
                    "status": status,
                    "candidate": candidate,
                    "expression": expression,
                    "return": metrics["return"],
                    "drawdown": metrics["drawdown"],
                    "holdout_start": str(panel.dates[ep["test_start"]]),
                    "holdout_end": str(panel.dates[ep["test_stop"] - 1]),
                    "metrics": metrics,
                }
                rows.append(row)
                reveals[arm] = row
            ledger.reveal(batch, reveals)
            batches.append({"batch": batch, "winners": winners, "commitment": commitment})
            for arm in plan.arms:
                for record in ledger.trials(batch, arm):
                    memories[arm].append(
                        {
                            "expression": record["payload"]["proposal"].get(
                                "expression", "<tree serialization failed>"
                            ),
                            "status": record["status"],
                            "score": record["score"],
                        }
                    )
            print(f"Factor round {len(batches)}/{len(schedule)}: {ep['phase']}", flush=True)
    except (ValueError, OSError) as error:
        (output / "incomplete.json").write_text(
            canonical(
                {
                    "status": "incomplete",
                    "error_type": type(error).__name__,
                    "completed_batches": len(batches),
                    "llm_usage": client.usage() if client else None,
                }
            )
        )
        raise
    primary = [row for row in rows if row["phase"] == "confirmation"]
    exploration = [row for row in rows if row["phase"] == "exploration"]
    return {
        "study_id": digest(registration)[:20],
        "registration": registration,
        "results": rows,
        "trials": trials,
        "batches": batches,
        "gate": factor_gate(primary, plan),
        "exploration_gate": factor_gate(exploration, plan),
        "ledger_head": ledger.verify(),
        "llm_usage": client.usage() if plan.llm and client else None,
        "limitations": [
            (
                "Limit-blocked exits "
                "are retained via delayed exits; unresolved exits use last observed close."
            ),
            (
                "Unit long and unit short imply gross exposure 2; "
                "short availability, borrow fees and margin are not modeled."
            ),
            "Drawdown uses open-marked equity, not intraday extrema.",
            "Each block resets NAV to 1; mean block returns are not a continuous portfolio.",
            f"Random/LLM share depth {plan.max_depth}, nodes {plan.max_nodes}, "
            f"lookback {plan.max_lookback}; outer rank depth exempt.",
            "Confirmation sample is short; non-overlap does not prove independent market blocks.",
            (
                "Model publication cutoff reduces historical "
                "look-ahead, not knowledge of published factor literature."
            ),
            (
                "Only price/volume inputs; no point-in-time "
                "fundamentals; test tail signals without in-block exits are omitted."
            ),
        ],
    }
