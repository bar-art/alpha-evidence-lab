"""独立簇差值的单侧符号翻转检验；有效性依赖零假设下的符号可交换性。

非零簇不超过 EXACT_MAX 时穷举全部 2^n 种符号；更多时用固定种子的蒙特卡洛符号翻转，
p = (1 + #{T* >= T}) / (1 + B)（Phipson & Smyth 2010），在有限抽样下仍是有效（保守）的 p 值。
"""

import math
from typing import Any

import numpy as np

EXACT_MAX = 20
MC_DRAWS = 99_999
MC_CHUNK = 10_000


def _tolerance(values: np.ndarray) -> float:
    return float(np.finfo(float).eps * max(float(np.abs(values).sum()), 1e-300) * 32)


def monte_carlo_sign_flip(
    differences: list[float], seed: int, draws: int = MC_DRAWS
) -> dict[str, Any]:
    values = np.asarray(differences, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("需要非空有限簇差值")
    if draws < 1:
        raise ValueError("蒙特卡洛抽样次数必须为正")
    nonzero = values[values != 0]
    observed = float(values.sum())
    tolerance = _tolerance(values)
    rng = np.random.default_rng(seed)
    exceed, remaining = 0, draws
    while remaining:
        size = min(MC_CHUNK, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, len(nonzero)))
        exceed += int(np.count_nonzero(signs @ nonzero >= observed - tolerance))
        remaining -= size
    return {
        "status": "monte_carlo",
        "p_value": float((1 + exceed) / (1 + draws)),
        "informative_clusters": len(nonzero),
        "permutations": draws,
        "seed": seed,
        "smallest_attainable_p": 1 / (1 + draws),
    }


def sign_flip(differences: list[float], seed: int, maximum: int = EXACT_MAX) -> dict[str, Any]:
    """非零簇不多于 maximum 时精确穷举，否则蒙特卡洛；两者都是单侧 p 值。"""
    exact = exact_sign_flip(differences, maximum)
    if exact["status"] != "too_many_exact_clusters":
        return exact
    return monte_carlo_sign_flip(differences, seed)


def exact_sign_flip(differences: list[float], maximum: int = EXACT_MAX) -> dict[str, Any]:
    values = np.asarray(differences, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("需要非空有限簇差值")
    # 去掉精确零只压缩重复排列，不删掉非零小差值，也不改变检验p值。
    nonzero = values[values != 0]
    if len(nonzero) > maximum:
        return {
            "status": "too_many_exact_clusters",
            "p_value": None,
            "informative_clusters": len(nonzero),
            "permutations": None,
        }
    sums = np.array([0.0])
    for value in nonzero:
        sums = np.concatenate((sums + value, sums - value))
    observed = float(values.sum())
    tolerance = _tolerance(values)
    p_value = float(np.count_nonzero(sums >= observed - tolerance) / len(sums))
    return {
        "status": "exact",
        "p_value": p_value,
        "informative_clusters": len(nonzero),
        "permutations": len(sums),
        "smallest_attainable_p": 1 / len(sums),
    }


def cluster_assignments(rows: list[dict[str, Any]], unit: str) -> dict[tuple[str, int], str]:
    if unit == "paths":
        return {(row["path"], row["episode"]): str(row["path"]) for row in rows}
    if unit != "time_blocks":
        raise ValueError("未知聚类方案")
    intervals = sorted(
        {(row["holdout_start"], row["holdout_end"], row["path"], row["episode"]) for row in rows}
    )
    result = {}
    end, number = "", -1
    for start, stop, path, episode in intervals:
        if stop < start:
            raise ValueError("测试区间倒置")
        if start > end:
            number += 1
            end = stop
        else:
            end = max(end, stop)
        result[path, episode] = f"time_block_{number}"
    return result


def paired_gate(
    rows: list[dict[str, Any]],
    arms: tuple[str, ...],
    samples: int,
    seed: int,
    alpha: float,
    protected_dd: float,
    minimum_paths: int = 6,
    *,
    cluster_unit: str = "paths",
    warmup_episodes: int = 1,
) -> dict[str, Any]:
    if len(arms) < 2 or arms[0] != "ordered" or len(set(arms)) != len(arms):
        raise ValueError("必须包含唯一的预注册基线和比较方法")
    if not 0 < alpha < 1 or warmup_episodes < 0:
        raise ValueError("显著性或暖机设置非法")
    if any(row["arm"] not in arms for row in rows):
        raise ValueError("比较结果出现未预注册方法")
    coverage = {arm: {(r["path"], r["episode"]) for r in rows if r["arm"] == arm} for arm in arms}
    counts = {arm: sum(r["arm"] == arm for r in rows) for arm in arms}
    keys = coverage[arms[0]]
    if not keys or any(coverage[a] != keys or counts[a] != len(keys) for a in arms):
        return {"outcome": "insufficient_evidence", "reason": "配对单元缺失或重复，禁止静默删行"}
    if any(
        r.get("status") != "ok"
        or not math.isfinite(r["return"])
        or not math.isfinite(r["drawdown"])
        for r in rows
    ):
        return {"outcome": "insufficient_evidence", "reason": "失败或非有限结果，未按零收益填补"}
    primary = [r for r in rows if r["episode"] >= warmup_episodes]
    lookup = {(r["arm"], r["path"], r["episode"]): r for r in primary}
    primary_keys = {(r["path"], r["episode"]) for r in primary}
    assignments = cluster_assignments(primary, cluster_unit)
    labels = sorted(set(assignments.values()))
    family = len(arms) - 1
    required = max(6, minimum_paths)
    while family / 2**required >= alpha:
        required += 1
    output: dict[str, Any] = {
        "method": "cluster_sign_flip_bonferroni_one_sided",
        "test": f"exact_up_to_{EXACT_MAX}_nonzero_clusters_else_monte_carlo_{MC_DRAWS}",
        "cluster_unit": cluster_unit,
        "clusters": len(labels),
        "minimum_clusters": required,
        "family_size": family,
        "paired_cells_per_arm": len(primary_keys),
        "warmup_episodes": warmup_episodes,
        "excluded_cells_per_arm": len(keys) - len(primary_keys),
        "comparisons": [],
        "cluster_members": {
            label: [list(key) for key, value in sorted(assignments.items()) if value == label]
            for label in labels
        },
        "assumption": "独立簇差值在零假设下联合符号对称/可交换；均值相等本身不足以保证有效性",
        "bootstrap_samples_unused": True,
        "monte_carlo_seed": seed,
        "smallest_possible_corrected_p": min(
            1.0, family * max(1 / 2 ** len(labels), 1 / (1 + MC_DRAWS))
        ),
    }
    if len(labels) < required:
        return {**output, "outcome": "insufficient_evidence", "reason": "独立簇或精确p值分辨率不足"}
    for arm in arms[1:]:
        differences = []
        for label in labels:
            cells = sorted(key for key in primary_keys if assignments[key] == label)
            differences.append(
                float(
                    np.mean(
                        [
                            lookup[arm, *key]["return"] - lookup[arms[0], *key]["return"]
                            for key in cells
                        ]
                    )
                )
            )
        exact = sign_flip(differences, seed)
        p_raw = exact["p_value"]
        p_corrected = min(1.0, p_raw * family) if p_raw is not None else None
        observed = float(np.mean(differences))
        max_dd = max(r["drawdown"] for r in rows if r["arm"] == arm)
        resolvable = (
            exact["status"] in {"exact", "monte_carlo"}
            and family * exact["smallest_attainable_p"] < alpha
        )
        passing = (
            resolvable
            and p_corrected is not None
            and observed > 0
            and p_corrected < alpha
            and max_dd <= protected_dd
        )
        output["comparisons"].append(
            {
                "arm": arm,
                "mean_net_return_delta": observed,
                "p_raw": p_raw,
                "p_corrected": p_corrected,
                "exact_test": exact,
                "interval": None,
                "max_episode_drawdown": max_dd,
                "outcome": "research_threshold_passed"
                if passing
                else "insufficient_evidence"
                if not resolvable
                else "not_promoted",
            }
        )
    output.update(outcome="evaluated", reason="p值仅在所述簇独立与符号对称假设下有效；不是实盘授权")
    return output
