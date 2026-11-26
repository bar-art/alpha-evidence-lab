import pytest

from alpha_evidence_lab.inference import paired_gate
from alpha_evidence_lab.models import ARMS


def sample_rows(n=6):
    return [
        {
            "arm": arm,
            "path": f"p{i}",
            "episode": ep,
            "status": "ok",
            "return": 0.001 * ep + (i * 0.002 + 0.02 if arm == "memory" else 0.0),
            "drawdown": 0.02,
        }
        for i in range(n)
        for ep in range(2)
        for arm in ARMS
    ]


def gate(rows):
    return paired_gate(rows, ARMS, 499, 42, 0.05, 0.25)


def test_cluster_unit_is_market_path_not_episode_or_search_seed():
    result = gate(sample_rows())
    assert result["clusters"] == 6
    assert result["paired_cells_per_arm"] == 6
    assert result == gate(list(reversed(sample_rows())))
    assert result["comparisons"][1]["mean_net_return_delta"] > 0
    for comparison in result["comparisons"]:
        assert comparison["p_corrected"] >= comparison["p_raw"]


@pytest.mark.parametrize("mode", ["missing", "duplicate", "failed", "nan"])
def test_missing_coverage_is_not_silently_discarded(mode):
    rows = sample_rows()
    if mode == "missing":
        rows.pop()
    elif mode == "duplicate":
        rows.append(dict(rows[-1]))
    elif mode == "failed":
        rows[-1]["status"] = "failed"
    else:
        rows[-1]["drawdown"] = float("nan")
    assert gate(rows)["outcome"] == "insufficient_evidence"


def test_small_samples_and_protected_metric():
    assert gate(sample_rows(2))["outcome"] == "insufficient_evidence"
    rows = sample_rows()
    rows[-1]["drawdown"] = 0.8
    result = gate(rows)
    assert result["comparisons"][1]["outcome"] == "not_promoted"
    assert result["comparisons"][0]["outcome"] == "insufficient_evidence"


def test_unregistered_comparison_refused():
    rows = sample_rows()
    rows[-1]["arm"] = "new_after_seeing_results"
    with pytest.raises(ValueError):
        gate(rows)


def test_exact_resolution_and_zeros():
    from alpha_evidence_lab.inference import exact_sign_flip

    assert exact_sign_flip([1.0] * 5)["p_value"] == 1 / 32
    assert 2 * exact_sign_flip([1.0] * 5)["p_value"] == 0.0625
    assert exact_sign_flip([1.0] * 6)["p_value"] == 1 / 64
    assert exact_sign_flip([1.0] * 5 + [0.0])["p_value"] == 1 / 32
    assert exact_sign_flip([0.0] * 6)["p_value"] == 1.0
    assert gate(sample_rows(5))["outcome"] == "insufficient_evidence"


def test_time_clusters_do_not_count_correlated_stocks_as_independent():
    rows = sample_rows(20)
    for row in rows:
        row["holdout_start"] = f"2024-0{row['episode'] + 1}-01"
        row["holdout_end"] = f"2024-0{row['episode'] + 1}-20"
    result = paired_gate(rows, ARMS, 999, 42, 0.05, 0.25, cluster_unit="time_blocks")
    assert result["clusters"] == 1 and result["outcome"] == "insufficient_evidence"


def test_warmup_is_excluded_before_results_are_seen():
    rows = sample_rows()
    reference = gate(rows)
    for row in rows:
        if row["episode"] == 0:
            row["return"] = 999 if row["arm"] == "memory" else -999
    changed = gate(rows)
    assert changed["comparisons"] == reference["comparisons"]
    assert changed["excluded_cells_per_arm"] == 6


def test_exact_null_false_positive_control():
    import numpy as np

    from alpha_evidence_lab.inference import exact_sign_flip

    rng = np.random.default_rng(117)
    rates = []
    for n in [5, 6, 10]:
        rejected = 0
        for _ in range(1000):
            base, first, second = rng.normal(size=(3, n))
            rejected += (
                any(
                    2 * exact_sign_flip((candidate - base).tolist())["p_value"] < 0.05
                    for candidate in [first, second]
                )
                if n >= 6
                else 0
            )
        rates.append(rejected / 1000)
    assert max(rates) < 0.07  # 固定种子的模拟容差；不是对任意非对称零均值分布的保证。


def test_monte_carlo_agrees_with_exact_and_is_valid():
    import numpy as np

    from alpha_evidence_lab.inference import (
        MC_DRAWS,
        exact_sign_flip,
        monte_carlo_sign_flip,
        sign_flip,
    )

    rng = np.random.default_rng(5)
    for _ in range(5):
        values = (rng.normal(size=14) + 0.3).tolist()
        exact = exact_sign_flip(values)["p_value"]
        approx = monte_carlo_sign_flip(values, seed=1)
        assert abs(approx["p_value"] - exact) < 0.01
        assert approx == monte_carlo_sign_flip(values, seed=1)  # 固定种子可复现
    many = [1.0] * 30
    result = sign_flip(many, seed=3)
    assert result["status"] == "monte_carlo"
    assert result["p_value"] == 1 / (1 + MC_DRAWS)  # 加一修正：p 不会为 0
    assert sign_flip([1.0] * 6, seed=3)["status"] == "exact"
    assert sign_flip([-1.0] * 30, seed=3)["p_value"] == 1.0


def test_monte_carlo_null_false_positive_control():
    import numpy as np

    from alpha_evidence_lab.inference import monte_carlo_sign_flip

    rng = np.random.default_rng(118)
    rejected = 0
    trials = 300
    for trial in range(trials):
        diff = rng.normal(size=25)
        rejected += monte_carlo_sign_flip(diff.tolist(), seed=trial, draws=1999)["p_value"] < 0.05
    assert rejected / trials < 0.08  # 固定种子的模拟容差


def test_gate_evaluates_long_studies_beyond_exact_limit():
    rows = sample_rows(30)
    result = gate(rows)
    assert result["clusters"] == 30 and result["outcome"] == "evaluated"
    memory = result["comparisons"][1]
    assert memory["exact_test"]["status"] == "monte_carlo"
    assert memory["outcome"] == "research_threshold_passed"
    assert result == gate(list(reversed(rows)))


def test_baseline_keyword_preserves_moving_average_output():
    rows = sample_rows()
    old = paired_gate(rows, ARMS, 499, 42, 0.05, 0.25)
    explicit = paired_gate(rows, ARMS, 499, 42, 0.05, 0.25, baseline="ordered")
    assert old == explicit
    renamed = [{**r, "arm": "library" if r["arm"] == "ordered" else r["arm"]} for r in rows]
    new = paired_gate(
        renamed, ("library", "random", "memory"), 499, 42, 0.05, 0.25, baseline="library"
    )
    assert new == old
