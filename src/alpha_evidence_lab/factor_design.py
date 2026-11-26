"""V3 comparison roles and explicitly approximate, post-hoc power description."""

import math
from statistics import NormalDist
from typing import Any


def power_description(
    blocks: int, reference_sd: float | None, alpha: float = 0.05
) -> dict[str, Any]:
    family = 2  # Planned family includes both llm/library and random/library.
    result: dict[str, Any] = {
        "confirmation_blocks": blocks,
        "planned_family_size": family,
        "minimum_raw_exact_p_if_all_nonzero": 2.0 ** (-blocks) if blocks else None,
        "minimum_corrected_exact_p_if_all_nonzero": min(1.0, family * 2.0 ** (-blocks))
        if blocks
        else None,
        "reference_sd": reference_sd,
        "reference": "post-hoc v2 library confirmation returns; not SD of paired LLM differences",
        "assumptions": "independent jointly sign-symmetric blocks; nonzero differences; "
        "overlapping settlement windows do not establish these assumptions",
        "power_method": "normal approximation at one-sided alpha/2 and 80% power; "
        "not exact finite-sample sign-flip power or a performance forecast",
    }
    if blocks and reference_sd is not None:
        z = NormalDist().inv_cdf(1 - alpha / family) + NormalDist().inv_cdf(0.8)
        result["effect_sensitivity"] = [
            {
                "assumed_arm_correlation": rho,
                "paired_difference_sd_proxy": reference_sd * math.sqrt(2 * (1 - rho)),
                "approx_mean_block_effect_for_80pct_power": z
                * reference_sd
                * math.sqrt(2 * (1 - rho))
                / math.sqrt(blocks),
            }
            for rho in (0.0, 0.5, 0.9)
        ]
    return result
