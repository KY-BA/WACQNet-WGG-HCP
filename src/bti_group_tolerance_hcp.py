"""Two-level group-tolerance calibration for frozen BTI-A4 scores.

The construction is deliberately simple.  It first maps every calibration
background to its finite-sample within-group score quantile, then calibrates
the distribution of those group quantiles with a second finite-sample order
statistic.  Under exchangeability of group-tail thresholds, the second layer
has an exact rank-based new-group success probability k/(m+1).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


def finite_order_quantile(values, probability: float) -> tuple[float, int]:
    ordered = np.sort(np.asarray(values, float))
    if len(ordered) == 0:
        raise ValueError("No scores")
    rank = min(len(ordered), max(1, int(math.ceil((len(ordered) + 1) * probability))))
    return float(ordered[rank - 1]), rank


def group_tail_table(frame: pd.DataFrame, scores: np.ndarray,
                     within_group_coverage: float) -> pd.DataFrame:
    scores = np.asarray(scores, float)
    if len(scores) != len(frame):
        raise ValueError("Score length mismatch")
    groups = frame.background_id.astype(str).to_numpy()
    rows = []
    for group in sorted(set(groups)):
        values = scores[groups == group]
        threshold, rank = finite_order_quantile(values, within_group_coverage)
        rows.append({
            "background_id": group,
            "n_scores": int(len(values)),
            "within_group_rank": int(rank),
            "tail_threshold": threshold,
        })
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class GroupToleranceResult:
    qhat: float
    n_groups: int
    group_rank: int
    guaranteed_group_success_probability: float
    within_group_coverage: float
    target_group_success_probability: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "qhat": self.qhat,
            "n_groups": self.n_groups,
            "group_rank": self.group_rank,
            "guaranteed_group_success_probability": self.guaranteed_group_success_probability,
            "within_group_coverage": self.within_group_coverage,
            "target_group_success_probability": self.target_group_success_probability,
        }


def group_tolerance_qhat(frame: pd.DataFrame, scores: np.ndarray, *,
                         within_group_coverage: float,
                         target_group_success_probability: float) -> tuple[GroupToleranceResult, pd.DataFrame]:
    tails = group_tail_table(frame, scores, within_group_coverage)
    m = len(tails)
    group_rank = min(
        m,
        max(1, int(math.ceil((m + 1) * target_group_success_probability))),
    )
    qhat = float(np.sort(tails.tail_threshold.to_numpy(float))[group_rank - 1])
    result = GroupToleranceResult(
        qhat=qhat,
        n_groups=m,
        group_rank=group_rank,
        guaranteed_group_success_probability=float(group_rank / (m + 1.0)),
        within_group_coverage=float(within_group_coverage),
        target_group_success_probability=float(target_group_success_probability),
    )
    return result, tails


def guarantee_scope() -> dict[str, object]:
    return {
        "guarantee_unit": "new background group tail threshold",
        "assumptions": [
            "calibration and future background-tail thresholds are exchangeable",
            "the same within-background Level-B sampling protocol is used",
            "the score function and A4 normalizer are fixed before calibration",
        ],
        "not_guaranteed": [
            "conditional coverage for every background",
            "coverage under arbitrary domain shift",
            "urban-domain coverage when urban groups are absent",
            "coverage for real instantaneous emissions outside the Level-B protocol",
        ],
    }
