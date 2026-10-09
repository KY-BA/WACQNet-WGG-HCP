import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.bti_group_tolerance_hcp import group_tolerance_qhat

ROOT = Path(__file__).resolve().parents[1]


def _frame(n_groups: int, n_rows: int = 20) -> pd.DataFrame:
    return pd.DataFrame({
        "background_id": np.repeat([f"g{x:02d}" for x in range(n_groups)], n_rows)
    })


def test_worst_guard_outer_lobo_uses_maximum_group_tail():
    frame = _frame(35)
    scores = np.concatenate([np.full(20, group + 1.0) for group in range(35)])
    result, tails = group_tolerance_qhat(
        frame, scores, within_group_coverage=.90,
        target_group_success_probability=.95,
    )
    assert result.group_rank == 35
    assert np.isclose(result.qhat, tails.tail_threshold.max())


def test_worst_guard_outer_lobo_rank_guarantee():
    frame = _frame(35)
    result, _ = group_tolerance_qhat(
        frame, np.ones(len(frame)), within_group_coverage=.90,
        target_group_success_probability=.95,
    )
    assert np.isclose(result.guaranteed_group_success_probability, 35 / 36)


def test_worst_guard_full_development_rank_guarantee():
    frame = _frame(36)
    result, _ = group_tolerance_qhat(
        frame, np.ones(len(frame)), within_group_coverage=.90,
        target_group_success_probability=.95,
    )
    assert result.group_rank == 36
    assert np.isclose(result.guaranteed_group_success_probability, 36 / 37)


def test_worst_guard_protocol_is_fixed_and_test_is_sealed():
    cfg = json.loads(
        (ROOT / "configs/bti_worst_group_guard_hcp_development.json").read_text(encoding="utf-8")
    )
    assert cfg["algorithm"]["target_group_success_probability"] == .95
    assert cfg["algorithm"]["candidate_search"] is False
    assert cfg["algorithm"]["safety_factor_search"] is False
    assert cfg["confirmatory_test_allowed"] is False


def test_worst_guard_gates_are_preregistered():
    cfg = json.loads(
        (ROOT / "configs/bti_worst_group_guard_hcp_development.json").read_text(encoding="utf-8")
    )
    gates = cfg["validation"]
    assert gates["minimum_heldout_groups_picp_at_least_0_90"] == 35
    assert gates["maximum_n_picp_below_0_85"] == 1
    assert gates["maximum_n_picp_below_0_80"] == 0
    assert gates["minimum_worst_background_picp"] == .80
    assert gates["maximum_median_normalized_width_ratio_vs_a4"] == 1.50
