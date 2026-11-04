import numpy as np
import pytest

from fraudlab.costs import (
    CostConfig,
    best_operating_point,
    decision_cost,
    decision_from_counts,
    encode_threshold,
    one_row_cost_table,
    realized_unit_cost,
    threshold_at_flag_rate,
    threshold_curve,
)


def test_hand_computed_costs_pick_the_two_frauds():
    y = np.array([0, 1, 1, 0])
    scores = np.array([0.1, 0.9, 0.8, 0.2])
    amounts = np.array([10.0, 100.0, 50.0, 5.0])
    cost = CostConfig(8, 12, 25)
    threshold, decision = best_operating_point(y, scores, amounts, cost)
    assert threshold == pytest.approx(0.8)
    assert decision.total_cost == pytest.approx(16)
    assert decision.true_positives == 2
    assert decision.false_positives == 0
    approve = decision_cost(y, scores, amounts, float("inf"), cost)
    assert approve.total_cost == pytest.approx(200)
    assert encode_threshold(float("inf")) is None


def test_curve_matches_mask_including_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 90)
    y[0] = 1
    y[1] = 0
    scores = rng.random(90)
    scores[2] = scores[3]
    amounts = rng.uniform(1, 80, 90)
    cost = CostConfig()
    curve = threshold_curve(y, scores, amounts, cost)
    for threshold, total in zip(curve["threshold"], curve["total_cost"], strict=True):
        got = decision_cost(y, scores, amounts, float(threshold), cost).total_cost
        assert got == pytest.approx(total)


def test_tie_breaks_toward_the_higher_threshold():
    y = np.array([1, 0, 0])
    scores = np.array([0.9, 0.5, 0.4])
    amounts = np.array([10.0, 0.0, 0.0])
    threshold, decision = best_operating_point(y, scores, amounts, CostConfig(0, 0, 0))
    assert threshold == pytest.approx(0.9)
    assert decision.total_cost == pytest.approx(0)

    all_free = best_operating_point(
        np.array([1, 1, 0]),
        np.array([0.9, 0.8, 0.1]),
        np.array([0.0, 0.0, 100.0]),
        CostConfig(0, 0, 0),
    )
    assert all_free[0] == float("inf")


def test_expensive_review_approves_everything():
    y = np.array([1, 0])
    scores = np.array([0.9, 0.1])
    amounts = np.array([30.0, 10.0])
    threshold, decision = best_operating_point(y, scores, amounts, CostConfig(10_000, 10_000, 0))
    assert threshold == float("inf")
    assert decision.total_cost == pytest.approx(30)


def test_cost_config_and_vector_guards():
    with pytest.raises(ValueError):
        CostConfig(review_cost=-1)
    with pytest.raises(ValueError):
        decision_cost([0, 1], [0.1], [1.0, 2.0], 0.5, CostConfig())
    with pytest.raises(ValueError):
        decision_cost([0, 1], [0.2, 0.8], [-1.0, 2.0], 0.5, CostConfig())
    with pytest.raises(ValueError):
        decision_cost([0, 1], [0.2, np.nan], [1.0, 2.0], 0.5, CostConfig())
    with pytest.raises(ValueError):
        threshold_curve(np.ones(4), np.linspace(0, 1, 4), np.ones(4), CostConfig())


def test_flag_rate_cutoff_and_unit_costs():
    scores = np.array([0.1, 0.4, 0.7, 0.9])
    assert threshold_at_flag_rate(scores, 0) == float("inf")
    assert threshold_at_flag_rate(scores, 1) == pytest.approx(0.1)
    mid = threshold_at_flag_rate(scores, 0.25)
    assert mid <= 0.9
    cost = CostConfig(8, 12, 25)
    assert realized_unit_cost(1, True, 40, cost) == pytest.approx(8)
    assert realized_unit_cost(0, True, 40, cost) == pytest.approx(20)
    assert realized_unit_cost(1, False, 40, cost) == pytest.approx(65)
    assert realized_unit_cost(0, False, 40, cost) == pytest.approx(0)
    table = one_row_cost_table(40, cost)
    assert table["approve_if_fraud"] == pytest.approx(65)
    assert table["review_if_legitimate"] == pytest.approx(20)
    with pytest.raises(ValueError):
        realized_unit_cost(2, False, 10, cost)
    with pytest.raises(ValueError):
        one_row_cost_table(-5, cost)
    rebuilt = decision_from_counts(
        true_positives=2,
        false_positives=0,
        true_negatives=2,
        false_negatives=0,
        false_negative_amount=0.0,
        n=4,
        cost=cost,
    )
    assert rebuilt.total_cost == pytest.approx(16)
    with pytest.raises(ValueError):
        decision_from_counts(
            true_positives=1,
            false_positives=0,
            true_negatives=0,
            false_negatives=0,
            false_negative_amount=0.0,
            n=4,
            cost=cost,
        )
    with pytest.raises(ValueError):
        decision_from_counts(
            true_positives=0,
            false_positives=0,
            true_negatives=1,
            false_negatives=0,
            false_negative_amount=-1.0,
            n=1,
            cost=cost,
        )
    with pytest.raises(ValueError):
        decision_from_counts(
            true_positives=0,
            false_positives=0,
            true_negatives=0,
            false_negatives=0,
            false_negative_amount=0.0,
            n=0,
            cost=cost,
        )
    with pytest.raises(ValueError):
        decision_from_counts(
            true_positives=-1,
            false_positives=2,
            true_negatives=0,
            false_negatives=0,
            false_negative_amount=0.0,
            n=1,
            cost=cost,
        )
