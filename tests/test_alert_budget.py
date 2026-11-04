import numpy as np
import pandas as pd
import pytest

from fraudlab.alert_budget import (
    PUBLIC_WINDOW_DAYS,
    budget_brief,
    budget_series,
    evaluate_budget,
    planning_daily_volume,
    published_alert_count,
    queue_precision_recall,
    reviews_per_10k_for_budget,
)
from fraudlab.app_cache import DeskCurve, save_desk_curve
from fraudlab.costs import CostConfig


def _curve(tmp_path):
    y = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 0], dtype=int)
    scores = np.array([0.1, 0.2, 0.05, 0.4, 0.9, 0.3, 0.15, 0.8, 0.12, 0.11])
    amounts = np.array([10, 20, 5, 15, 100, 30, 8, 80, 12, 11], dtype=float)
    return save_desk_curve(tmp_path / "curve.npz", y, scores, amounts, saved_threshold=0.5)


def test_planning_volume_is_the_book_split_across_the_source_window():
    assert planning_daily_volume(275_663) == pytest.approx(275_663 / PUBLIC_WINDOW_DAYS)
    with pytest.raises(ValueError):
        planning_daily_volume(0)
    with pytest.raises(ValueError):
        planning_daily_volume(10, days=0)


def test_budget_scales_validation_rates_and_the_pieces_sum(tmp_path):
    curve = _curve(tmp_path)
    cost = CostConfig()
    with pytest.raises(ValueError):
        reviews_per_10k_for_budget(1, 0)
    with pytest.raises(ValueError):
        reviews_per_10k_for_budget(-1, 100)
    budget = evaluate_budget(curve, 2, 100.0, cost)
    assert budget.alerts_requested == 2
    pieces = (
        budget.expected_missed_principal
        + budget.expected_chargeback_fees
        + budget.expected_review_spend
        + budget.expected_false_decline_spend
    )
    assert pieces == pytest.approx(budget.expected_cost)
    assert budget.expected_caught + budget.expected_missed == pytest.approx(budget.expected_frauds)
    assert "Not a recount of the test set" in budget.basis
    assert str(budget.alerts_requested) in budget_brief(budget)
    assert budget_series(curve, 100.0, cost)["expected_caught"].iloc[-1] >= 0
    recall, precision = queue_precision_recall(curve)
    assert len(recall) == len(precision)
    assert len(recall) >= 1
    published = published_alert_count(curve, 100.0, cost)
    assert published >= 0


def test_budget_clamps_and_rejects_bad_inputs(tmp_path):
    curve = _curve(tmp_path)
    cost = CostConfig()
    clamped = evaluate_budget(curve, 80, 100.0, cost, slider_max=10)
    assert clamped.clamped
    assert clamped.reviews_per_10k_grid == 10
    with pytest.raises(ValueError):
        evaluate_budget(curve, True, 100.0, cost)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        evaluate_budget(curve, 1.5, 100.0, cost)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        evaluate_budget(curve, -1, 100.0, cost)
    with pytest.raises(ValueError):
        evaluate_budget(curve, 1, 100.0, cost, slider_max=0)
    with pytest.raises(ValueError):
        published_alert_count(curve, 0, cost)
    with pytest.raises(ValueError):
        budget_series(curve, -5, cost)
    empty = DeskCurve(
        reviews_per_10k=np.array([0]),
        threshold=np.array([np.inf]),
        true_positives=np.array([0]),
        false_positives=np.array([0]),
        true_negatives=np.array([0]),
        false_negatives=np.array([0]),
        false_negative_amount=np.array([0.0]),
        n=0,
        saved_threshold=np.inf,
        saved_true_positives=0,
        saved_false_positives=0,
        saved_true_negatives=0,
        saved_false_negatives=0,
        saved_false_negative_amount=0.0,
    )
    with pytest.raises(ValueError):
        budget_series(empty, 10, cost)
    quiet = queue_precision_recall(empty)
    assert len(quiet[0]) == 0
    assert isinstance(budget_series(curve, 100.0, cost), pd.DataFrame)
