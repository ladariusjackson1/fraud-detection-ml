"""Daily alert budget from the cached validation queue.

The public extract has no timestamp. The source study covers two days, so a
planning day is the modeling table divided by two. Every count below is a
validation rate multiplied by that volume. It is not a second measurement of
the test set, and it does not retune the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraudlab.app_cache import SLIDER_MAX, DeskCurve, grid_costs, grid_decision, saved_decision
from fraudlab.costs import CostConfig
from fraudlab.reporting import money

PUBLIC_WINDOW_DAYS = 2


@dataclass(frozen=True)
class AlertBudget:
    alerts_requested: int
    transactions_per_day: float
    reviews_per_10k_requested: float
    reviews_per_10k_grid: int
    threshold: float
    clamped: bool
    validation_rows: int
    validation_frauds: int
    validation_caught: int
    validation_missed: int
    validation_false_alerts: int
    validation_cost: float
    validation_precision: float
    validation_recall: float
    expected_frauds: float
    expected_caught: float
    expected_missed: float
    expected_false_alerts: float
    expected_alerts: float
    expected_cost: float
    expected_missed_principal: float
    expected_chargeback_fees: float
    expected_review_spend: float
    expected_false_decline_spend: float
    analyst_hours: float
    basis: str


def planning_daily_volume(modeling_rows: int, days: int = PUBLIC_WINDOW_DAYS) -> float:
    """Transactions per day when the only fact available is how many days the file spans."""
    rows = int(modeling_rows)
    window = int(days)
    if rows <= 0:
        raise ValueError("modeling_rows must be positive")
    if window <= 0:
        raise ValueError("days must be positive")
    return rows / window


def reviews_per_10k_for_budget(alerts_per_day: float, transactions_per_day: float) -> float:
    if not np.isfinite(transactions_per_day) or transactions_per_day <= 0:
        raise ValueError("transactions_per_day must be finite and > 0")
    if not np.isfinite(alerts_per_day) or alerts_per_day < 0:
        raise ValueError("alerts_per_day must be finite and >= 0")
    return float(alerts_per_day) / float(transactions_per_day) * 10_000.0


def published_alert_count(curve: DeskCurve, transactions_per_day: float, cost: CostConfig) -> int:
    """Review slots a day at the cutoff that was frozen during training."""
    if not np.isfinite(transactions_per_day) or transactions_per_day <= 0:
        raise ValueError("transactions_per_day must be finite and > 0")
    decision = saved_decision(curve, cost)
    return int(round(decision.flag_rate * transactions_per_day))


def evaluate_budget(
    curve: DeskCurve,
    alerts_per_day: int,
    transactions_per_day: float,
    cost: CostConfig,
    *,
    slider_max: int = SLIDER_MAX,
) -> AlertBudget:
    """Snap a daily review cap to the cached grid and scale the validation rates."""
    if isinstance(alerts_per_day, bool) or not isinstance(alerts_per_day, (int, np.integer)):
        raise ValueError("alerts_per_day must be an integer")
    alerts = int(alerts_per_day)
    if alerts < 0:
        raise ValueError("alerts_per_day must be >= 0")
    if slider_max < 1:
        raise ValueError("slider_max must be at least 1")
    requested = reviews_per_10k_for_budget(alerts, transactions_per_day)
    clamped = requested > slider_max
    grid_point = int(min(round(requested), slider_max))
    threshold, decision = grid_decision(curve, grid_point, cost)
    scale = transactions_per_day / decision.n
    frauds = decision.true_positives + decision.false_negatives
    precision = float(decision.precision) if np.isfinite(decision.precision) else float("nan")
    return AlertBudget(
        alerts_requested=alerts,
        transactions_per_day=float(transactions_per_day),
        reviews_per_10k_requested=float(requested),
        reviews_per_10k_grid=int(grid_point),
        threshold=float(threshold),
        clamped=bool(clamped),
        validation_rows=int(decision.n),
        validation_frauds=int(frauds),
        validation_caught=int(decision.true_positives),
        validation_missed=int(decision.false_negatives),
        validation_false_alerts=int(decision.false_positives),
        validation_cost=float(decision.total_cost),
        validation_precision=precision,
        validation_recall=float(decision.recall),
        expected_frauds=float(frauds * scale),
        expected_caught=float(decision.true_positives * scale),
        expected_missed=float(decision.false_negatives * scale),
        expected_false_alerts=float(decision.false_positives * scale),
        expected_alerts=float((decision.true_positives + decision.false_positives) * scale),
        expected_cost=float(decision.total_cost * scale),
        expected_missed_principal=float(decision.false_negative_amount * scale),
        expected_chargeback_fees=float(decision.false_negatives * cost.chargeback_fee * scale),
        expected_review_spend=float(
            (decision.true_positives + decision.false_positives) * cost.review_cost * scale
        ),
        expected_false_decline_spend=float(
            decision.false_positives * cost.false_decline_cost * scale
        ),
        analyst_hours=float(decision.analyst_hours * scale),
        basis=(
            "Validation rates on the cached queue grid, multiplied by the planning "
            "daily volume. Not a recount of the test set."
        ),
    )


def budget_series(
    curve: DeskCurve,
    transactions_per_day: float,
    cost: CostConfig,
) -> pd.DataFrame:
    """One row per cached queue size, scaled to the planning day."""
    if not np.isfinite(transactions_per_day) or transactions_per_day <= 0:
        raise ValueError("transactions_per_day must be finite and > 0")
    if curve.n <= 0:
        raise ValueError("curve has no rows")
    costs = grid_costs(curve, cost)
    scale = float(transactions_per_day) / float(curve.n)
    true_positives = curve.true_positives.astype(float)
    false_positives = curve.false_positives.astype(float)
    false_negatives = curve.false_negatives.astype(float)
    flagged = true_positives + false_positives
    frauds = true_positives + false_negatives
    precision = np.full(len(flagged), np.nan, dtype=float)
    np.divide(true_positives, flagged, out=precision, where=flagged > 0)
    recall = np.divide(true_positives, frauds, out=np.zeros(len(frauds)), where=frauds > 0)
    return pd.DataFrame(
        {
            "alerts_per_day": flagged * scale,
            "expected_caught": true_positives * scale,
            "expected_missed": false_negatives * scale,
            "expected_false_alerts": false_positives * scale,
            "expected_cost": costs * scale,
            "precision": precision,
            "recall": recall,
            "threshold": curve.threshold.astype(float),
            "reviews_per_10k": curve.reviews_per_10k.astype(float),
        }
    )


def queue_precision_recall(curve: DeskCurve) -> tuple[np.ndarray, np.ndarray]:
    """Precision and recall along the cached queue. The flag-nobody point is omitted."""
    true_positives = curve.true_positives.astype(float)
    false_positives = curve.false_positives.astype(float)
    false_negatives = curve.false_negatives.astype(float)
    flagged = true_positives + false_positives
    keep = flagged > 0
    precision = true_positives[keep] / flagged[keep]
    frauds = true_positives + false_negatives
    recall = np.divide(
        true_positives,
        frauds,
        out=np.zeros_like(true_positives),
        where=frauds > 0,
    )
    return recall[keep], precision


def budget_brief(budget: AlertBudget) -> str:
    """One sentence a README can quote. The figures come from ``evaluate_budget``."""
    return (
        f"Planning volume is {budget.transactions_per_day:,.1f} transactions a day "
        f"({PUBLIC_WINDOW_DAYS} days in the source extract, no timestamps to split). "
        f"At {budget.alerts_requested:,} review slots a day the cached validation grid "
        f"expects {budget.expected_caught:.1f} frauds caught, "
        f"{budget.expected_missed:.1f} missed, and {money(budget.expected_cost)} of loss."
    )
