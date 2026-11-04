"""Dollar cost of an approve-or-review decision.

Every transaction is either approved automatically or sent to a manual review.
The cost model is a planning assumption for the case study:

* a review costs ``review_cost`` (analyst time) whether or not it is fraud
* a reviewed legitimate transaction also costs ``false_decline_cost``
* an approved fraud costs the transaction amount plus ``chargeback_fee``
* an approved legitimate transaction costs nothing in this loss account

The threshold rule is ``score >= threshold``. A non-finite threshold approves
everything. Among thresholds with the same validation cost, the higher
threshold wins so the queue stays smaller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraudlab.constants import (
    DEFAULT_CHARGEBACK_FEE,
    DEFAULT_FALSE_DECLINE_COST,
    DEFAULT_REVIEW_COST,
    MINUTES_PER_REVIEW,
)


@dataclass(frozen=True)
class CostConfig:
    review_cost: float = DEFAULT_REVIEW_COST
    false_decline_cost: float = DEFAULT_FALSE_DECLINE_COST
    chargeback_fee: float = DEFAULT_CHARGEBACK_FEE
    minutes_per_review: float = MINUTES_PER_REVIEW

    def __post_init__(self) -> None:
        for name in ("review_cost", "false_decline_cost", "chargeback_fee", "minutes_per_review"):
            value = getattr(self, name)
            if value < 0 or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number >= 0")

    def as_dict(self) -> dict[str, float]:
        return {
            "review_cost": float(self.review_cost),
            "false_decline_cost": float(self.false_decline_cost),
            "chargeback_fee": float(self.chargeback_fee),
            "minutes_per_review": float(self.minutes_per_review),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> CostConfig:
        return cls(
            review_cost=float(payload["review_cost"]),
            false_decline_cost=float(payload["false_decline_cost"]),
            chargeback_fee=float(payload["chargeback_fee"]),
            minutes_per_review=float(payload.get("minutes_per_review", MINUTES_PER_REVIEW)),
        )


@dataclass(frozen=True)
class DecisionCost:
    total_cost: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    false_negative_amount: float
    precision: float
    recall: float
    flag_rate: float
    reviews_per_10k: float
    analyst_hours: float
    n: int

    def as_dict(self) -> dict[str, float | int | None]:
        precision = None if not np.isfinite(self.precision) else float(self.precision)
        return {
            "total_cost": float(self.total_cost),
            "true_positives": int(self.true_positives),
            "false_positives": int(self.false_positives),
            "true_negatives": int(self.true_negatives),
            "false_negatives": int(self.false_negatives),
            "false_negative_amount": float(self.false_negative_amount),
            "precision": precision,
            "recall": float(self.recall),
            "flag_rate": float(self.flag_rate),
            "reviews_per_10k": float(self.reviews_per_10k),
            "analyst_hours": float(self.analyst_hours),
            "n": int(self.n),
        }


def encode_threshold(value: float) -> float | None:
    if not np.isfinite(value):
        return None
    return float(value)


def decode_threshold(value: float | None) -> float:
    if value is None:
        return float("inf")
    threshold = float(value)
    if np.isnan(threshold):
        raise ValueError("Threshold cannot be NaN")
    return threshold


def _as_vectors(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(scores, dtype=float)
    amount = np.asarray(amounts, dtype=float)
    if y.shape != score.shape or y.shape != amount.shape or y.ndim != 1:
        raise ValueError("y, scores, and amounts must be 1-D and the same length")
    if len(y) == 0:
        raise ValueError("Need at least one transaction")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1")
    if not np.isfinite(score).all():
        raise ValueError("Scores must be finite")
    if not np.isfinite(amount).all():
        raise ValueError("Amounts must be finite")
    if np.any(amount < -1e-8):
        raise ValueError("Amounts must be >= 0")
    return y, score, amount


def _total_cost(
    true_positives: np.ndarray | float,
    false_positives: np.ndarray | float,
    false_negatives: np.ndarray | float,
    false_negative_amount: np.ndarray | float,
    cost: CostConfig,
) -> np.ndarray | float:
    false_positive_unit = cost.review_cost + cost.false_decline_cost
    return (
        false_positives * false_positive_unit
        + true_positives * cost.review_cost
        + false_negative_amount
        + false_negatives * cost.chargeback_fee
    )


def decision_cost(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    cost: CostConfig,
) -> DecisionCost:
    y, score, amount = _as_vectors(y_true, scores, amounts)
    if np.isnan(threshold):
        raise ValueError("Threshold cannot be NaN")
    if np.isfinite(threshold):
        flagged = score >= threshold
    else:
        flagged = np.zeros(len(y), dtype=bool)
    positive = y == 1
    true_positives = int(np.sum(flagged & positive))
    false_positives = int(np.sum(flagged & ~positive))
    false_negatives_mask = ~flagged & positive
    false_negatives = int(np.sum(false_negatives_mask))
    true_negatives = int(np.sum(~flagged & ~positive))
    false_negative_amount = float(amount[false_negatives_mask].sum())
    total = float(
        _total_cost(true_positives, false_positives, false_negatives, false_negative_amount, cost)
    )
    flagged_n = true_positives + false_positives
    precision = float("nan") if flagged_n == 0 else true_positives / flagged_n
    fraud_n = true_positives + false_negatives
    recall = 0.0 if fraud_n == 0 else true_positives / fraud_n
    flag_rate = flagged_n / len(y)
    return DecisionCost(
        total_cost=total,
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        false_negative_amount=false_negative_amount,
        precision=float(precision),
        recall=float(recall),
        flag_rate=float(flag_rate),
        reviews_per_10k=float(flag_rate * 10_000),
        analyst_hours=float(flagged_n * cost.minutes_per_review / 60.0),
        n=int(len(y)),
    )


def threshold_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    cost: CostConfig,
) -> pd.DataFrame:
    """One row per distinct score, plus a flag-nothing row at +inf."""
    y, score, amount = _as_vectors(y_true, scores, amounts)
    fraud_total = int(y.sum())
    if fraud_total == 0 or fraud_total == len(y):
        raise ValueError("Threshold curve needs both classes")
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    amount_sorted = amount[order]
    score_sorted = score[order]
    tp_cum = np.cumsum(y_sorted)
    caught_amount = np.cumsum(np.where(y_sorted == 1, amount_sorted, 0.0))
    flagged_cum = np.arange(1, len(y_sorted) + 1)
    boundary = np.empty(len(score_sorted), dtype=bool)
    boundary[-1] = True
    boundary[:-1] = score_sorted[:-1] != score_sorted[1:]
    idx = np.flatnonzero(boundary)
    true_positives = tp_cum[idx]
    flagged = flagged_cum[idx]
    false_positives = flagged - true_positives
    false_negatives = fraud_total - true_positives
    fraud_amount = float(amount_sorted[y_sorted == 1].sum())
    false_negative_amount = fraud_amount - caught_amount[idx]
    thresholds = score_sorted[idx]
    return _curve_frame(
        thresholds=np.append(thresholds, np.inf),
        true_positives=np.append(true_positives, 0),
        false_positives=np.append(false_positives, 0),
        false_negatives=np.append(false_negatives, fraud_total),
        false_negative_amount=np.append(false_negative_amount, fraud_amount),
        n=len(y),
        cost=cost,
    )


def _curve_frame(
    *,
    thresholds: np.ndarray,
    true_positives: np.ndarray,
    false_positives: np.ndarray,
    false_negatives: np.ndarray,
    false_negative_amount: np.ndarray,
    n: int,
    cost: CostConfig,
) -> pd.DataFrame:
    true_positives = true_positives.astype(float)
    false_positives = false_positives.astype(float)
    false_negatives = false_negatives.astype(float)
    false_negative_amount = false_negative_amount.astype(float)
    total = np.asarray(
        _total_cost(true_positives, false_positives, false_negatives, false_negative_amount, cost),
        dtype=float,
    )
    flagged = true_positives + false_positives
    precision = np.full(len(total), np.nan, dtype=float)
    np.divide(true_positives, flagged, out=precision, where=flagged > 0)
    frauds = true_positives + false_negatives
    recall = np.divide(true_positives, frauds, out=np.zeros(len(total)), where=frauds > 0)
    flag_rate = flagged / n
    true_negatives = n - flagged - false_negatives
    return pd.DataFrame(
        {
            "threshold": thresholds.astype(float),
            "total_cost": total,
            "true_positives": true_positives.astype(int),
            "false_positives": false_positives.astype(int),
            "true_negatives": true_negatives.astype(int),
            "false_negatives": false_negatives.astype(int),
            "false_negative_amount": false_negative_amount,
            "precision": precision,
            "recall": recall,
            "flag_rate": flag_rate,
            "reviews_per_10k": flag_rate * 10_000,
            "analyst_hours": flagged * cost.minutes_per_review / 60.0,
        }
    )


def best_operating_point(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    cost: CostConfig,
) -> tuple[float, DecisionCost]:
    curve = threshold_curve(y_true, scores, amounts, cost)
    threshold_for_sort = np.where(np.isfinite(curve["threshold"]), curve["threshold"], np.inf)
    order = np.lexsort((-threshold_for_sort, curve["total_cost"].to_numpy()))
    threshold = float(curve["threshold"].iloc[int(order[0])])
    return threshold, decision_cost(y_true, scores, amounts, threshold, cost)


def threshold_at_flag_rate(scores: np.ndarray, flag_rate: float) -> float:
    """Cutoff that flags about ``flag_rate`` of the scored rows (the top of the list)."""
    score = np.asarray(scores, dtype=float)
    if score.ndim != 1 or len(score) == 0 or not np.isfinite(score).all():
        raise ValueError("Scores must be a non-empty finite vector")
    if flag_rate <= 0:
        return float("inf")
    if flag_rate >= 1:
        return float(np.min(score))
    return float(np.quantile(score, 1.0 - flag_rate))


def realized_unit_cost(label: int, flagged: bool, amount: float, cost: CostConfig) -> float:
    if label not in (0, 1):
        raise ValueError("Label must be 0 or 1")
    if amount < 0 or not np.isfinite(amount):
        raise ValueError("Amount must be finite and >= 0")
    if flagged and label == 1:
        return float(cost.review_cost)
    if flagged and label == 0:
        return float(cost.review_cost + cost.false_decline_cost)
    if (not flagged) and label == 1:
        return float(amount + cost.chargeback_fee)
    return 0.0


def one_row_cost_table(amount: float, cost: CostConfig) -> dict[str, float]:
    """What this single ticket costs in each state of the world. The label is unknown."""
    if amount < 0 or not np.isfinite(amount):
        raise ValueError("Amount must be finite and >= 0")
    return {
        "approve_if_legitimate": 0.0,
        "approve_if_fraud": float(amount + cost.chargeback_fee),
        "review_if_legitimate": float(cost.review_cost + cost.false_decline_cost),
        "review_if_fraud": float(cost.review_cost),
    }
