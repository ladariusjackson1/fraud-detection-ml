"""Cached queue grid for the Streamlit desk.

The public extract is 284,807 rows. The app does not load it and does not
download it. Training writes ``artifacts/desk_curve.npz``: 1,001 queue sizes,
with the counts needed to reprice the queue when the dollar assumptions
change. Individual tickets come from the demo sample and are scored by the
saved model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fraudlab.costs import (
    CostConfig,
    DecisionCost,
    _total_cost,
    decision_cost,
    decision_from_counts,
)

SLIDER_MAX = 1000
MAX_MODEL_BYTES = 8 * 1024 * 1024
MAX_CURVE_BYTES = 1 * 1024 * 1024
MAX_DEMO_BYTES = 2 * 1024 * 1024
MAX_DEMO_ROWS = 200
CURVE_NAME = "desk_curve.npz"


@dataclass(frozen=True)
class DeskCurve:
    reviews_per_10k: np.ndarray
    threshold: np.ndarray
    true_positives: np.ndarray
    false_positives: np.ndarray
    true_negatives: np.ndarray
    false_negatives: np.ndarray
    false_negative_amount: np.ndarray
    n: int
    saved_threshold: float
    saved_true_positives: int
    saved_false_positives: int
    saved_true_negatives: int
    saved_false_negatives: int
    saved_false_negative_amount: float


def startup_note(*, validation_rows: int, demo_rows: int, grid_points: int) -> str:
    return (
        f"Cached for a free-tier start: {grid_points:,} queue sizes scored on the "
        f"{validation_rows:,}-row validation book, plus {demo_rows:,} sample tickets and the saved model. "
        "The public file is not in this app and is not downloaded."
    )


def assert_cloud_budget(root: Path) -> None:
    """Refuse a cache that would crowd a 1 GB Streamlit Community Cloud instance."""
    problems: list[str] = []
    checked = {
        root / "artifacts" / CURVE_NAME: MAX_CURVE_BYTES,
        root / "artifacts" / "demo_transactions.csv": MAX_DEMO_BYTES,
        root / "artifacts" / "selected_model.joblib": MAX_MODEL_BYTES,
    }
    for path, limit in checked.items():
        if path.exists() and path.stat().st_size > limit:
            problems.append(f"{path.name} is {path.stat().st_size:,} bytes (limit {limit:,})")
    if problems:
        raise RuntimeError(
            "Refusing to start the desk. The cache must stay small enough for a free-tier host. "
            + " ".join(problems)
        )


def build_queue_grid(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    *,
    slider_max: int = SLIDER_MAX,
) -> dict[str, np.ndarray]:
    """Counts at each integer queue size from 0 through ``slider_max`` reviews per 10,000.

    The counts do not depend on the dollar assumptions. The desk recomputes
    dollars from these counts when the inputs change.
    """
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(scores, dtype=float)
    amount = np.asarray(amounts, dtype=float)
    if y.shape != score.shape or y.shape != amount.shape or y.ndim != 1:
        raise ValueError("y, scores, and amounts must be 1-D and the same length")
    if len(y) == 0:
        raise ValueError("Need at least one transaction")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1")
    if not np.isfinite(score).all() or not np.isfinite(amount).all():
        raise ValueError("Scores and amounts must be finite")
    if np.any(amount < -1e-8):
        raise ValueError("Amounts must be >= 0")
    if slider_max < 1:
        raise ValueError("slider_max must be at least 1")

    reviews = np.arange(0, slider_max + 1, dtype=np.int16)
    rates = reviews.astype(np.float64) / 10_000.0
    thresholds = np.full(len(reviews), np.inf, dtype=np.float64)
    positive = (rates > 0) & (rates < 1)
    if np.any(positive):
        thresholds[positive] = np.quantile(score, 1.0 - rates[positive])
    if np.any(rates >= 1):
        thresholds[rates >= 1] = float(np.min(score))

    order = np.argsort(score, kind="mergesort")
    score_sorted = score[order]
    y_sorted = y[order]
    fraud_amount_sorted = np.where(y_sorted == 1, amount[order], 0.0)
    y_suffix = np.concatenate([np.cumsum(y_sorted[::-1])[::-1], np.zeros(1)])
    amount_suffix = np.concatenate([np.cumsum(fraud_amount_sorted[::-1])[::-1], np.zeros(1)])
    count_suffix = np.arange(len(score), -1, -1)

    index = np.full(len(thresholds), len(score), dtype=int)
    finite = np.isfinite(thresholds)
    if np.any(finite):
        index[finite] = np.searchsorted(score_sorted, thresholds[finite], side="left")

    true_positives = y_suffix[index].astype(np.int32)
    flagged = count_suffix[index].astype(np.int32)
    false_positives = (flagged - true_positives).astype(np.int32)
    fraud_total = int(y.sum())
    false_negatives = (fraud_total - true_positives).astype(np.int32)
    fraud_amount = float(amount[y == 1].sum()) if fraud_total else 0.0
    false_negative_amount = (fraud_amount - amount_suffix[index]).astype(np.float64)
    true_negatives = (len(y) - flagged - false_negatives).astype(np.int32)
    return {
        "reviews_per_10k": reviews,
        "threshold": thresholds,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "false_negative_amount": false_negative_amount,
        "n": np.int32(len(y)),
    }


def save_desk_curve(
    path: Path,
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    *,
    saved_threshold: float,
) -> DeskCurve:
    grid = build_queue_grid(y_true, scores, amounts)
    saved = decision_cost(y_true, scores, amounts, float(saved_threshold), CostConfig())
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        reviews_per_10k=grid["reviews_per_10k"],
        threshold=grid["threshold"],
        true_positives=grid["true_positives"],
        false_positives=grid["false_positives"],
        true_negatives=grid["true_negatives"],
        false_negatives=grid["false_negatives"],
        false_negative_amount=grid["false_negative_amount"],
        n=grid["n"],
        saved_threshold=np.float64(saved_threshold),
        saved_true_positives=np.int32(saved.true_positives),
        saved_false_positives=np.int32(saved.false_positives),
        saved_true_negatives=np.int32(saved.true_negatives),
        saved_false_negatives=np.int32(saved.false_negatives),
        saved_false_negative_amount=np.float64(saved.false_negative_amount),
    )
    return load_desk_curve(path)


def load_desk_curve(path: Path) -> DeskCurve:
    blob = np.load(path)
    return DeskCurve(
        reviews_per_10k=np.asarray(blob["reviews_per_10k"], dtype=int),
        threshold=np.asarray(blob["threshold"], dtype=float),
        true_positives=np.asarray(blob["true_positives"], dtype=int),
        false_positives=np.asarray(blob["false_positives"], dtype=int),
        true_negatives=np.asarray(blob["true_negatives"], dtype=int),
        false_negatives=np.asarray(blob["false_negatives"], dtype=int),
        false_negative_amount=np.asarray(blob["false_negative_amount"], dtype=float),
        n=int(blob["n"]),
        saved_threshold=float(blob["saved_threshold"]),
        saved_true_positives=int(blob["saved_true_positives"]),
        saved_false_positives=int(blob["saved_false_positives"]),
        saved_true_negatives=int(blob["saved_true_negatives"]),
        saved_false_negatives=int(blob["saved_false_negatives"]),
        saved_false_negative_amount=float(blob["saved_false_negative_amount"]),
    )


def _grid_index(curve: DeskCurve, reviews_per_10k: float) -> int:
    if not np.isfinite(reviews_per_10k):
        raise ValueError("reviews_per_10k must be finite")
    last = len(curve.reviews_per_10k) - 1
    return int(np.clip(int(np.rint(reviews_per_10k)), 0, last))


def decision_at(curve: DeskCurve, index: int, cost: CostConfig) -> tuple[float, DecisionCost]:
    threshold = float(curve.threshold[index])
    decision = decision_from_counts(
        true_positives=int(curve.true_positives[index]),
        false_positives=int(curve.false_positives[index]),
        true_negatives=int(curve.true_negatives[index]),
        false_negatives=int(curve.false_negatives[index]),
        false_negative_amount=float(curve.false_negative_amount[index]),
        n=curve.n,
        cost=cost,
    )
    return threshold, decision


def grid_decision(
    curve: DeskCurve,
    reviews_per_10k: float,
    cost: CostConfig,
) -> tuple[float, DecisionCost]:
    return decision_at(curve, _grid_index(curve, reviews_per_10k), cost)


def grid_costs(curve: DeskCurve, cost: CostConfig) -> np.ndarray:
    return np.asarray(
        _total_cost(
            curve.true_positives,
            curve.false_positives,
            curve.false_negatives,
            curve.false_negative_amount,
            cost,
        ),
        dtype=float,
    )


def best_grid_queue(curve: DeskCurve, cost: CostConfig) -> tuple[int, float, DecisionCost]:
    """Cheapest integer queue. Ties break toward the higher score cutoff."""
    totals = grid_costs(curve, cost)
    threshold_for_sort = np.where(np.isfinite(curve.threshold), curve.threshold, np.inf)
    order = np.lexsort((-threshold_for_sort, totals))
    index = int(order[0])
    threshold, decision = decision_at(curve, index, cost)
    return int(curve.reviews_per_10k[index]), threshold, decision


def saved_decision(curve: DeskCurve, cost: CostConfig) -> DecisionCost:
    """Validation counts at the cutoff that was frozen during training."""
    return decision_from_counts(
        true_positives=curve.saved_true_positives,
        false_positives=curve.saved_false_positives,
        true_negatives=curve.saved_true_negatives,
        false_negatives=curve.saved_false_negatives,
        false_negative_amount=curve.saved_false_negative_amount,
        n=curve.n,
        cost=cost,
    )
