"""Ranking metrics that stay meaningful when fraud is rare."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from fraudlab.costs import CostConfig, decision_cost


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(scores, dtype=float)
    if y.min() == y.max():
        raise ValueError("Average precision needs both classes")
    return float(average_precision_score(y, score))


def recall_at_precision(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_precision: float,
) -> float | None:
    """Largest recall whose precision is still at least ``target_precision``.

    The final precision-recall point that sklearn appends (precision 1, recall 0,
    no threshold) is ignored. Flagging nobody is not credit for a precision target.
    """
    if not 0 < target_precision <= 1:
        raise ValueError("target_precision must be in (0, 1]")
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(scores, dtype=float)
    precision, recall, _thresholds = precision_recall_curve(y, score)
    if len(precision) < 2:
        return None
    precision = precision[:-1]
    recall = recall[:-1]
    ok = precision >= target_precision
    if not np.any(ok):
        return None
    return float(np.max(recall[ok]))


def bootstrap_test_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    cost: CostConfig,
    *,
    n_boot: int = 400,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, list[float | None] | int]:
    """Percentile intervals at a frozen threshold. The cutoff is not re-tuned."""
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(scores, dtype=float)
    amount = np.asarray(amounts, dtype=float)
    rng = np.random.default_rng(seed)
    ap_draws: list[float] = []
    cost_draws: list[float] = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if y_b.min() != y_b.max():
            ap_draws.append(float(average_precision_score(y_b, score[idx])))
        cost_draws.append(decision_cost(y_b, score[idx], amount[idx], threshold, cost).total_cost)
    return {
        "average_precision_ci95": _percentile_interval(ap_draws, alpha),
        "cost_ci95": _percentile_interval(cost_draws, alpha),
        "n_boot": int(n_boot),
        "n_boot_with_both_classes": int(len(ap_draws)),
    }


def _percentile_interval(draws: list[float], alpha: float) -> list[float | None]:
    if not draws:
        return [None, None]
    lo, hi = np.quantile(np.asarray(draws, dtype=float), [alpha / 2, 1 - alpha / 2])
    return [float(lo), float(hi)]


def choose_model(rows: list[dict[str, float | str]]) -> str:
    """Minimum validation cost, then higher validation PR-AUC, then key."""
    if not rows:
        raise ValueError("No models to choose from")

    def sort_key(row: dict[str, float | str]) -> tuple[float, float, str]:
        return (float(row["val_cost"]), -float(row["val_average_precision"]), str(row["key"]))

    return str(sorted(rows, key=sort_key)[0]["key"])
