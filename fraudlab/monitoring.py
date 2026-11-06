"""Reference profile for a later book. This file does not contain a drift event.

OpenML data id 1597 has no Time column, so calendar drift cannot be measured
here. What can be computed is the validation-score reference and a population
stability index against that reference when a new batch of scores arrives.
The random-half PSI is a stability check on one book, not a monitoring alarm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudlab.costs import CostConfig, decision_cost

QUANTILES = (0.5, 0.9, 0.99, 0.999)
QUANTILE_LABELS = {0.5: "p50", 0.9: "p90", 0.99: "p99", 0.999: "p999"}
MIN_PSI_ROWS = 30


def _as_scores(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty 1-D vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def quantile_block(values: np.ndarray) -> dict[str, float]:
    array = _as_scores(values, "values")
    computed = np.quantile(array, QUANTILES)
    return {QUANTILE_LABELS[q]: float(v) for q, v in zip(QUANTILES, computed, strict=True)}


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """PSI of ``current`` against quantile bins of ``reference``.

    Bin edges come from the reference only. A shared sample lands near zero.
    A PSI above 0.25 is a common credit-scorecard review trigger. That cutoff
    is a convention, not a threshold estimated from this file.
    """
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and > 0")
    ref = _as_scores(reference, "reference")
    cur = _as_scores(current, "current")
    if len(ref) < bins:
        raise ValueError("reference is shorter than the number of bins")
    edges = np.quantile(ref, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges.astype(float))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
    ref_share = ref_counts / ref_counts.sum()
    cur_share = cur_counts / cur_counts.sum()
    ref_share = np.clip(ref_share, epsilon, None)
    cur_share = np.clip(cur_share, epsilon, None)
    ref_share = ref_share / ref_share.sum()
    cur_share = cur_share / cur_share.sum()
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def half_split_psi(scores: np.ndarray, *, seed: int = 42, bins: int = 10) -> float:
    """PSI between two random halves of one score vector."""
    array = _as_scores(scores, "scores")
    if len(array) < max(4, bins * 2):
        raise ValueError("Need more scores to split into two PSI halves")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(array))
    mid = len(array) // 2
    return population_stability_index(array[order[:mid]], array[order[mid:]], bins=bins)


def batch_psi(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    minimum_rows: int = MIN_PSI_ROWS,
) -> float | None:
    """PSI for an uploaded batch. Short batches return None instead of a noisy index."""
    current_array = np.asarray(current, dtype=float)
    if len(current_array) < minimum_rows:
        return None
    return population_stability_index(reference, current_array)


def reference_profile(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    cost: CostConfig,
) -> dict:
    """Quantiles and the frozen-cutoff operating point on one labeled book."""
    y = np.asarray(y_true, dtype=int)
    score = _as_scores(scores, "scores")
    amount = _as_scores(amounts, "amounts")
    if y.shape != score.shape or y.shape != amount.shape:
        raise ValueError("y, scores, and amounts must be the same length")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1")
    if int((y == 0).sum()) == 0 or int((y == 1).sum()) == 0:
        raise ValueError("Reference profile needs both classes")
    decision = decision_cost(y, score, amount, threshold, cost)
    precision = None if not np.isfinite(decision.precision) else float(decision.precision)
    return {
        "rows": int(decision.n),
        "frauds": int(decision.true_positives + decision.false_negatives),
        "fraud_rate": float(y.mean()),
        "threshold": None if not np.isfinite(threshold) else float(threshold),
        "flag_rate": float(decision.flag_rate),
        "precision": precision,
        "recall": float(decision.recall),
        "reviews_per_10k": float(decision.reviews_per_10k),
        "score_quantiles": {
            "all": quantile_block(score),
            "legitimate": quantile_block(score[y == 0]),
            "fraud": quantile_block(score[y == 1]),
        },
        "amount_quantiles_usd": {
            "legitimate": quantile_block(amount[y == 0]),
            "fraud": quantile_block(amount[y == 1]),
        },
        "validation_half_psi": half_split_psi(score),
    }


def monitoring_notes(profile: dict, *, has_time: bool) -> list[str]:
    if has_time:
        opening = (
            "This run includes a Time column, so the model saw a clock feature. "
            "That is still one window. Calendar drift was not measured."
        )
    else:
        opening = (
            "This file has no Time column, so calendar drift was not measured. "
            "The figures below are the validation-book reference a later extract would be compared with."
        )
    precision = profile["precision"]
    if precision is None:
        precision_text = "precision is undefined because nothing is flagged"
    else:
        precision_text = f"precision {float(precision):.3f}"
    operating = (
        f"At the frozen cutoff the validation book flags {float(profile['flag_rate']):.3%} of tickets "
        f"({float(profile['reviews_per_10k']):.1f} per 10,000), with {precision_text} "
        f"and recall {float(profile['recall']):.3f}."
    )
    psi = (
        f"A random split of the validation scores has PSI {float(profile['validation_half_psi']):.4f}. "
        "That number checks whether two halves of the same book look alike. "
        "It is not a drift event. Compare a new batch with population_stability_index "
        "against these validation scores. A PSI above 0.25 is a common credit-scorecard "
        "review trigger; that cutoff is a convention, not a result estimated from this file."
    )
    watch = (
        "When a new book arrives, compare four things with this reference before refitting: "
        "the flag rate at the frozen cutoff, the score PSI, the precision of tickets that were "
        "actually reviewed once labels arrive, and the amount median. "
        "A one-day move inside the bootstrap interval on test cost is not a reason to refit."
    )
    return [opening, operating, psi, watch]


def build_monitoring_reference(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    cost: CostConfig,
    *,
    has_time: bool,
) -> dict:
    profile = reference_profile(y_true, scores, amounts, threshold, cost)
    profile["has_time"] = bool(has_time)
    profile["notes"] = monitoring_notes(profile, has_time=bool(has_time))
    return profile


def profile_table(profile: dict) -> pd.DataFrame:
    """Quantile table for the monitoring tab and the model card."""
    rows: list[dict[str, str | float]] = []
    blocks = [
        ("Score, all", profile["score_quantiles"]["all"]),
        ("Score, legitimate", profile["score_quantiles"]["legitimate"]),
        ("Score, fraud", profile["score_quantiles"]["fraud"]),
        ("Amount USD, legitimate", profile["amount_quantiles_usd"]["legitimate"]),
        ("Amount USD, fraud", profile["amount_quantiles_usd"]["fraud"]),
    ]
    for label, block in blocks:
        row: dict[str, str | float] = {"Slice": label}
        for key in ("p50", "p90", "p99", "p999"):
            row[key] = float(block[key])
        rows.append(row)
    return pd.DataFrame(rows)
