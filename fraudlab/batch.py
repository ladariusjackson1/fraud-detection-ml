"""Score an uploaded CSV with the saved model and return a downloadable table.

The desk caps the batch so a free-tier host does not score a full extract in
the request. Reasons are filled for the first flagged rows only. Rescoring
one field at a time for every row would multiply the model calls by the
width of the ticket.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudlab.costs import CostConfig, realized_unit_cost
from fraudlab.dashboard import score_frame
from fraudlab.explain import explain_ticket

MAX_BATCH_ROWS = 2_000
DEFAULT_EXPLAIN_LIMIT = 12
_GENERATED = ("row_id", "fraud_score", "decision", "top_reason", "realized_cost")


def prepare_batch(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Batch scoring expects a pandas DataFrame")
    if len(frame) == 0:
        raise ValueError("CSV has no rows")
    if len(frame) > MAX_BATCH_ROWS:
        raise ValueError(
            f"CSV has {len(frame):,} rows. This desk scores at most {MAX_BATCH_ROWS:,} at a time."
        )
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError("CSV is missing columns: " + ", ".join(missing))
    cleaned = frame.drop(columns=[column for column in _GENERATED if column in frame.columns])
    for column in feature_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    if cleaned[feature_columns].isna().any().any():
        raise ValueError("Feature columns must be numeric")
    if (cleaned["Amount"] < 0).any():
        raise ValueError("Amount must be >= 0")
    if "Class" in cleaned.columns:
        labels = pd.to_numeric(cleaned["Class"], errors="coerce")
        if labels.isna().any() or ~labels.isin([0, 1]).all():
            raise ValueError("Class, when present, must be 0 or 1")
        cleaned["Class"] = labels.astype(int)
    return cleaned.reset_index(drop=True)


def score_batch(
    frame: pd.DataFrame,
    model: object | None,
    *,
    model_key: str,
    feature_columns: list[str],
    score_kind: str,
    threshold: float,
    reference: dict[str, float] | None,
    cost: CostConfig,
    explain_limit: int = DEFAULT_EXPLAIN_LIMIT,
) -> pd.DataFrame:
    """Add ``fraud_score``, ``decision``, and a short reason on flagged rows."""
    if explain_limit < 0:
        raise ValueError("explain_limit must be >= 0")
    prepared = prepare_batch(frame, feature_columns)
    scores = np.asarray(
        score_frame(model, prepared[feature_columns], model_key),
        dtype=float,
    )
    if len(scores) != len(prepared):
        raise RuntimeError("Scorer returned a different number of rows")
    flagged = np.isfinite(threshold) & (scores >= float(threshold))
    reasons = [""] * len(prepared)
    if reference is not None and explain_limit > 0:
        explained = 0
        for index in np.flatnonzero(flagged):
            if explained >= explain_limit:
                break
            ticket = prepared.iloc[[int(index)]][feature_columns]
            explanation = explain_ticket(
                model,
                ticket,
                reference,
                model_key=model_key,
                feature_columns=feature_columns,
                score_kind=score_kind,
            )
            reasons[int(index)] = explanation.summary
            explained += 1
    result = prepared.copy()
    result.insert(0, "row_id", np.arange(1, len(result) + 1))
    result["fraud_score"] = scores
    result["decision"] = np.where(flagged, "review", "approve")
    result["top_reason"] = reasons
    if "Class" in result.columns:
        result["realized_cost"] = [
            realized_unit_cost(int(label), bool(flag), float(amount), cost)
            for label, flag, amount in zip(result["Class"], flagged, result["Amount"], strict=True)
        ]
    return result


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def template_from_demo(
    demo: pd.DataFrame,
    feature_columns: list[str],
    n: int = 8,
) -> pd.DataFrame:
    """A few feature rows and no label, so the upload looks like a new book."""
    if n < 1:
        raise ValueError("template needs at least one row")
    if len(demo) == 0:
        raise ValueError("Demo sample is empty")
    missing = [column for column in feature_columns if column not in demo.columns]
    if missing:
        raise ValueError(f"Demo sample is missing {missing}")
    take = min(int(n), len(demo))
    return demo.loc[: take - 1, feature_columns].reset_index(drop=True)
