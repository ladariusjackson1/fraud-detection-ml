"""Score the held-out demo sample one ticket at a time.

The stream is a replay of ``artifacts/demo_transactions.csv``. It is not a
feed from a processor. Each ticket is scored by the saved model, then
explained by setting one field at a time back to a typical legitimate ticket.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from fraudlab.costs import CostConfig, realized_unit_cost
from fraudlab.dashboard import score_frame
from fraudlab.explain import ScoreExplanation, explain_ticket


@dataclass(frozen=True)
class StreamTicket:
    position: int
    demo_id: str
    amount: float
    score: float
    threshold: float
    flagged: bool
    label: int | None
    explanation: ScoreExplanation

    def at_threshold(self, threshold: float) -> StreamTicket:
        flagged = bool(np.isfinite(threshold) and self.score >= threshold)
        return replace(self, threshold=float(threshold), flagged=flagged)


@dataclass(frozen=True)
class ReplayTally:
    seen: int
    flagged: int
    frauds_caught: int
    frauds_missed: int
    false_alerts: int
    legitimate_approved: int
    realized_cost: float


def score_demo_row(
    demo: pd.DataFrame,
    position: int,
    model: object | None,
    *,
    model_key: str,
    feature_columns: list[str],
    score_kind: str,
    reference: dict[str, float],
    threshold: float,
) -> StreamTicket:
    """Score one demo row. ``position`` is zero-based."""
    if len(demo) == 0:
        raise ValueError("Demo stream is empty")
    if position < 0 or position >= len(demo):
        raise IndexError(f"Position {position} is outside the demo stream of {len(demo)}")
    row = demo.iloc[position]
    missing = [column for column in feature_columns if column not in demo.columns]
    if missing:
        raise ValueError(f"Demo stream is missing {missing}")
    frame = pd.DataFrame([{column: float(row[column]) for column in feature_columns}])
    score = float(score_frame(model, frame, model_key)[0])
    flagged = bool(np.isfinite(threshold) and score >= threshold)
    label = int(row["Class"]) if "Class" in demo.columns and pd.notna(row["Class"]) else None
    explanation = explain_ticket(
        model,
        frame,
        reference,
        model_key=model_key,
        feature_columns=feature_columns,
        score_kind=score_kind,
    )
    demo_id = str(row["demo_id"]) if "demo_id" in demo.columns else f"row-{position + 1}"
    return StreamTicket(
        position=int(position),
        demo_id=demo_id,
        amount=float(row["Amount"]),
        score=score,
        threshold=float(threshold),
        flagged=flagged,
        label=label,
        explanation=explanation,
    )


def replay_tally(rows: list[dict], cost: CostConfig) -> ReplayTally:
    """Totals for tickets already shown. Each row needs flagged, label, and amount."""
    if not rows:
        return ReplayTally(0, 0, 0, 0, 0, 0, 0.0)
    seen = 0
    flagged_n = 0
    caught = 0
    missed = 0
    false_alerts = 0
    approved_legit = 0
    realized = 0.0
    for row in rows:
        label = row.get("label")
        if label is None:
            raise ValueError("Replay tally needs a held-out label on every seen ticket")
        label = int(label)
        flagged = bool(row["flagged"])
        amount = float(row["amount"])
        seen += 1
        flagged_n += int(flagged)
        if label == 1 and flagged:
            caught += 1
        elif label == 1 and not flagged:
            missed += 1
        elif label == 0 and flagged:
            false_alerts += 1
        elif label == 0 and not flagged:
            approved_legit += 1
        else:
            raise ValueError("Label must be 0 or 1")
        realized += realized_unit_cost(label, flagged, amount, cost)
    return ReplayTally(
        seen=seen,
        flagged=flagged_n,
        frauds_caught=caught,
        frauds_missed=missed,
        false_alerts=false_alerts,
        legitimate_approved=approved_legit,
        realized_cost=float(realized),
    )
