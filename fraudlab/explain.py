"""Plain-English account of one fraud score.

For each field, the ticket is rescored with that field set back to a typical
legitimate ticket from the training fold. If the score falls, the field was
pushing the score up. If the score rises, the field was pushing it down.
Amount is described in dollars. V1–V28 stay codes. The lines do not have to
add up to the gap, because a tree model can use fields together.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from fraudlab.dashboard import score_frame, use_one_worker
from fraudlab.reporting import money

MAX_LINES = 4
COMPONENT_NOTE = (
    "V1–V28 are principal components from the public extract. "
    "This desk does not know the original card fields, so a component is named only by its code. "
    "The comparison row is the median legitimate ticket in the training fold."
)


@dataclass(frozen=True)
class ScoreDriver:
    feature: str
    value: float
    reference: float
    delta: float
    direction: str


@dataclass(frozen=True)
class ScoreExplanation:
    score: float
    reference_score: float
    drivers: tuple[ScoreDriver, ...]
    sentences: tuple[str, ...]
    summary: str


def explain_ticket(
    model: object | None,
    frame: pd.DataFrame,
    reference: Mapping[str, float],
    *,
    model_key: str,
    feature_columns: list[str],
    score_kind: str,
) -> ScoreExplanation:
    """Explain one row. ``delta`` is score(ticket) minus score(ticket with this field reset)."""
    if len(frame) != 1:
        raise ValueError("Explain one transaction at a time")
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Ticket is missing {missing}")
    absent = [column for column in feature_columns if column not in reference]
    if absent:
        raise ValueError(f"Reference profile is missing {absent}")

    actual = {column: float(frame.iloc[0][column]) for column in feature_columns}
    baseline = {column: float(reference[column]) for column in feature_columns}
    rows = [actual, baseline]
    for column in feature_columns:
        swapped = dict(actual)
        swapped[column] = baseline[column]
        rows.append(swapped)
    batch = pd.DataFrame(rows, columns=feature_columns)
    scores = score_frame(use_one_worker(model), batch, model_key)
    score = float(scores[0])
    reference_score = float(scores[1])
    epsilon = _epsilon(score_kind)
    drivers: list[ScoreDriver] = []
    for offset, column in enumerate(feature_columns):
        delta = score - float(scores[2 + offset])
        if delta > epsilon:
            direction = "up"
        elif delta < -epsilon:
            direction = "down"
        else:
            direction = "flat"
        drivers.append(
            ScoreDriver(
                feature=column,
                value=actual[column],
                reference=baseline[column],
                delta=delta,
                direction=direction,
            )
        )
    drivers.sort(key=lambda driver: (-abs(driver.delta), driver.feature))
    shown = _shown(drivers)
    sentences = tuple(_sentence(driver, score_kind) for driver in shown)
    summary = _summary(score, reference_score, drivers, score_kind)
    return ScoreExplanation(
        score=score,
        reference_score=reference_score,
        drivers=tuple(drivers),
        sentences=sentences,
        summary=summary,
    )


def _epsilon(score_kind: str) -> float:
    if score_kind == "amount_usd":
        return 0.5
    return 0.005


def _score_noun(score_kind: str) -> str:
    if score_kind == "amount_usd":
        return "amount score"
    if score_kind == "anomaly":
        return "anomaly score"
    return "fraud score"


def _fmt_score(value: float, score_kind: str) -> str:
    if score_kind == "amount_usd":
        return money(value, cents=True)
    return f"{value:.3f}"


def _fmt_move(delta: float, score_kind: str) -> str:
    magnitude = abs(delta)
    if score_kind == "amount_usd":
        return money(magnitude, cents=True)
    return f"{magnitude:.3f}"


def _fmt_signed(delta: float, score_kind: str) -> str:
    if score_kind == "amount_usd":
        sign = "+" if delta > 0 else "-" if delta < 0 else ""
        return f"{sign}{money(abs(delta), cents=True)}"
    return f"{delta:+.3f}"


def _shown(drivers: list[ScoreDriver]) -> list[ScoreDriver]:
    ups = [driver for driver in drivers if driver.direction == "up"][:MAX_LINES]
    downs = [driver for driver in drivers if driver.direction == "down"][:MAX_LINES]
    return ups + downs


def _summary(
    score: float,
    reference_score: float,
    drivers: list[ScoreDriver],
    score_kind: str,
) -> str:
    noun = _score_noun(score_kind)
    ups = [driver for driver in drivers if driver.direction == "up"]
    downs = [driver for driver in drivers if driver.direction == "down"]
    opening = (
        f"The {noun} is {_fmt_score(score, score_kind)}. "
        f"A typical legitimate ticket scores {_fmt_score(reference_score, score_kind)}."
    )
    if not ups and not downs:
        return opening + " No single field pushes it up or down."
    parts = [opening]
    if ups:
        parts.append(
            f"The strongest upward push is {ups[0].feature} ({_fmt_signed(ups[0].delta, score_kind)})."
        )
    if downs:
        parts.append(
            "The strongest downward push is "
            f"{downs[0].feature} ({_fmt_signed(downs[0].delta, score_kind)})."
        )
    parts.append(
        "Each line sets one field back to that typical ticket and rescores. "
        "The lines do not have to add up to the gap, because fields can work together."
    )
    return " ".join(parts)


def _sentence(driver: ScoreDriver, score_kind: str) -> str:
    noun = _score_noun(score_kind)
    push = "up" if driver.direction == "up" else "down"
    move = _fmt_move(driver.delta, score_kind)
    if driver.feature == "Amount":
        return _amount_sentence(driver, noun, push, move)
    if driver.feature == "Time":
        return (
            f"The clock on this ticket is {driver.value:,.0f} seconds from the start of the window, "
            f"compared with {driver.reference:,.0f} on a typical legitimate ticket. "
            f"That difference pushed the {noun} {push} by {move}."
        )
    undo = "lower" if driver.direction == "up" else "raise"
    return (
        f"{driver.feature} is {_component(driver.value)}, compared with {_component(driver.reference)} "
        "on a typical legitimate ticket. "
        f"Setting it back to that typical value would {undo} the {noun} by {move}, "
        f"so this value pushed the {noun} {push}."
    )


def _amount_sentence(driver: ScoreDriver, noun: str, push: str, move: str) -> str:
    ticket = money(driver.value, cents=True)
    typical = money(driver.reference, cents=True)
    if abs(driver.value - driver.reference) < 0.005:
        relation = "about the same as"
        adjective = "similar"
    elif driver.value > driver.reference:
        relation = "higher than"
        adjective = "higher"
    else:
        relation = "lower than"
        adjective = "lower"
    return (
        f"The amount is {ticket}, {relation} the {typical} on a typical legitimate ticket. "
        f"That {adjective} amount pushed the {noun} {push} by {move}."
    )


def _component(value: float) -> str:
    return f"{value:.2f}"
