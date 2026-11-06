"""Scorecard text written next to the charts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def json_safe(value):
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")


def money(value: float, *, cents: bool = False) -> str:
    """Format a loss.

    Nearest-dollar display uses Python's round-half-even formatting, applied
    to each amount on its own. ``round(a) - round(b)`` can therefore differ by
    $1 from ``round(a - b)``. Savings in this project are ``round(a - b)``.
    """
    if cents:
        sign = "-" if value < 0 else ""
        return f"{sign}${abs(value):,.2f}"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def nearest_dollars(value: float) -> int:
    """Integer dollars using the same rounding as :func:`money`."""
    sign = -1 if value < 0 else 1
    return sign * int(f"{abs(float(value)):.0f}")


def independent_rounding_note(approve_all: float, policy_cost: float) -> str:
    """Say how the displayed savings relate to the displayed costs."""
    exact_savings = float(approve_all) - float(policy_cost)
    rounded_savings = nearest_dollars(exact_savings)
    headline_gap = nearest_dollars(approve_all) - nearest_dollars(policy_cost)
    text = (
        "Each displayed dollar amount is rounded to the nearest dollar on its own. "
        f"Exact approve-all cost is {money(approve_all, cents=True)}. "
        f"Exact policy cost is {money(policy_cost, cents=True)}. "
        f"Exact savings are {money(exact_savings, cents=True)}, "
        f"which rounds to {money(float(rounded_savings))}."
    )
    if headline_gap == rounded_savings:
        return text + " Subtracting those rounded headlines gives the same dollar."
    return (
        text
        + " Subtracting the already-rounded headlines gives "
        + f"{money(approve_all)} − {money(policy_cost)} = {money(float(headline_gap))}."
    )


def savings_column_note(metrics: dict) -> str:
    """Which models' rounded savings disagree with a subtraction of rounded costs."""
    names: list[str] = []
    for row in metrics["models"]:
        exact = float(row["approve_all_test_cost"]) - float(row["test_cost"])
        headline_gap = nearest_dollars(row["approve_all_test_cost"]) - nearest_dollars(
            row["test_cost"]
        )
        if headline_gap != nearest_dollars(exact):
            names.append(str(row["display_name"]))
    if not names:
        return (
            "The savings column is the exact cost difference, rounded to the nearest dollar. "
            "For every model that matches the difference of the rounded costs."
        )
    return (
        "The savings column is the exact cost difference, rounded to the nearest dollar. "
        "It is not the difference of the rounded cost figures. "
        f"Those two procedures differ by $1 for: {'; '.join(names)}."
    )


def approve_all_composition_note(metrics: dict) -> str | None:
    """Principal on every test fraud, distinct from the selected model's misses.

    The amount-only rule's missed principal is the full fraud principal only
    when that rule approves every fraud. Otherwise this returns None.
    """
    amount = next((row for row in metrics["models"] if row["key"] == "amount_rule"), None)
    selected = next(row for row in metrics["models"] if row["key"] == metrics["selected_model"])
    if amount is None:
        return None
    test_frauds = int(metrics["split"]["test"]["frauds"])
    if int(amount["test_false_negatives"]) != test_frauds:
        return None
    fee = float(metrics["cost_config"]["chargeback_fee"])
    principal = float(amount["test_false_negative_amount"])
    fees = test_frauds * fee
    missed = float(selected["test_false_negative_amount"])
    missed_n = int(selected["test_false_negatives"])
    return (
        f"Approve-all pays the principal on all {test_frauds} test frauds, "
        f"{money(principal, cents=True)}, plus {money(fee, cents=True)} on each "
        f"({money(fees, cents=True)}), and no review cost: "
        f"{money(principal + fees, cents=True)}. "
        f"The {money(missed)} line is the principal of the {missed_n} missed frauds only."
    )


def percent(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}%}"


def format_ci(point: float, interval: list[float | None] | None, digits: int = 3) -> str:
    if not interval or interval[0] is None or interval[1] is None:
        return f"{point:.{digits}f}"
    return f"{point:.{digits}f} ({interval[0]:.{digits}f}–{interval[1]:.{digits}f})"


def _rate(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _threshold_text(row: dict) -> str:
    if row.get("threshold") is None:
        return "approve all"
    if row["score_kind"] == "amount_usd":
        return money(float(row["threshold"]), cents=True)
    return f"{float(row['threshold']):.6f}"


def render_results_markdown(metrics: dict) -> str:
    source = metrics["data_source"]
    lines = [
        "# Scorecard",
        "",
        "Generated by `python -m fraudlab`. Do not hand-edit.",
        "",
        f"Data: **{source['chart_tag']}**.",
        "",
        source["detail"],
        "",
        "## Protocol",
        "",
        "1. Verify the OpenML fingerprint or label the run as synthetic.",
        "2. Drop exact duplicate rows after a successful fingerprint.",
        "3. Convert public EUR amounts to USD at the planning rate.",
        "4. Stratified split: 60% train, 20% validation, 20% test.",
        "5. Fit imbalance treatments on the training fold only.",
        "6. Choose each cutoff by minimum validation dollar cost.",
        "7. Choose the model by validation cost, then validation PR-AUC.",
        "8. Report the test set once, at that frozen cutoff.",
        "9. Bootstrap the test PR-AUC and test cost without retuning.",
        "",
        "## Ranking on the test set",
        "",
        "| Model | Strategy | Val PR-AUC | Test PR-AUC (95% bootstrap CI) | Recall at precision ≥ 0.50 | Recall at precision ≥ 0.80 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics["models"]:
        mark = "**" if row["key"] == metrics["selected_model"] else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{mark}{row['display_name']}{mark}",
                    row["strategy"],
                    f"{row['val_average_precision']:.3f}",
                    format_ci(row["test_average_precision"], row["test_average_precision_ci95"]),
                    _rate(row["recall_at_precision_0_50"]),
                    _rate(row["recall_at_precision_0_80"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Operating point (cutoff frozen on validation)",
            "",
            "| Model | Cutoff | Precision | Recall | Flag rate | Reviews / 10k | Test cost | Savings vs approve-all |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in metrics["models"]:
        mark = "**" if row["key"] == metrics["selected_model"] else ""
        precision = "n/a" if row["test_precision"] is None else f"{row['test_precision']:.3f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{mark}{row['display_name']}{mark}",
                    _threshold_text(row),
                    precision,
                    f"{row['test_recall']:.3f}",
                    percent(row["test_flag_rate"]),
                    f"{row['test_reviews_per_10k']:.1f}",
                    money(row["test_cost"]),
                    money(row["savings_vs_approve_all"]),
                ]
            )
            + " |"
        )
    approve = metrics["approve_all_test_cost"]
    lines.extend(
        [
            "",
            f"Approve-all test cost: {money(approve)}.",
            independent_rounding_note(
                approve,
                next(
                    row["test_cost"]
                    for row in metrics["models"]
                    if row["key"] == metrics["selected_model"]
                ),
            ),
            savings_column_note(metrics),
            f"Selected model: `{metrics['selected_model']}`.",
            f"Selection rule: {metrics['selection_rule']}",
            "",
        ]
    )
    if metrics.get("default_cutoff_comparison"):
        lines.extend(["## Default 0.50 cutoff versus the cost cutoff", ""])
        lines.append("| Model | Test cost at 0.50 | Test cost at validation cutoff |")
        lines.append("| --- | ---: | ---: |")
        for row in metrics["default_cutoff_comparison"]:
            lines.append(
                f"| {row['display_name']} | {money(row['test_cost_at_0_5'])} | {money(row['test_cost'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_data_source_note(path: Path, source: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if source["kind"] == "synthetic":
        heading = "SYNTHETIC FALLBACK"
    else:
        heading = "OPENML CREDITCARD"
    text = "\n".join(
        [
            heading,
            "",
            source["chart_tag"],
            "",
            source["detail"],
            "",
            f"Rows in the modeling table: {source['rows']:,}",
            f"Fraud rows in the modeling table: {source['fraud_count']:,}",
            f"Fraud rate: {source['fraud_rate']:.6%}",
            f"Amount unit: {source['amount_unit']}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
