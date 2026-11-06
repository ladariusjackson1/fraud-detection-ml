import json
from pathlib import Path

import pytest

from fraudlab.reporting import (
    approve_all_composition_note,
    independent_rounding_note,
    money,
    nearest_dollars,
    savings_column_note,
)

ROOT = Path(__file__).resolve().parents[1]


def _metrics() -> dict:
    return json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))


def test_nearest_dollar_matches_money_and_half_even():
    assert nearest_dollars(10903.354) == 10903
    assert nearest_dollars(-1.5) == -2
    assert money(1.5) == "$2"
    assert money(2.5) == "$2"


def test_public_savings_are_the_exact_difference_then_rounded():
    metrics = _metrics()
    approve = metrics["approve_all_test_cost"]
    for row in metrics["models"]:
        exact = approve - row["test_cost"]
        assert row["savings_vs_approve_all"] == pytest.approx(exact, abs=1e-6)
        assert money(row["savings_vs_approve_all"]) == money(exact)
    forest = next(row for row in metrics["models"] if row["key"] == "random_forest")
    assert money(forest["savings_vs_approve_all"]) == "$10,903"
    headline_gap = nearest_dollars(approve) - nearest_dollars(forest["test_cost"])
    assert headline_gap == 10904
    note = independent_rounding_note(approve, forest["test_cost"])
    assert "$10,903.35" in note
    assert "$10,904" in note
    logistic = next(row for row in metrics["models"] if row["key"] == "logistic_balanced")
    same = independent_rounding_note(approve, logistic["test_cost"])
    assert "same dollar" in same
    gaps = savings_column_note(metrics)
    assert "Random forest" in gaps
    assert "Logistic regression, undersampled" in gaps
    assert "Isolation forest" in gaps
    assert "Gradient boosting" not in gaps


def test_approve_all_principal_is_every_test_fraud_not_the_misses():
    metrics = _metrics()
    note = approve_all_composition_note(metrics)
    assert note is not None
    assert "$14,939.52" in note
    assert "$2,375.00" in note
    assert "$17,314.52" in note
    assert "19 missed frauds only" in note
    amount = next(row for row in metrics["models"] if row["key"] == "amount_rule")
    assert amount["test_false_negative_amount"] + 95 * 25 == pytest.approx(
        metrics["approve_all_test_cost"], abs=1e-6
    )
    quiet = json.loads(json.dumps(metrics))
    for row in quiet["models"]:
        if row["key"] == "amount_rule":
            row["test_false_negatives"] = 1
    assert approve_all_composition_note(quiet) is None
    dropped = json.loads(json.dumps(metrics))
    dropped["models"] = [row for row in dropped["models"] if row["key"] != "amount_rule"]
    assert approve_all_composition_note(dropped) is None
    empty = json.loads(json.dumps(metrics))
    empty["models"] = []
    assert savings_column_note(empty).startswith("The savings column is the exact cost difference")
