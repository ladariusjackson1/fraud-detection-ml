from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from fraudlab.constants import V_COLUMNS
from fraudlab.dashboard import load_desk
from fraudlab.explain import COMPONENT_NOTE, ScoreDriver, _sentence, explain_ticket
from fraudlab.features import DesignMatrix
from fraudlab.models import SPECS, fit_spec

ROOT = Path(__file__).resolve().parents[1]

ROOT_COLUMNS = [*V_COLUMNS, "Amount"]


def _zeros(**overrides) -> dict[str, float]:
    row = {column: 0.0 for column in ROOT_COLUMNS}
    row.update(overrides)
    return row


def _frame(mapping: dict[str, float], columns: list[str] | None = None) -> pd.DataFrame:
    used = ROOT_COLUMNS if columns is None else columns
    return pd.DataFrame([{column: float(mapping[column]) for column in used}])


def test_amount_rule_names_the_direction():
    reference = _zeros(Amount=10)
    higher = explain_ticket(
        None,
        _frame(_zeros(Amount=40)),
        reference,
        model_key="amount_rule",
        feature_columns=ROOT_COLUMNS,
        score_kind="amount_usd",
    )
    amount = next(driver for driver in higher.drivers if driver.feature == "Amount")
    assert amount.direction == "up"
    assert amount.delta == pytest.approx(30)
    assert all(driver.delta == 0 for driver in higher.drivers if driver.feature != "Amount")
    text = " ".join(higher.sentences)
    assert "higher" in text
    assert "pushed the amount score up" in text
    assert "$40.00" in text and "$10.00" in text
    assert "merchant" not in text.lower()
    assert "up" in higher.summary

    lower = explain_ticket(
        None,
        _frame(_zeros(Amount=4)),
        reference,
        model_key="amount_rule",
        feature_columns=ROOT_COLUMNS,
        score_kind="amount_usd",
    )
    assert (
        next(driver.direction for driver in lower.drivers if driver.feature == "Amount") == "down"
    )
    assert "lower" in " ".join(lower.sentences)
    assert "pushed the amount score down" in " ".join(lower.sentences)

    same = explain_ticket(
        None,
        _frame(reference),
        reference,
        model_key="amount_rule",
        feature_columns=ROOT_COLUMNS,
        score_kind="amount_usd",
    )
    assert same.sentences == ()
    assert "No single field" in same.summary


def test_logistic_drivers_match_a_rescore_and_a_small_amount_can_raise_the_score():
    n_legit, n_fraud = 240, 80
    n = n_legit + n_fraud
    frame = pd.DataFrame({column: np.zeros(n) for column in V_COLUMNS})
    frame["Amount"] = np.array([80.0] * n_legit + [5.0] * n_fraud)
    frame["V14"] = np.array([0.0] * n_legit + [-5.0] * n_fraud)
    frame["Class"] = np.array([0] * n_legit + [1] * n_fraud)
    spec = next(item for item in SPECS if item.key == "logistic_balanced")
    fitted = fit_spec(spec, frame, fast=True, seed=0)
    reference = _zeros(Amount=80.0, V14=0.0)
    ticket = _zeros(Amount=5.0, V14=-5.0)
    explained = explain_ticket(
        fitted.model,
        _frame(ticket),
        reference,
        model_key=spec.key,
        feature_columns=ROOT_COLUMNS,
        score_kind="probability",
    )
    by_name = {driver.feature: driver for driver in explained.drivers}
    assert by_name["V14"].direction == "up"
    assert by_name["Amount"].direction == "up"
    amount_sentence = next(
        sentence for sentence in explained.sentences if sentence.startswith("The amount")
    )
    assert "lower" in amount_sentence
    assert "pushed the fraud score up" in amount_sentence
    v14_sentence = next(sentence for sentence in explained.sentences if sentence.startswith("V14"))
    assert "pushed the fraud score up" in v14_sentence
    assert "merchant" not in v14_sentence.lower()
    assert "principal component" in COMPONENT_NOTE
    for feature in ("V14", "Amount", "V3"):
        driver = by_name[feature]
        swapped = dict(ticket)
        swapped[feature] = reference[feature]
        rescored = explain_ticket(
            fitted.model,
            _frame(swapped),
            reference,
            model_key=spec.key,
            feature_columns=ROOT_COLUMNS,
            score_kind="probability",
        )
        assert driver.delta == pytest.approx(explained.score - rescored.score, abs=1e-6)


class _AdditiveScore(BaseEstimator):
    """Score rises with the sum of the design-matrix columns. Deterministic, no fit."""

    def fit(self, X, y=None):
        del y
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = int(np.asarray(X).shape[1])
        return self

    def predict_proba(self, X):
        raw = np.asarray(X, dtype=float).sum(axis=1) / 20.0
        score = 1.0 / (1.0 + np.exp(-raw))
        return np.column_stack([1.0 - score, score])


def test_several_upward_fields_keep_only_the_strongest_lines_and_time_is_plain():
    columns = [*V_COLUMNS, "Amount"]
    train = pd.DataFrame([{column: 0.0 for column in columns}, {column: 1.0 for column in columns}])
    model = Pipeline([("design", DesignMatrix()), ("model", _AdditiveScore())])
    model.fit(train, np.array([0, 1]))
    reference = {column: 0.0 for column in columns}
    ticket = {column: 0.0 for column in columns}
    for index in range(1, 7):
        ticket[f"V{index}"] = 3.0
    ticket["V8"] = -3.0
    explained = explain_ticket(
        model,
        _frame(ticket, columns),
        reference,
        model_key="logistic_balanced",
        feature_columns=columns,
        score_kind="anomaly",
    )
    ups = [driver for driver in explained.drivers if driver.direction == "up"]
    downs = [driver for driver in explained.drivers if driver.direction == "down"]
    assert len(ups) >= 5
    assert downs
    up_sentences = [
        sentence for sentence in explained.sentences if "pushed the anomaly score up" in sentence
    ]
    assert len(up_sentences) == 4
    assert "anomaly score" in explained.summary
    assert "do not have to add up" in explained.summary
    clock = _sentence(ScoreDriver("Time", 50_000.0, 1_000.0, 0.2, "up"), "probability")
    assert "start of the window" in clock
    assert "pushed the fraud score up" in clock
    lower = _sentence(ScoreDriver("V8", -3.0, 0.0, -0.2, "down"), "probability")
    assert "would raise the fraud score" in lower
    assert "pushed the fraud score down" in lower


def test_missing_column_and_a_similar_amount_that_still_moves_the_score():
    reference = _zeros(Amount=10)
    short = _frame(reference).drop(columns=["V1"])
    with pytest.raises(ValueError, match="Ticket is missing"):
        explain_ticket(
            None,
            short,
            reference,
            model_key="amount_rule",
            feature_columns=ROOT_COLUMNS,
            score_kind="amount_usd",
        )
    from fraudlab.explain import _amount_sentence

    text = _amount_sentence(
        ScoreDriver("Amount", 10.001, 10.0, 0.2, "up"),
        "fraud score",
        "up",
        "0.200",
    )
    assert "about the same as" in text
    assert "similar amount" in text


def test_explain_guards_and_the_shipped_fraud_ticket():
    reference = _zeros(Amount=10)
    with pytest.raises(ValueError, match="one transaction"):
        explain_ticket(
            None,
            pd.concat([_frame(reference), _frame(reference)], ignore_index=True),
            reference,
            model_key="amount_rule",
            feature_columns=ROOT_COLUMNS,
            score_kind="amount_usd",
        )
    with pytest.raises(ValueError, match="Reference profile"):
        explain_ticket(
            None,
            _frame(reference),
            {"Amount": 10},
            model_key="amount_rule",
            feature_columns=ROOT_COLUMNS,
            score_kind="amount_usd",
        )

    desk = load_desk(ROOT)
    row = desk.demo.loc[desk.demo["demo_id"] == "D002"].iloc[0]
    columns = list(desk.metadata["feature_columns"])
    ticket = _frame({column: float(row[column]) for column in columns}, columns)
    explained = explain_ticket(
        desk.model,
        ticket,
        desk.scenarios["typical_legitimate"],
        model_key=desk.metadata["model_key"],
        feature_columns=columns,
        score_kind=desk.metadata["score_kind"],
    )
    assert explained.score == pytest.approx(float(row["score"]), abs=1e-4)
    assert explained.score > explained.reference_score
    assert any(driver.direction == "up" for driver in explained.drivers)
    text = " ".join(explained.sentences)
    assert "pushed the fraud score up" in text
    assert "merchant" not in text.lower()
    for sentence in explained.sentences:
        up = "pushed the fraud score up" in sentence
        down = "pushed the fraud score down" in sentence
        assert up or down
        assert not (up and down)
