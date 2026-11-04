import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from fraudlab.costs import CostConfig
from fraudlab.metrics import (
    average_precision,
    bootstrap_test_metrics,
    choose_model,
    recall_at_precision,
)
from fraudlab.reporting import dump_json, format_ci, json_safe, money, percent
from fraudlab.sampling import balanced_sample_weight, random_undersample


def test_recall_at_precision_ignores_the_empty_endpoint():
    y = np.array([1, 0, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.4, 0.3, 0.1])
    assert recall_at_precision(y, scores, 0.60) == pytest.approx(1.0)
    assert recall_at_precision(y, scores, 0.90) == pytest.approx(0.5)
    assert recall_at_precision(y, scores, 0.99) == pytest.approx(0.5)
    assert average_precision(y, scores) == pytest.approx(average_precision_score(y, scores))
    with pytest.raises(ValueError):
        recall_at_precision(y, scores, 0)
    hopeless = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    # Ranking puts both frauds last, so precision never reaches 0.8 at a real cutoff.
    assert recall_at_precision(y, hopeless, 0.80) is None


def test_bootstrap_interval_and_model_choice():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 80)
    y[:4] = 1
    scores = rng.random(80)
    amounts = rng.uniform(5, 40, 80)
    stats = bootstrap_test_metrics(y, scores, amounts, 0.4, CostConfig(), n_boot=25, seed=3)
    lo, hi = stats["cost_ci95"]
    assert lo <= hi
    assert stats["n_boot"] == 25
    chosen = choose_model(
        [
            {"key": "b", "val_cost": 10, "val_average_precision": 0.4},
            {"key": "a", "val_cost": 10, "val_average_precision": 0.9},
            {"key": "c", "val_cost": 12, "val_average_precision": 0.99},
        ]
    )
    assert chosen == "a"
    with pytest.raises(ValueError):
        choose_model([])


def test_undersample_keeps_every_positive_and_balances_weight():
    frame = pd.DataFrame(
        {
            "Amount": np.arange(30),
            "Class": [1] * 4 + [0] * 26,
        }
    )
    under = random_undersample(frame, negative_per_positive=3, seed=4)
    assert int(under["Class"].sum()) == 4
    assert int((under["Class"] == 0).sum()) == 12
    weights = balanced_sample_weight(frame["Class"].to_numpy())
    y = frame["Class"].to_numpy()
    assert weights[y == 1].sum() == pytest.approx(weights[y == 0].sum())
    with pytest.raises(ValueError):
        random_undersample(frame, negative_per_positive=0, seed=1)
    with pytest.raises(ValueError):
        balanced_sample_weight(np.ones(5, dtype=int))


def test_money_and_json_safe(tmp_path):
    assert money(-1200) == "-$1,200"
    assert money(10.5, cents=True) == "$10.50"
    assert percent(0.172) == "17.2%"
    assert format_ci(0.5, [0.4, 0.6]) == "0.500 (0.400–0.600)"
    payload = {"a": np.float64("nan"), "b": np.int64(3), "c": np.bool_(True)}
    safe = json_safe(payload)
    assert safe["a"] is None
    assert safe["b"] == 3
    assert safe["c"] is True
    dump_json(tmp_path / "x.json", payload)
    assert "null" in (tmp_path / "x.json").read_text()
