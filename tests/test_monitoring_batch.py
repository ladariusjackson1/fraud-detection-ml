import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudlab.batch import prepare_batch, score_batch, template_from_demo, to_csv_bytes
from fraudlab.costs import CostConfig
from fraudlab.dashboard import headline_comparison, load_desk, overview_blurb, single_ticket_costs
from fraudlab.model_card import render_model_card
from fraudlab.monitoring import (
    batch_psi,
    build_monitoring_reference,
    half_split_psi,
    monitoring_notes,
    population_stability_index,
    profile_table,
    quantile_block,
    reference_profile,
)
from fraudlab.reporting import format_ci, money
from fraudlab.stream import replay_tally, score_demo_row

ROOT = Path(__file__).resolve().parents[1]


def _frame(n=6, fraud_at=0):
    data = {f"V{i}": np.linspace(0, 1, n) for i in range(1, 29)}
    data["Amount"] = np.linspace(5, 40, n)
    data["Class"] = np.zeros(n, dtype=int)
    data["Class"][fraud_at] = 1
    return pd.DataFrame(data)


def test_psi_is_near_zero_on_one_sample_and_moves_when_the_batch_shifts():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=400)
    assert population_stability_index(reference, reference) < 0.02
    assert population_stability_index(reference, reference + 4.0) > 0.25
    assert population_stability_index(np.ones(30), np.ones(10)) == 0.0
    assert half_split_psi(reference, seed=1) >= 0
    assert batch_psi(reference, reference[:10]) is None
    assert batch_psi(reference, reference) is not None
    with pytest.raises(ValueError):
        population_stability_index(reference, reference, bins=1)
    with pytest.raises(ValueError):
        population_stability_index(reference, reference, epsilon=0)
    with pytest.raises(ValueError):
        population_stability_index(reference[:3], reference, bins=10)
    with pytest.raises(ValueError):
        half_split_psi(reference[:5])
    with pytest.raises(ValueError):
        quantile_block(np.array([]))
    assert set(quantile_block(reference)) == {"p50", "p90", "p99", "p999"}


def test_reference_profile_and_notes_cover_both_clocks():
    y = np.array([0, 0, 0, 1, 0, 1, 0, 0] * 5)
    scores = np.linspace(0.01, 0.99, len(y))
    amounts = np.linspace(1, 50, len(y))
    cost = CostConfig()
    profile = reference_profile(y, scores, amounts, 0.8, cost)
    assert profile["frauds"] == int(y.sum())
    assert profile["precision"] is not None
    quiet = build_monitoring_reference(y, scores, amounts, float("inf"), cost, has_time=False)
    assert quiet["precision"] is None
    assert quiet["has_time"] is False
    assert "no Time column" in quiet["notes"][0]
    assert "undefined" in quiet["notes"][1]
    timed = monitoring_notes(profile, has_time=True)
    assert "Time column" in timed[0]
    assert "PSI" in timed[2]
    table = profile_table(profile)
    assert list(table["Slice"])[0] == "Score, all"
    with pytest.raises(ValueError):
        reference_profile(np.zeros(8, dtype=int), scores[:8], amounts[:8], 0.5, cost)
    with pytest.raises(ValueError):
        reference_profile(y, scores[:-1], amounts, 0.5, cost)
    with pytest.raises(ValueError):
        reference_profile(np.array([0, 2, 1, 0]), scores[:4], amounts[:4], 0.5, cost)
    with pytest.raises(ValueError):
        population_stability_index(np.array([np.nan, 1.0, 2.0, 3.0]), np.ones(4))


def test_amount_rule_batch_and_stream_round_trip():
    columns = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    frame = _frame()
    cost = CostConfig()
    scored = score_batch(
        frame,
        None,
        model_key="amount_rule",
        feature_columns=columns,
        score_kind="amount_usd",
        threshold=20,
        reference={column: 0.0 for column in columns} | {"Amount": 12.0},
        cost=cost,
        explain_limit=1,
    )
    assert list(scored["decision"]).count("review") >= 1
    assert scored.loc[scored["decision"] == "review", "top_reason"].iloc[0]
    assert "realized_cost" in scored.columns
    again = score_batch(
        scored,
        None,
        model_key="amount_rule",
        feature_columns=columns,
        score_kind="amount_usd",
        threshold=20,
        reference=None,
        cost=cost,
        explain_limit=0,
    )
    assert again["top_reason"].eq("").all()
    csv_text = to_csv_bytes(again).decode("utf-8")
    assert "fraud_score" in csv_text
    silent = score_batch(
        frame.drop(columns=["Class"]),
        None,
        model_key="amount_rule",
        feature_columns=columns,
        score_kind="amount_usd",
        threshold=float("inf"),
        reference=None,
        cost=cost,
    )
    assert set(silent["decision"]) == {"approve"}
    with pytest.raises(ValueError, match="missing"):
        prepare_batch(frame.drop(columns=["Amount"]), columns)
    with pytest.raises(ValueError, match="no rows"):
        prepare_batch(frame.iloc[0:0], columns)
    with pytest.raises(ValueError, match="2,000"):
        prepare_batch(pd.concat([frame] * 400, ignore_index=True), columns)
    broken = frame.astype(object)
    broken.loc[0, "V1"] = "nope"
    with pytest.raises(ValueError, match="numeric"):
        prepare_batch(broken, columns)
    negative = frame.copy()
    negative.loc[0, "Amount"] = -1
    with pytest.raises(ValueError, match="Amount"):
        prepare_batch(negative, columns)
    bad_label = frame.copy()
    bad_label.loc[0, "Class"] = 2
    with pytest.raises(ValueError, match="Class"):
        prepare_batch(bad_label, columns)
    with pytest.raises(TypeError):
        prepare_batch([], columns)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        score_batch(
            frame,
            None,
            model_key="amount_rule",
            feature_columns=columns,
            score_kind="amount_usd",
            threshold=1,
            reference=None,
            cost=cost,
            explain_limit=-1,
        )
    demo = frame.copy()
    demo.insert(0, "demo_id", [f"D{i}" for i in range(len(demo))])
    template = template_from_demo(demo, columns, n=3)
    assert list(template.columns) == columns
    assert len(template) == 3
    with pytest.raises(ValueError):
        template_from_demo(demo, columns, n=0)
    with pytest.raises(ValueError):
        template_from_demo(demo.iloc[0:0], columns, n=1)
    with pytest.raises(ValueError, match="missing"):
        template_from_demo(demo.drop(columns=["V1"]), columns, n=1)
    ticket = score_demo_row(
        demo,
        0,
        None,
        model_key="amount_rule",
        feature_columns=columns,
        score_kind="amount_usd",
        reference={column: 0.0 for column in columns} | {"Amount": 12.0},
        threshold=4,
    )
    assert ticket.flagged
    assert ticket.at_threshold(10_000).flagged is False
    assert ticket.label == 1
    with pytest.raises(IndexError):
        score_demo_row(
            demo,
            99,
            None,
            model_key="amount_rule",
            feature_columns=columns,
            score_kind="amount_usd",
            reference={column: 0.0 for column in columns},
            threshold=1,
        )
    with pytest.raises(ValueError, match="missing"):
        score_demo_row(
            demo.drop(columns=["V2"]),
            0,
            None,
            model_key="amount_rule",
            feature_columns=columns,
            score_kind="amount_usd",
            reference={column: 0.0 for column in columns},
            threshold=1,
        )
    with pytest.raises(ValueError, match="empty"):
        score_demo_row(
            demo.iloc[0:0],
            0,
            None,
            model_key="amount_rule",
            feature_columns=columns,
            score_kind="amount_usd",
            reference={},
            threshold=1,
        )
    unlabeled = demo.drop(columns=["Class"])
    blank = score_demo_row(
        unlabeled,
        1,
        None,
        model_key="amount_rule",
        feature_columns=columns,
        score_kind="amount_usd",
        reference={column: 0.0 for column in columns} | {"Amount": 12.0},
        threshold=10,
    )
    assert blank.label is None
    tally = replay_tally(
        [
            {"flagged": True, "label": 1, "amount": 40},
            {"flagged": False, "label": 1, "amount": 10},
            {"flagged": True, "label": 0, "amount": 8},
            {"flagged": False, "label": 0, "amount": 8},
        ],
        cost,
    )
    assert tally.frauds_caught == 1
    assert tally.frauds_missed == 1
    assert tally.false_alerts == 1
    assert tally.legitimate_approved == 1
    assert replay_tally([], cost).seen == 0
    with pytest.raises(ValueError):
        replay_tally([{"flagged": True, "label": None, "amount": 1}], cost)
    with pytest.raises(ValueError):
        replay_tally([{"flagged": True, "label": 3, "amount": 1}], cost)


def test_saved_forest_scores_match_the_demo_file_and_the_card_matches_the_scorecard():
    assets = load_desk(ROOT)
    columns = assets.metadata["feature_columns"]
    demo = assets.demo
    ticket = score_demo_row(
        demo,
        0,
        assets.model,
        model_key=assets.metadata["model_key"],
        feature_columns=columns,
        score_kind=assets.metadata["score_kind"],
        reference=assets.scenarios["typical_legitimate"],
        threshold=assets.metadata["threshold"],
    )
    assert ticket.score == pytest.approx(float(demo.iloc[0]["score"]), rel=1e-5, abs=1e-4)
    assert ticket.explanation.summary
    sample = template_from_demo(demo, columns, n=2)
    scored = score_batch(
        sample,
        assets.model,
        model_key=assets.metadata["model_key"],
        feature_columns=columns,
        score_kind=assets.metadata["score_kind"],
        threshold=assets.metadata["threshold"],
        reference=assets.scenarios["typical_legitimate"],
        cost=CostConfig.from_dict(assets.metadata["cost_config"]),
        explain_limit=2,
    )
    assert np.allclose(
        scored["fraud_score"], demo["score"].iloc[:2].to_numpy(dtype=float), rtol=1e-5, atol=1e-4
    )
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    rendered = render_model_card(metrics, assets.monitoring)
    assert (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8") == rendered
    selected = next(row for row in metrics["models"] if row["key"] == metrics["selected_model"])
    assert (
        format_ci(selected["test_average_precision"], selected["test_average_precision_ci95"])
        in rendered
    )
    assert money(selected["test_cost"]) in rendered
    assert "Logistic regression" in rendered
    assert "Gradient boosting" in rendered
    table = headline_comparison(metrics)
    assert list(table["Model"]) == [
        "Logistic regression",
        "Random forest",
        "Gradient boosting",
    ]
    assert format_ci(
        selected["test_average_precision"], selected["test_average_precision_ci95"]
    ) in set(table["Test PR-AUC (95% CI)"])
    blurb = overview_blurb(metrics)
    assert money(selected["test_cost"]) in blurb
    assert metrics["test_cost_leader"] in blurb
    matched = json.loads(json.dumps(metrics))
    matched["test_cost_leader"] = matched["selected_model"]
    matched["selection_matches_test_leader"] = True
    assert "also the test-cost leader" in overview_blurb(matched)
    with pytest.raises(KeyError):
        headline_comparison({"models": [], "selected_model": "missing"})
    with pytest.raises(ValueError):
        render_model_card(metrics, {"has_time": False})
    costs = single_ticket_costs(10, CostConfig())
    assert costs["approve_if_fraud"] == pytest.approx(35)
    quiet = json.loads(json.dumps(metrics))
    chosen = next(row for row in quiet["models"] if row["key"] == quiet["selected_model"])
    chosen["test_cost_ci95"] = [None, None]
    chosen["threshold"] = None
    assert "n/a" in render_model_card(quiet, assets.monitoring)
    assert "approve all" in render_model_card(quiet, assets.monitoring)
    priced = json.loads(json.dumps(metrics))
    priced_row = next(row for row in priced["models"] if row["key"] == priced["selected_model"])
    priced_row["score_kind"] = "amount_usd"
    priced_row["threshold"] = 10.5
    assert "$10.50" in render_model_card(priced, assets.monitoring)
    dropped = json.loads(json.dumps(metrics))
    dropped["models"] = [row for row in dropped["models"] if row["key"] != "gradient_boosting"]
    with pytest.raises(KeyError):
        render_model_card(dropped, assets.monitoring)
