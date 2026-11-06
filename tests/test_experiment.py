import numpy as np
import pandas as pd
import pytest

from fraudlab.app_cache import saved_decision
from fraudlab.cli import main
from fraudlab.constants import TAG_SYNTHETIC
from fraudlab.costs import CostConfig
from fraudlab.dashboard import (
    comparison_frame,
    decode_saved_threshold,
    format_threshold,
    load_desk,
    load_run,
    missing_artifacts,
    missing_desk_files,
    recommended_queue,
    score_frame,
    source_banner,
)
from fraudlab.experiment import run_experiment
from fraudlab.explain import explain_ticket
from fraudlab.models import SPECS, fit_spec
from fraudlab.plots import FIGURE_FILES, focus_queue


def test_amount_rule_does_not_fit_a_model():
    frame = pd.DataFrame(
        {"Amount": [1.0, 2.0, 9.0], "Class": [0, 1, 0], **{f"V{i}": 0.0 for i in range(1, 29)}}
    )
    # The amount spec needs the V columns only when a real model is fit. Amount rule ignores them.
    result = fit_spec(SPECS[-1], frame, fast=True, seed=0)
    assert result.model is None
    assert SPECS[-1].key == "amount_rule"


def test_too_few_frauds_stops_before_a_split(tmp_path):
    with pytest.raises(RuntimeError):
        run_experiment(
            output_dir=tmp_path,
            source="synthetic",
            synthetic_rows=200,
            fraud_rate=0.02,
            fast=True,
            n_boot=5,
        )


def test_cli_scorecard_round_trip(tmp_path):
    code = main(
        [
            "--source",
            "synthetic",
            "--synthetic-rows",
            "4000",
            "--fraud-rate",
            "0.02",
            "--fast",
            "--bootstraps",
            "12",
            "--seed",
            "7",
            "--output",
            str(tmp_path),
            "--review-cost",
            "8",
            "--false-decline-cost",
            "12",
            "--chargeback-fee",
            "25",
        ]
    )
    assert code == 0
    note = (tmp_path / "reports" / "DATA_SOURCE.txt").read_text(encoding="utf-8")
    assert note.startswith("SYNTHETIC FALLBACK")
    assert TAG_SYNTHETIC in note
    for name in FIGURE_FILES:
        path = tmp_path / "reports" / "figures" / name
        assert path.stat().st_size > 1000
        assert path.read_bytes().startswith(b"\x89PNG")
    results = (tmp_path / "reports" / "results.md").read_text(encoding="utf-8")
    assert "Selected model" in results

    assets = load_run(tmp_path)
    assert assets.metadata["data_source"]["kind"] == "synthetic"
    level, banner = source_banner(assets.metadata["data_source"])
    assert level == "warning"
    assert "SYNTHETIC FALLBACK" in banner
    selected = next(
        row for row in assets.metrics["models"] if row["key"] == assets.metadata["model_key"]
    )
    assert selected["selected"] is True
    cost = CostConfig.from_dict(assets.metadata["cost_config"])
    threshold, decision = recommended_queue(assets.y_val, assets.score_val, assets.amount_val, cost)
    assert threshold == pytest.approx(decode_saved_threshold(assets.metadata))
    assert decision.total_cost == pytest.approx(selected["val_cost"])
    columns = assets.metadata["feature_columns"]
    live = score_frame(assets.model, assets.demo[columns], assets.metadata["model_key"])
    assert np.allclose(live, assets.demo["score"].to_numpy(dtype=float), rtol=1e-5, atol=1e-4)
    table = comparison_frame(assets.metrics)
    assert "Test PR-AUC" in table.columns
    assert (table["Selected"] == "yes").sum() == 1
    under = next(row for row in assets.metrics["models"] if row["key"] == "logistic_undersampled")
    assert under["fit_rows"] < assets.metrics["split"]["train"]["rows"]
    assert assets.metrics["split"]["validation"]["fraud_rate"] == pytest.approx(0.02, abs=0.015)
    assert assets.metrics["split"]["test"]["fraud_rate"] == pytest.approx(0.02, abs=0.015)
    assert format_threshold(float("inf"), "probability") == "no review (approve all)"
    assert assets.metadata["app_cache"]["loads_public_file"] is False
    assert assets.metadata["app_cache"]["demo_rows"] == len(assets.demo)
    desk = load_desk(tmp_path)
    if desk.model is not None:
        assert desk.model.named_steps["model"].n_jobs == 1
    assert desk.curve.n == len(assets.y_val)
    replay = saved_decision(desk.curve, cost)
    assert replay.total_cost == pytest.approx(selected["val_cost"])
    assert (tmp_path / "artifacts" / "desk_curve.npz").stat().st_size < 1_000_000
    sample = desk.demo.iloc[0]
    columns = desk.metadata["feature_columns"]
    sample_frame = pd.DataFrame([{column: float(sample[column]) for column in columns}])
    explained = explain_ticket(
        desk.model,
        sample_frame,
        desk.scenarios["typical_legitimate"],
        model_key=desk.metadata["model_key"],
        feature_columns=columns,
        score_kind=desk.metadata["score_kind"],
    )
    assert explained.summary
    assert len(explained.drivers) == len(columns)
    assert missing_desk_files(tmp_path / "missing")
    info_level, _info = source_banner(
        {"kind": "openml", "detail": "checked", "chart_tag": "OpenML"}
    )
    assert info_level == "info"
    assert missing_artifacts(tmp_path / "missing")
    monitoring_path = tmp_path / "artifacts" / "monitoring_reference.json"
    assert monitoring_path.exists()
    card = (tmp_path / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "precision-recall AUC" in card
    assert assets.metadata["model_key"] in card
    assert desk.monitoring["notes"]


def test_focus_queue_hides_the_flag_everyone_tail():
    reviews = np.array([0, 5, 10, 15, 20, 30, 40, 60, 80, 5_000, 10_000], dtype=float)
    costs = np.linspace(8_000, 20_000, len(reviews))
    shown_reviews, shown_costs = focus_queue(reviews, costs, chosen_reviews=12)
    assert shown_reviews.max() <= 100
    assert 10_000 not in shown_reviews
    assert len(shown_costs) == len(shown_reviews)
    short_reviews, short_costs = focus_queue(np.array([0.0, 1.0]), np.array([3.0, 2.0]), 1)
    assert len(short_reviews) == 2
    assert len(short_costs) == 2


def test_help_exits_cleanly():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_streamlit_module_imports():
    import app.streamlit_app as app

    assert callable(app.main)
