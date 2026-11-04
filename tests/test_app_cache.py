import math
from pathlib import Path

import numpy as np
import pytest

from fraudlab.app_cache import (
    MAX_CURVE_BYTES,
    assert_cloud_budget,
    best_grid_queue,
    build_queue_grid,
    grid_decision,
    load_desk_curve,
    save_desk_curve,
    saved_decision,
    startup_note,
)
from fraudlab.costs import CostConfig, decision_cost, threshold_at_flag_rate
from fraudlab.dashboard import load_desk, missing_desk_files, use_one_worker

ROOT = Path(__file__).resolve().parents[1]


def test_grid_matches_the_mask_and_reprices_without_the_rows(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 80)
    y[0] = 1
    y[1] = 0
    scores = rng.random(80)
    scores[4] = scores[5]
    amounts = rng.uniform(1, 40, 80)
    path = tmp_path / "desk_curve.npz"
    curve = save_desk_curve(path, y, scores, amounts, saved_threshold=0.5)
    loaded = load_desk_curve(path)
    assert curve.n == loaded.n
    assert loaded.n == 80
    base = CostConfig(8, 12, 25)
    pricey = CostConfig(10, 12, 25)
    for k in (0, 1, 17, 80, 1000, -4, 5000):
        threshold, decision = grid_decision(loaded, k, base)
        if k <= 0:
            assert math.isinf(threshold)
        elif k >= 1000:
            assert threshold == pytest.approx(float(np.quantile(scores, 0.9)))
        else:
            assert threshold == pytest.approx(threshold_at_flag_rate(scores, k / 10_000))
        raw = decision_cost(y, scores, amounts, threshold, base)
        assert decision.total_cost == pytest.approx(raw.total_cost)
        assert decision.true_positives == raw.true_positives
        assert decision.false_negatives == raw.false_negatives
        _, dearer = grid_decision(loaded, max(k, 0), pricey)
        extra_reviews = decision.true_positives + decision.false_positives
        assert dearer.total_cost == pytest.approx(decision.total_cost + 2 * extra_reviews)
    saved = saved_decision(loaded, base)
    assert saved.total_cost == pytest.approx(
        decision_cost(y, scores, amounts, 0.5, base).total_cost
    )


def test_free_review_picks_the_shortest_queue_that_catches_the_fraud(tmp_path):
    y = np.array([1, 0, 0, 0])
    scores = np.array([0.9, 0.2, 0.1, 0.05])
    amounts = np.array([10.0, 0.0, 0.0, 0.0])
    wide = build_queue_grid(y, scores, amounts, slider_max=10_000)
    assert wide["threshold"][-1] == pytest.approx(float(np.min(scores)))
    curve = save_desk_curve(
        tmp_path / "desk_curve.npz",
        y,
        scores,
        amounts,
        saved_threshold=float("inf"),
    )
    assert math.isinf(curve.saved_threshold)
    per_10k, threshold, decision = best_grid_queue(curve, CostConfig(0, 0, 0))
    assert decision.total_cost == pytest.approx(0)
    assert decision.false_negatives == 0
    assert math.isfinite(threshold)
    approve_k, approve_threshold, approve = best_grid_queue(curve, CostConfig(10_000, 10_000, 0))
    assert approve_k == 0
    assert math.isinf(approve_threshold)
    assert approve.total_cost == pytest.approx(10)
    assert per_10k > 0
    note = startup_note(
        validation_rows=curve.n, demo_rows=4, grid_points=len(curve.reviews_per_10k)
    )
    assert "not downloaded" in note
    assert "4 sample" in note


def test_grid_guards_and_worker_pin(tmp_path):
    with pytest.raises(ValueError):
        build_queue_grid([0, 1], [0.2], [1.0, 2.0])
    with pytest.raises(ValueError):
        build_queue_grid([], [], [])
    with pytest.raises(ValueError):
        build_queue_grid([0, 2], [0.1, 0.2], [1.0, 1.0])
    with pytest.raises(ValueError):
        build_queue_grid([0, 1], [0.1, np.nan], [1.0, 1.0])
    with pytest.raises(ValueError):
        build_queue_grid([0, 1], [0.1, 0.2], [1.0, 1.0], slider_max=0)
    with pytest.raises(ValueError):
        build_queue_grid([0, 1], [0.1, 0.2], [1.0, -1.0])
    tiny = save_desk_curve(
        tmp_path / "desk_curve.npz",
        np.array([0, 1]),
        np.array([0.1, 0.9]),
        np.array([1.0, 2.0]),
        saved_threshold=0.9,
    )
    with pytest.raises(ValueError):
        grid_decision(tiny, float("nan"), CostConfig())

    class Estimator:
        n_jobs = -1

    class Pipe:
        named_steps = {"model": Estimator()}

    class Bare:
        pass

    class BarePipe:
        named_steps = {"model": Bare()}

    class EmptyStep:
        named_steps = {"model": None}

    pinned = use_one_worker(Pipe())
    assert pinned.named_steps["model"].n_jobs == 1
    assert use_one_worker(None) is None
    assert use_one_worker(object()) is not None
    assert isinstance(use_one_worker(BarePipe()).named_steps["model"], Bare)
    assert use_one_worker(EmptyStep()).named_steps["model"] is None


def test_budget_refuses_an_oversized_cache(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    blob = artifacts / "desk_curve.npz"
    blob.write_bytes(b"x" * (MAX_CURVE_BYTES + 1))
    with pytest.raises(RuntimeError, match="free-tier"):
        assert_cloud_budget(tmp_path)
    assert missing_desk_files(tmp_path)


def test_cloud_contract_and_shipped_cache():
    config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "[server]" in config
    assert "gatherUsageStats = false" in config
    assert "fileWatcherType" in config
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip().startswith("3.12")
    entry = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    app = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    cache = (ROOT / "fraudlab" / "app_cache.py").read_text(encoding="utf-8")
    explain = (ROOT / "fraudlab" / "explain.py").read_text(encoding="utf-8")
    for blob in (entry, app, cache, explain):
        assert "fetch_openml" not in blob
        assert "urlopen" not in blob
        assert "requests" not in blob
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, version = stripped.split("==")
        assert name.strip()
        assert version.strip()

    desk = load_desk(ROOT)
    assert desk.metadata["app_cache"]["loads_public_file"] is False
    assert desk.curve.n == desk.metadata["validation_rows"] == 55_133
    assert len(desk.demo) == 65
    assert len(desk.demo) <= 200
    assert desk.model.named_steps["model"].n_jobs == 1
    curve_bytes = (ROOT / "artifacts" / "desk_curve.npz").stat().st_size
    model_bytes = (ROOT / "artifacts" / "selected_model.joblib").stat().st_size
    demo_bytes = (ROOT / "artifacts" / "demo_transactions.csv").stat().st_size
    assert curve_bytes < 1_000_000
    assert model_bytes < 8_000_000
    assert curve_bytes + model_bytes + demo_bytes < 2_000_000
    cost = CostConfig.from_dict(desk.metadata["cost_config"])
    selected = next(
        row for row in desk.metrics["models"] if row["key"] == desk.metadata["model_key"]
    )
    assert saved_decision(desk.curve, cost).total_cost == pytest.approx(selected["val_cost"])
    raw = np.load(ROOT / "artifacts" / "validation_scores.npz")
    for k in (0, 17, 100, 1000):
        threshold, decision = grid_decision(desk.curve, k, cost)
        if k == 0:
            assert math.isinf(threshold)
        else:
            assert threshold == pytest.approx(threshold_at_flag_rate(raw["score"], k / 10_000))
        direct = decision_cost(raw["y"], raw["score"], raw["amount"], threshold, cost)
        assert decision.true_positives == direct.true_positives
        assert decision.total_cost == pytest.approx(direct.total_cost)


def test_load_desk_rejects_a_missing_cache(tmp_path):
    with pytest.raises(FileNotFoundError, match="desk_curve"):
        load_desk(tmp_path)
