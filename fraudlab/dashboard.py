"""Pure helpers for the review desk. Streamlit stays in ``app/``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraudlab.constants import TAG_SYNTHETIC
from fraudlab.costs import (
    CostConfig,
    DecisionCost,
    best_operating_point,
    decode_threshold,
    one_row_cost_table,
)
from fraudlab.models import positive_scores
from fraudlab.reporting import money


@dataclass
class RunAssets:
    root: Path
    metadata: dict
    metrics: dict
    model: object | None
    y_val: np.ndarray
    score_val: np.ndarray
    amount_val: np.ndarray
    scenarios: dict
    demo: pd.DataFrame


def missing_artifacts(root: Path) -> list[str]:
    required = [
        root / "reports" / "metrics.json",
        root / "artifacts" / "metadata.json",
        root / "artifacts" / "validation_scores.npz",
        root / "artifacts" / "scenarios.json",
        root / "artifacts" / "demo_transactions.csv",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    meta_path = root / "artifacts" / "metadata.json"
    model_path = root / "artifacts" / "selected_model.joblib"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("model_key") != "amount_rule" and not model_path.exists():
            missing.append("artifacts/selected_model.joblib")
    elif not model_path.exists():
        missing.append("artifacts/selected_model.joblib")
    return missing


def load_run(root: Path) -> RunAssets:
    missing = missing_artifacts(root)
    if missing:
        raise FileNotFoundError("Missing artifacts: " + ", ".join(missing))
    metadata = json.loads((root / "artifacts" / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "reports" / "metrics.json").read_text(encoding="utf-8"))
    model = None
    if metadata["model_key"] != "amount_rule":
        model = joblib.load(root / "artifacts" / "selected_model.joblib")
    arrays = np.load(root / "artifacts" / "validation_scores.npz")
    scenarios = json.loads((root / "artifacts" / "scenarios.json").read_text(encoding="utf-8"))
    demo = pd.read_csv(root / "artifacts" / "demo_transactions.csv")
    return RunAssets(
        root=root,
        metadata=metadata,
        metrics=metrics,
        model=model,
        y_val=arrays["y"].astype(int),
        score_val=arrays["score"].astype(float),
        amount_val=arrays["amount"].astype(float),
        scenarios=scenarios,
        demo=demo,
    )


def source_banner(info: dict) -> tuple[str, str]:
    if info.get("kind") == "synthetic" or info.get("chart_tag") == TAG_SYNTHETIC:
        return (
            "warning",
            "SYNTHETIC FALLBACK. This desk was not fit on the public OpenML transactions. "
            + str(info.get("detail", "")),
        )
    return (
        "info",
        "Trained on the public OpenML creditcard file (data id 1597), after the fingerprint "
        "check and duplicate drop. " + str(info.get("detail", "")),
    )


def score_frame(model: object | None, frame: pd.DataFrame, model_key: str) -> np.ndarray:
    if model_key == "amount_rule":
        return frame["Amount"].to_numpy(dtype=float)
    if model is None:
        raise RuntimeError("No model is loaded")
    return positive_scores(model, frame)


def frame_from_mapping(mapping: dict, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{column: float(mapping[column]) for column in columns}])


def format_threshold(threshold: float, score_kind: str) -> str:
    if not np.isfinite(threshold):
        return "no review (approve all)"
    if score_kind == "amount_usd":
        return money(threshold, cents=True)
    return f"{threshold:.4f}"


def queue_decision(
    y_val: np.ndarray,
    score_val: np.ndarray,
    amount_val: np.ndarray,
    cost: CostConfig,
    reviews_per_10k: float,
) -> tuple[float, DecisionCost]:
    from fraudlab.costs import decision_cost, threshold_at_flag_rate

    threshold = threshold_at_flag_rate(score_val, reviews_per_10k / 10_000)
    return threshold, decision_cost(y_val, score_val, amount_val, threshold, cost)


def recommended_queue(
    y_val: np.ndarray,
    score_val: np.ndarray,
    amount_val: np.ndarray,
    cost: CostConfig,
) -> tuple[float, DecisionCost]:
    return best_operating_point(y_val, score_val, amount_val, cost)


def comparison_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for row in metrics["models"]:
        rows.append(
            {
                "Model": row["display_name"],
                "Selected": "yes" if row["key"] == metrics["selected_model"] else "",
                "Test PR-AUC": round(float(row["test_average_precision"]), 3),
                "Recall": round(float(row["test_recall"]), 3),
                "Precision": None
                if row["test_precision"] is None
                else round(float(row["test_precision"]), 3),
                "Test cost (USD)": round(float(row["test_cost"]), 2),
                "Savings vs approve-all (USD)": round(float(row["savings_vs_approve_all"]), 2),
            }
        )
    return pd.DataFrame(rows)


def single_ticket_costs(amount: float, cost: CostConfig) -> dict[str, float]:
    return one_row_cost_table(amount, cost)


def decode_saved_threshold(metadata: dict) -> float:
    return decode_threshold(metadata.get("threshold"))
