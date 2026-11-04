"""Train, select on validation cost, and write the scorecard.

The test set is scored once, at cutoffs chosen on validation. Bootstrap intervals
resample the test rows and do not retune the cutoff or the model.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import logging
import platform
import shutil
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

matplotlib.use("Agg", force=True)

from fraudlab.constants import (
    DEFAULT_SEED,
    DEFAULT_USD_PER_EUR,
    PRECISION_TARGETS,
    PUBLIC_ROWS,
)
from fraudlab.costs import CostConfig, best_operating_point, decision_cost, encode_threshold
from fraudlab.data import load_dataset
from fraudlab.metrics import (
    average_precision,
    bootstrap_test_metrics,
    choose_model,
    recall_at_precision,
)
from fraudlab.models import SPECS, fit_spec, positive_scores
from fraudlab.plots import (
    save_amount_by_class,
    save_class_balance,
    save_confusion,
    save_cost_curve,
    save_model_comparison,
    save_pr_curves,
    save_score_distributions,
)
from fraudlab.reporting import dump_json, render_results_markdown, write_data_source_note

logger = logging.getLogger("fraudlab")


def run_experiment(
    *,
    output_dir: Path | str = ".",
    source: str = "auto",
    synthetic_rows: int = PUBLIC_ROWS,
    fraud_rate: float | None = None,
    random_state: int = DEFAULT_SEED,
    usd_per_eur: float = DEFAULT_USD_PER_EUR,
    cost: CostConfig | None = None,
    fast: bool = False,
    n_boot: int = 400,
) -> dict:
    root = Path(output_dir)
    cost = cost or CostConfig()
    n_boot = 20 if fast else n_boot
    artifacts = root / "artifacts"
    figures = root / "reports" / "figures"
    artifacts.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    candidate_dir = artifacts / "_candidates"
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir(parents=True)

    started = time.perf_counter()
    try:
        frame, source_info = load_dataset(
            source=source,
            n_rows=synthetic_rows,
            fraud_rate=fraud_rate,
            random_state=random_state,
            usd_per_eur=usd_per_eur,
            data_home=root / "data" / "openml_cache",
        )
        _require_both_classes(frame, minimum=12 if fast else 40)
        train, validation, test = _split(frame, random_state)
        descriptives = _amount_descriptives(frame)
        records: list[dict] = []
        score_bank: dict[str, dict[str, np.ndarray]] = {}
        curves: list[dict] = []
        for spec in SPECS:
            tick = time.perf_counter()
            logger.info("Fitting %s", spec.key)
            fitted = fit_spec(spec, train, fast=fast, seed=random_state)
            if spec.family == "amount":
                val_scores = validation["Amount"].to_numpy(dtype=float)
                test_scores = test["Amount"].to_numpy(dtype=float)
            else:
                assert fitted.model is not None
                val_scores = positive_scores(fitted.model, validation.drop(columns=["Class"]))
                test_scores = positive_scores(fitted.model, test.drop(columns=["Class"]))
                joblib.dump(fitted.model, candidate_dir / f"{spec.key}.joblib", compress=3)
            record, bank = _score_spec(
                spec=spec,
                fitted_rows=fitted.fit_rows,
                fit_detail=fitted.fit_detail,
                hyperparameters=fitted.hyperparameters,
                validation=validation,
                test=test,
                val_scores=val_scores,
                test_scores=test_scores,
                cost=cost,
                n_boot=n_boot,
                seed=random_state,
            )
            records.append(record)
            score_bank[spec.key] = bank
            precision, recall, _thresholds = precision_recall_curve(bank["y_test"], test_scores)
            curves.append(
                {
                    "key": spec.key,
                    "name": spec.display_name,
                    "precision": precision,
                    "recall": recall,
                }
            )
            logger.info("Scored %s in %.1fs", spec.key, time.perf_counter() - tick)

        selected_key = choose_model(records)
        for record in records:
            record["selected"] = record["key"] == selected_key
        _promote_model(candidate_dir, artifacts, selected_key)
        metrics = _assemble_metrics(
            source_info=source_info,
            cost=cost,
            random_state=random_state,
            fast=fast,
            n_boot=n_boot,
            usd_per_eur=usd_per_eur,
            train=train,
            validation=validation,
            test=test,
            descriptives=descriptives,
            records=records,
            selected_key=selected_key,
            elapsed=time.perf_counter() - started,
        )
        _write_artifacts(
            root=root,
            metrics=metrics,
            source_info=source_info,
            records=records,
            score_bank=score_bank,
            curves=curves,
            train=train,
            validation=validation,
            test=test,
            selected_key=selected_key,
            cost=cost,
        )
        return metrics
    finally:
        if candidate_dir.exists():
            shutil.rmtree(candidate_dir)


def _require_both_classes(frame: pd.DataFrame, *, minimum: int) -> None:
    frauds = int(frame["Class"].sum())
    legit = int(len(frame) - frauds)
    if frauds < minimum or legit < minimum:
        raise RuntimeError(
            f"Need at least {minimum} rows of each class to run the protocol; "
            f"got {frauds} fraud and {legit} legitimate"
        )


def _split(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(
        frame,
        test_size=0.20,
        random_state=seed,
        stratify=frame["Class"],
    )
    train, validation = train_test_split(
        train_val,
        test_size=0.25,
        random_state=seed,
        stratify=train_val["Class"],
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def _amount_descriptives(frame: pd.DataFrame) -> dict:
    summary = {}
    for label, name in ((0, "legitimate"), (1, "fraud")):
        amounts = frame.loc[frame["Class"] == label, "Amount"]
        summary[name] = {
            "count": int(amounts.shape[0]),
            "mean": float(amounts.mean()),
            "median": float(amounts.median()),
            "p90": float(amounts.quantile(0.90)),
            "max": float(amounts.max()),
        }
    return summary


def _score_spec(
    *,
    spec,
    fitted_rows: int,
    fit_detail: str,
    hyperparameters: dict,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    cost: CostConfig,
    n_boot: int,
    seed: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    y_val = validation["Class"].to_numpy(dtype=int)
    y_test = test["Class"].to_numpy(dtype=int)
    amount_val = validation["Amount"].to_numpy(dtype=float)
    amount_test = test["Amount"].to_numpy(dtype=float)
    threshold, val_decision = best_operating_point(y_val, val_scores, amount_val, cost)
    test_decision = decision_cost(y_test, test_scores, amount_test, threshold, cost)
    approve_all = decision_cost(y_test, test_scores, amount_test, float("inf"), cost).total_cost
    val_ap = average_precision(y_val, val_scores)
    test_ap = average_precision(y_test, test_scores)
    intervals = bootstrap_test_metrics(
        y_test,
        test_scores,
        amount_test,
        threshold,
        cost,
        n_boot=n_boot,
        seed=seed + 1,
    )
    half = None
    if spec.score_kind == "probability":
        half = decision_cost(y_test, test_scores, amount_test, 0.5, cost).total_cost
    record = {
        "key": spec.key,
        "display_name": spec.display_name,
        "strategy": spec.strategy,
        "family": spec.family,
        "score_kind": spec.score_kind,
        "fit_rows": int(fitted_rows),
        "fit_detail": fit_detail,
        "hyperparameters": hyperparameters,
        "threshold": encode_threshold(threshold),
        "val_cost": val_decision.total_cost,
        "val_average_precision": val_ap,
        "val_precision": None
        if not np.isfinite(val_decision.precision)
        else val_decision.precision,
        "val_recall": val_decision.recall,
        "val_flag_rate": val_decision.flag_rate,
        "val_reviews_per_10k": val_decision.reviews_per_10k,
        "test_average_precision": test_ap,
        "test_average_precision_ci95": intervals["average_precision_ci95"],
        "test_cost": test_decision.total_cost,
        "test_cost_ci95": intervals["cost_ci95"],
        "n_boot": intervals["n_boot"],
        "approve_all_test_cost": approve_all,
        "savings_vs_approve_all": approve_all - test_decision.total_cost,
        "savings_per_10k": (approve_all - test_decision.total_cost) / len(y_test) * 10_000,
        "test_precision": None
        if not np.isfinite(test_decision.precision)
        else test_decision.precision,
        "test_recall": test_decision.recall,
        "test_flag_rate": test_decision.flag_rate,
        "test_reviews_per_10k": test_decision.reviews_per_10k,
        "test_analyst_hours": test_decision.analyst_hours,
        "recall_at_precision_0_50": recall_at_precision(y_test, test_scores, PRECISION_TARGETS[0]),
        "recall_at_precision_0_80": recall_at_precision(y_test, test_scores, PRECISION_TARGETS[1]),
        "test_true_positives": test_decision.true_positives,
        "test_false_positives": test_decision.false_positives,
        "test_true_negatives": test_decision.true_negatives,
        "test_false_negatives": test_decision.false_negatives,
        "test_false_negative_amount": test_decision.false_negative_amount,
        "test_cost_at_0_5": half,
    }
    bank = {
        "y_val": y_val,
        "score_val": val_scores,
        "amount_val": amount_val,
        "y_test": y_test,
        "score_test": test_scores,
        "amount_test": amount_test,
    }
    return record, bank


def _promote_model(candidate_dir: Path, artifacts: Path, selected_key: str) -> None:
    destination = artifacts / "selected_model.joblib"
    source = candidate_dir / f"{selected_key}.joblib"
    if source.exists():
        if destination.exists():
            destination.unlink()
        shutil.move(str(source), str(destination))
    elif destination.exists():
        destination.unlink()


def _split_block(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "frauds": int(frame["Class"].sum()),
        "fraud_rate": float(frame["Class"].mean()),
    }


def _assemble_metrics(
    *,
    source_info: dict,
    cost: CostConfig,
    random_state: int,
    fast: bool,
    n_boot: int,
    usd_per_eur: float,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    descriptives: dict,
    records: list[dict],
    selected_key: str,
    elapsed: float,
) -> dict:
    import sklearn

    test_leader = sorted(
        records, key=lambda row: (row["test_cost"], -row["test_average_precision"])
    )[0]
    half_rows = [
        {
            "key": row["key"],
            "display_name": row["display_name"],
            "test_cost_at_0_5": row["test_cost_at_0_5"],
            "test_cost": row["test_cost"],
        }
        for row in records
        if row["test_cost_at_0_5"] is not None
    ]
    selected = next(row for row in records if row["key"] == selected_key)
    return {
        "data_source": source_info,
        "cost_config": cost.as_dict(),
        "usd_per_eur_argument": None if source_info["kind"] == "synthetic" else float(usd_per_eur),
        "random_state": int(random_state),
        "fast": bool(fast),
        "n_boot": int(n_boot),
        "selection_rule": (
            "Minimum validation dollar cost. Ties break toward higher validation "
            "PR-AUC, then the model key. The test set is not used to choose the model "
            "or the cutoff."
        ),
        "selected_model": selected_key,
        "test_cost_leader": test_leader["key"],
        "selection_matches_test_leader": test_leader["key"] == selected_key,
        "approve_all_test_cost": selected["approve_all_test_cost"],
        "split": {
            "train": _split_block(train),
            "validation": _split_block(validation),
            "test": _split_block(test),
            "fractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
        },
        "descriptives_usd": descriptives,
        "models": records,
        "default_cutoff_comparison": half_rows,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "runtime_seconds": float(elapsed),
    }


def _write_artifacts(
    *,
    root: Path,
    metrics: dict,
    source_info: dict,
    records: list[dict],
    score_bank: dict[str, dict[str, np.ndarray]],
    curves: list[dict],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    selected_key: str,
    cost: CostConfig,
) -> None:
    selected = next(row for row in records if row["key"] == selected_key)
    bank = score_bank[selected_key]
    feature_columns = [column for column in train.columns if column != "Class"]
    metadata = {
        "model_key": selected_key,
        "display_name": selected["display_name"],
        "score_kind": selected["score_kind"],
        "threshold": selected["threshold"],
        "cost_config": cost.as_dict(),
        "feature_columns": feature_columns,
        "has_time": "Time" in feature_columns,
        "data_source": source_info,
        "test_summary": {
            "average_precision": selected["test_average_precision"],
            "average_precision_ci95": selected["test_average_precision_ci95"],
            "precision": selected["test_precision"],
            "recall": selected["test_recall"],
            "flag_rate": selected["test_flag_rate"],
            "reviews_per_10k": selected["test_reviews_per_10k"],
            "cost": selected["test_cost"],
            "cost_ci95": selected["test_cost_ci95"],
            "savings_vs_approve_all": selected["savings_vs_approve_all"],
            "approve_all_cost": selected["approve_all_test_cost"],
        },
        "validation_rows": int(len(validation)),
        "random_state": metrics["random_state"],
    }
    artifacts = root / "artifacts"
    dump_json(artifacts / "metadata.json", metadata)
    dump_json(root / "reports" / "metrics.json", metrics)
    np.savez_compressed(
        artifacts / "validation_scores.npz",
        y=bank["y_val"].astype(np.int8),
        score=bank["score_val"].astype(np.float64),
        amount=bank["amount_val"].astype(np.float64),
    )
    _write_demo(artifacts, test, bank["score_test"], metrics["random_state"])
    dump_json(artifacts / "scenarios.json", _profiles(train, feature_columns))
    write_data_source_note(root / "reports" / "DATA_SOURCE.txt", source_info)
    (root / "reports" / "results.md").write_text(render_results_markdown(metrics), encoding="utf-8")
    tag = source_info["chart_tag"]
    figures = root / "reports" / "figures"
    save_class_balance(
        figures / "01_class_balance.png",
        legitimate=int(
            (train["Class"] == 0).sum()
            + (validation["Class"] == 0).sum()
            + (test["Class"] == 0).sum()
        ),
        fraud=int(train["Class"].sum() + validation["Class"].sum() + test["Class"].sum()),
        source_tag=tag,
    )
    save_amount_by_class(
        figures / "02_amount_by_class.png",
        legitimate_amounts=pd.concat(
            [
                train.loc[train["Class"] == 0, "Amount"],
                validation.loc[validation["Class"] == 0, "Amount"],
                test.loc[test["Class"] == 0, "Amount"],
            ]
        ).to_numpy(dtype=float),
        fraud_amounts=pd.concat(
            [
                train.loc[train["Class"] == 1, "Amount"],
                validation.loc[validation["Class"] == 1, "Amount"],
                test.loc[test["Class"] == 1, "Amount"],
            ]
        ).to_numpy(dtype=float),
        source_tag=tag,
    )
    save_pr_curves(
        figures / "03_precision_recall.png",
        curves=curves,
        prevalence=float(test["Class"].mean()),
        source_tag=tag,
        selected_key=selected_key,
    )
    curve = _validation_curve_arrays(bank, cost)
    chosen = decision_cost(
        bank["y_val"], bank["score_val"], bank["amount_val"], _threshold(selected), cost
    )
    save_cost_curve(
        figures / "04_cost_vs_threshold.png",
        reviews_per_10k=curve["reviews_per_10k"],
        costs=curve["total_cost"],
        chosen_reviews=chosen.reviews_per_10k,
        chosen_cost=chosen.total_cost,
        model_name=selected["display_name"],
        source_tag=tag,
    )
    save_model_comparison(
        figures / "05_model_comparison.png",
        names=[row["display_name"] for row in records],
        keys=[row["key"] for row in records],
        pr_auc=[row["test_average_precision"] for row in records],
        savings=[row["savings_vs_approve_all"] for row in records],
        selected_key=selected_key,
        source_tag=tag,
    )
    save_confusion(
        figures / "06_confusion_matrix.png",
        true_negatives=selected["test_true_negatives"],
        false_positives=selected["test_false_positives"],
        false_negatives=selected["test_false_negatives"],
        true_positives=selected["test_true_positives"],
        false_negative_amount=selected["test_false_negative_amount"],
        model_name=selected["display_name"],
        source_tag=tag,
    )
    y_test = bank["y_test"]
    save_score_distributions(
        figures / "07_score_distributions.png",
        legitimate_scores=bank["score_test"][y_test == 0],
        fraud_scores=bank["score_test"][y_test == 1],
        threshold=_threshold(selected),
        score_label=_score_label(selected["score_kind"]),
        model_name=selected["display_name"],
        source_tag=tag,
    )


def _threshold(record: dict) -> float:
    if record["threshold"] is None:
        return float("inf")
    return float(record["threshold"])


def _score_label(score_kind: str) -> str:
    if score_kind == "amount_usd":
        return "Amount (USD)"
    if score_kind == "anomaly":
        return "Anomaly score (higher means more unusual)"
    return "Fraud score (class-weighted; not a calibrated probability)"


def _validation_curve_arrays(
    bank: dict[str, np.ndarray], cost: CostConfig
) -> dict[str, np.ndarray]:
    from fraudlab.costs import threshold_curve

    curve = threshold_curve(bank["y_val"], bank["score_val"], bank["amount_val"], cost)
    return {
        "reviews_per_10k": curve["reviews_per_10k"].to_numpy(dtype=float),
        "total_cost": curve["total_cost"].to_numpy(dtype=float),
    }


def _write_demo(artifacts: Path, test: pd.DataFrame, scores: np.ndarray, seed: int) -> None:
    rng = np.random.default_rng(seed)
    fraud_pos = np.flatnonzero(test["Class"].to_numpy() == 1)
    legit_pos = np.flatnonzero(test["Class"].to_numpy() == 0)
    take_fraud = rng.choice(fraud_pos, size=min(25, len(fraud_pos)), replace=False)
    take_legit = rng.choice(legit_pos, size=min(40, len(legit_pos)), replace=False)
    chosen = np.concatenate([take_fraud, take_legit])
    rng.shuffle(chosen)
    demo = test.iloc[chosen].copy().reset_index(drop=True)
    demo.insert(0, "demo_id", [f"D{i:03d}" for i in range(1, len(demo) + 1)])
    demo.insert(1, "score", scores[chosen])
    demo.to_csv(artifacts / "demo_transactions.csv", index=False)


def _profiles(train: pd.DataFrame, feature_columns: list[str]) -> dict[str, dict[str, float]]:
    fraud = train[train["Class"] == 1]
    legit = train[train["Class"] == 0]
    median_amount = float(fraud["Amount"].median())
    smaller = fraud[fraud["Amount"] <= median_amount]
    larger = fraud[fraud["Amount"] > median_amount]
    if smaller.empty:
        smaller = fraud
    if larger.empty:
        larger = fraud

    def pack(frame: pd.DataFrame) -> dict[str, float]:
        medians = frame[feature_columns].median(numeric_only=True)
        return {column: round(float(medians[column]), 6) for column in feature_columns}

    return {
        "typical_legitimate": pack(legit),
        "smaller_fraud_profile": pack(smaller),
        "larger_fraud_profile": pack(larger),
    }
