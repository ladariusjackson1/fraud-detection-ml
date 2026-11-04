"""Model specifications.

Imbalance handling is part of the specification, not a post-hoc tweak:

* logistic regression with ``class_weight='balanced'``
* logistic regression fit after random undersampling, then a threshold search
* random forest with ``class_weight='balanced_subsample'``
* histogram gradient boosting with balanced sample weights
* isolation forest fit only on legitimate training rows
* an amount-only rule: review the ticket when the dollar amount is high enough

The amount rule is the business baseline. It is selected if it has the lowest
validation cost. No scorer is allowed to see validation or test labels at fit
time. Thresholds are chosen later, on validation scores only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fraudlab.constants import NEGATIVE_PER_POSITIVE
from fraudlab.features import DesignMatrix
from fraudlab.sampling import balanced_sample_weight, random_undersample


@dataclass(frozen=True)
class Spec:
    key: str
    display_name: str
    strategy: str
    family: str
    score_kind: str
    class_weight: bool = False
    undersample: bool = False
    sample_weight: bool = False


SPECS: tuple[Spec, ...] = (
    Spec(
        key="logistic_balanced",
        display_name="Logistic regression",
        strategy="Class weights (balanced) on every training row",
        family="logistic",
        score_kind="probability",
        class_weight=True,
    ),
    Spec(
        key="logistic_undersampled",
        display_name="Logistic regression, undersampled",
        strategy=f"Random undersample to {NEGATIVE_PER_POSITIVE}:1, then unweighted logistic regression",
        family="logistic",
        score_kind="probability",
        undersample=True,
    ),
    Spec(
        key="random_forest",
        display_name="Random forest",
        strategy="balanced_subsample class weights; 30,000-row bootstrap when training is larger",
        family="forest",
        score_kind="probability",
    ),
    Spec(
        key="gradient_boosting",
        display_name="Gradient boosting",
        strategy="HistGradientBoosting with balanced sample weights",
        family="boosting",
        score_kind="probability",
        sample_weight=True,
    ),
    Spec(
        key="isolation_forest",
        display_name="Isolation forest",
        strategy="Unsupervised; fit on legitimate training rows only, capped at 120,000",
        family="isolation",
        score_kind="anomaly",
    ),
    Spec(
        key="amount_rule",
        display_name="Amount-only rule",
        strategy="No model: review a ticket when its USD amount is at least the cutoff",
        family="amount",
        score_kind="amount_usd",
    ),
)


@dataclass
class FitResult:
    model: Pipeline | None
    fit_rows: int
    fit_detail: str
    hyperparameters: dict


def fit_spec(spec: Spec, train_df: pd.DataFrame, *, fast: bool, seed: int) -> FitResult:
    if spec.family == "amount":
        return FitResult(
            model=None,
            fit_rows=0,
            fit_detail=spec.strategy,
            hyperparameters={"score": "amount_usd"},
        )
    frame = train_df
    detail = spec.strategy
    if spec.undersample:
        frame = random_undersample(
            frame,
            negative_per_positive=NEGATIVE_PER_POSITIVE,
            seed=seed,
        )
        detail = f"{spec.strategy}. Fit rows: {len(frame):,}."
    x_train = frame.drop(columns=["Class"])
    y_train = frame["Class"].to_numpy(dtype=int)
    if spec.family == "isolation":
        return _fit_isolation(x_train, y_train, fast=fast, seed=seed, detail=detail)
    estimator, params = _supervised_estimator(spec, fast=fast, seed=seed, n_rows=len(x_train))
    pipeline = Pipeline([("design", DesignMatrix()), ("model", estimator)])
    if spec.sample_weight:
        weights = balanced_sample_weight(y_train)
        pipeline.fit(x_train, y_train, model__sample_weight=weights)
    else:
        pipeline.fit(x_train, y_train)
    return FitResult(pipeline, len(x_train), detail, params)


def _fit_isolation(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    fast: bool,
    seed: int,
    detail: str,
) -> FitResult:
    legit = x_train.iloc[np.flatnonzero(y_train == 0)]
    available = len(legit)
    cap = 800 if fast else 120_000
    used = legit
    if len(used) > cap:
        used = used.sample(n=cap, random_state=seed)
    estimator, params = _isolation_estimator(fast=fast, seed=seed, n_rows=len(used))
    pipeline = Pipeline([("design", DesignMatrix()), ("model", estimator)])
    pipeline.fit(used)
    params["legit_rows_available"] = int(available)
    params["legit_rows_fit"] = int(len(used))
    text = f"{detail} Used {len(used):,} of {available:,} legitimate training rows."
    return FitResult(pipeline, int(len(used)), text, params)


def _supervised_estimator(
    spec: Spec,
    *,
    fast: bool,
    seed: int,
    n_rows: int,
) -> tuple[LogisticRegression | RandomForestClassifier | HistGradientBoostingClassifier, dict]:
    if spec.family == "logistic":
        params: dict = {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 300 if fast else 1000,
            "class_weight": "balanced" if spec.class_weight else None,
            "random_state": seed,
        }
        return LogisticRegression(**params), params
    if spec.family == "forest":
        max_samples = None if fast or n_rows <= 30_000 else 30_000
        params = {
            "n_estimators": 20 if fast else 100,
            "max_depth": 8 if fast else 12,
            "min_samples_leaf": 2 if fast else 25,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "max_samples": max_samples,
            "n_jobs": 1 if fast else -1,
            "random_state": seed,
        }
        return RandomForestClassifier(**params), params
    if spec.family == "boosting":
        params = {
            "learning_rate": 0.08,
            "max_iter": 25 if fast else 200,
            "max_leaf_nodes": 15 if fast else 31,
            "min_samples_leaf": 5 if fast else 20,
            "l2_regularization": 1.0,
            "random_state": seed,
        }
        return HistGradientBoostingClassifier(**params), params
    raise ValueError(f"Unknown supervised family: {spec.family}")


def _isolation_estimator(*, fast: bool, seed: int, n_rows: int) -> tuple[IsolationForest, dict]:
    configured = 128 if fast else 8_192
    params = {
        "n_estimators": 20 if fast else 200,
        "max_samples": int(min(configured, n_rows)),
        "contamination": "auto",
        "n_jobs": 1 if fast else -1,
        "random_state": seed,
    }
    return IsolationForest(**params), params


def positive_scores(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Higher means more suspicious. Isolation forest signs are flipped."""
    estimator = model.named_steps["model"]
    if isinstance(estimator, IsolationForest):
        scores = -model.decision_function(frame)
    else:
        classes = [int(value) for value in estimator.classes_]
        if 1 not in classes:
            raise RuntimeError("Classifier did not observe the fraud class during fit")
        scores = model.predict_proba(frame)[:, classes.index(1)]
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise RuntimeError("Scorer returned a non-finite score vector")
    return scores
