"""Train-only imbalance tools.

Undersampling is applied to the training fold and nowhere else. Validation and
test rows keep the natural fraud rate, which is the rate a reviewer queue would
see.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def random_undersample(
    frame: pd.DataFrame,
    *,
    negative_per_positive: int,
    seed: int,
    label: str = "Class",
) -> pd.DataFrame:
    if negative_per_positive < 1:
        raise ValueError("negative_per_positive must be >= 1")
    data = frame.reset_index(drop=True)
    positives = data[data[label] == 1]
    negatives = data[data[label] == 0]
    if positives.empty or negatives.empty:
        raise ValueError("Undersampling needs both classes")
    n_neg = min(len(negatives), int(negative_per_positive) * len(positives))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(negatives.index.to_numpy(), size=n_neg, replace=False)
    mixed = pd.concat([positives, negatives.loc[chosen]], axis=0)
    return mixed.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def balanced_sample_weight(y_true: np.ndarray) -> np.ndarray:
    """Weights that give the positive class and the negative class equal total mass."""
    y = np.asarray(y_true, dtype=int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Balanced weights need both classes")
    n = len(y)
    weights = np.empty(n, dtype=float)
    weights[y == 1] = n / (2.0 * n_pos)
    weights[y == 0] = n / (2.0 * n_neg)
    return weights
