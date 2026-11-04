"""Synthetic card transactions. This is not the OpenML / ULB file.

The generator is an original teaching stand-in used only when the public
download is unavailable or was not requested. It copies the column layout of
the public extract (Time, V1–V28, Amount, Class) and the published fraud rate.
It does not copy public rows, the real covariance, or the real principal
components.

Legitimate components are independent noise. Fraud shifts only a handful of
components, which is the qualitative pattern of the public PCA file: a few
axes carry separation and the rest do not. Amounts mix small card-testing
frauds with a heavier tail, so a "review the large tickets" rule is not given
an easy win. Time covers a simulated 48-hour window; that column is not in
OpenML data id 1597, and it exists here only on the synthetic path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudlab.constants import PUBLIC_FRAUD_RATE, V_COLUMNS

# Invented shifts (mean, std) on 1-based component indexes. Not estimated from OpenML.
FRAUD_COMPONENT_SHIFTS: dict[int, tuple[float, float]] = {
    1: (-1.6, 1.35),
    2: (1.8, 1.35),
    3: (-2.2, 1.40),
    4: (1.5, 1.30),
    10: (-1.4, 1.35),
    11: (1.3, 1.25),
    12: (-1.7, 1.35),
    14: (-2.4, 1.45),
    17: (-1.5, 1.30),
}
CARD_TEST_FRACTION = 0.55


def make_synthetic(
    n_rows: int,
    *,
    fraud_rate: float = PUBLIC_FRAUD_RATE,
    n_fraud: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    if n_rows < 20:
        raise ValueError("n_rows must be at least 20")
    if n_fraud is None:
        if not 0 < fraud_rate < 1:
            raise ValueError("fraud_rate must be between 0 and 1")
        n_fraud = int(round(n_rows * fraud_rate))
    n_fraud = int(n_fraud)
    if n_fraud < 1 or n_fraud >= n_rows:
        raise ValueError("Need at least one fraud row and one legitimate row")
    n_legit = n_rows - n_fraud
    rng = np.random.default_rng(random_state)

    legit_scale = np.linspace(0.85, 1.15, len(V_COLUMNS))
    v_legit = rng.normal(0.0, 1.0, size=(n_legit, len(V_COLUMNS))) * legit_scale
    v_fraud = rng.normal(0.0, 1.0, size=(n_fraud, len(V_COLUMNS))) * legit_scale
    for component, (mean, std) in FRAUD_COMPONENT_SHIFTS.items():
        v_fraud[:, component - 1] = rng.normal(mean, std, size=n_fraud)

    amount_legit = np.clip(rng.lognormal(2.9, 1.05, n_legit), 0, 20_000)
    card_test = rng.random(n_fraud) < CARD_TEST_FRACTION
    amount_fraud = np.empty(n_fraud, dtype=float)
    n_card = int(card_test.sum())
    n_large = n_fraud - n_card
    amount_fraud[card_test] = np.clip(rng.lognormal(1.05, 0.40, n_card), 0, 80)
    if n_large:
        amount_fraud[~card_test] = np.clip(rng.lognormal(4.4, 0.75, n_large), 0, 4_000)

    legit_hour = np.mod(rng.normal(14.5, 3.5, n_legit), 24)
    fraud_hour = np.empty(n_fraud, dtype=float)
    night = rng.random(n_fraud) < 0.65
    fraud_hour[night] = np.mod(rng.normal(3.0, 2.0, int(night.sum())), 24)
    fraud_hour[~night] = rng.uniform(0, 24, int((~night).sum()))
    time_legit = (
        rng.integers(0, 2, n_legit) * 86_400 + legit_hour * 3600 + rng.uniform(0, 3600, n_legit)
    )
    time_fraud = (
        rng.integers(0, 2, n_fraud) * 86_400 + fraud_hour * 3600 + rng.uniform(0, 3600, n_fraud)
    )

    columns: dict[str, np.ndarray] = {
        "Time": np.concatenate([time_legit, time_fraud]),
        "Amount": np.round(np.concatenate([amount_legit, amount_fraud]), 2),
        "Class": np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)]),
    }
    components = np.vstack([v_legit, v_fraud])
    for index, name in enumerate(V_COLUMNS):
        columns[name] = components[:, index]
    frame = pd.DataFrame(columns)
    ordered = ["Time", *V_COLUMNS, "Amount", "Class"]
    frame = frame[ordered]
    return frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
