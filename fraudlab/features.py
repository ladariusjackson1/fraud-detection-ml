"""Design matrix shared by every scorer.

V1–V28 pass through. They are already principal components on the public file,
and the synthetic file is built on that same scale. Amount is log1p-scaled
using training statistics only. When Time is present (synthetic fallback only;
OpenML 1597 does not ship it), the matrix adds a scaled time and a 24-hour
sine/cosine pair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from fraudlab.constants import V_COLUMNS


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame with named columns")
    missing = [column for column in [*V_COLUMNS, "Amount"] if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = pd.DataFrame(index=frame.index)
    for column in V_COLUMNS:
        out[column] = pd.to_numeric(frame[column], errors="coerce")
    out["Amount"] = pd.to_numeric(frame["Amount"], errors="coerce").clip(lower=0)
    if "Time" in frame.columns:
        out["Time"] = pd.to_numeric(frame["Time"], errors="coerce")
    if out.isna().any().any():
        out = out.fillna(0.0)
    if "Time" in out.columns:
        hours = np.mod(out["Time"].to_numpy(dtype=float), 86_400) / 3600.0
        out["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    return out


class DesignMatrix(BaseEstimator, TransformerMixin):
    """Train-fit scaler for log amount and, when available, Time."""

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> DesignMatrix:
        del y
        frame = feature_frame(X)
        self.has_time_ = "Time" in frame.columns
        amount = np.log1p(frame[["Amount"]].to_numpy(dtype=float))
        self.scaler_amount_ = StandardScaler().fit(amount)
        if self.has_time_:
            self.scaler_time_ = StandardScaler().fit(frame[["Time"]].to_numpy(dtype=float))
        else:
            self.scaler_time_ = None
        self.n_features_in_ = int(frame.shape[1])
        self.n_features_out_ = 29 + (3 if self.has_time_ else 0)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "scaler_amount_")
        frame = feature_frame(X)
        if self.has_time_ and "Time" not in frame.columns:
            raise ValueError("This model was fit with Time, but Time is missing")
        parts = [
            frame[V_COLUMNS].to_numpy(dtype=float),
            self.scaler_amount_.transform(np.log1p(frame[["Amount"]].to_numpy(dtype=float))),
        ]
        if self.has_time_:
            parts.append(self.scaler_time_.transform(frame[["Time"]].to_numpy(dtype=float)))
            parts.append(frame[["hour_sin", "hour_cos"]].to_numpy(dtype=float))
        return np.hstack(parts)

    def get_feature_names_out(self, input_features: np.ndarray | None = None) -> np.ndarray:
        del input_features
        check_is_fitted(self, "scaler_amount_")
        names = [*V_COLUMNS, "log_amount"]
        if self.has_time_:
            names.extend(["time", "hour_sin", "hour_cos"])
        return np.asarray(names, dtype=object)
