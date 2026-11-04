"""Load the public OpenML credit-card file, or a labeled synthetic fallback.

OpenML data id 1597 is the ULB credit-card extract named ``creditcard``:
284,807 rows, 492 frauds, columns V1–V28, Amount, and Class. This mirror does
not include Time. The loader checks that fingerprint before training. Exact
duplicate rows are removed only after the fingerprint matches, so a random
split cannot place the same transaction on both sides.

If the download fails, the name/id does not match, or the fingerprint does not
match, ``source="auto"`` switches to :func:`fraudlab.synthetic.make_synthetic`
and labels the run as synthetic. ``source="openml"`` raises instead of
silently substituting.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

from fraudlab.constants import (
    DEFAULT_USD_PER_EUR,
    OPENML_DATA_ID,
    OPENML_NAME,
    PUBLIC_FRAUD_RATE,
    PUBLIC_FRAUDS,
    PUBLIC_ROWS,
    TAG_OPENML,
    TAG_SYNTHETIC,
    V_COLUMNS,
)
from fraudlab.synthetic import make_synthetic


class DataLoadError(RuntimeError):
    """The requested table could not be loaded under the stated rules."""


def assert_openml_identity(details: dict) -> None:
    name = str(details.get("name", "")).strip().lower()
    data_id = str(details.get("id", "")).strip()
    if name != OPENML_NAME or data_id != str(OPENML_DATA_ID):
        raise DataLoadError(
            f"Refusing OpenML payload name={name!r} id={data_id!r}; "
            f"expected {OPENML_NAME!r} id {OPENML_DATA_ID}"
        )


def matches_public_fingerprint(
    frame: pd.DataFrame,
    *,
    rows: int = PUBLIC_ROWS,
    frauds: int = PUBLIC_FRAUDS,
) -> bool:
    return len(frame) == rows and int(frame["Class"].sum()) == frauds


def canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    expected = {column.lower(): column for column in ["Time", "Amount", "Class", *V_COLUMNS]}
    mapping = {}
    for column in frame.columns:
        key = str(column).strip().lower()
        mapping[column] = expected.get(key, str(column).strip())
    renamed = frame.rename(columns=mapping)
    if renamed.columns.duplicated().any():
        raise DataLoadError("Duplicate column names after normalization")
    return renamed


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = canonicalize_columns(frame)
    missing = [
        column for column in [*V_COLUMNS, "Amount", "Class"] if column not in renamed.columns
    ]
    if missing:
        raise DataLoadError(f"Unexpected schema, missing {missing}")
    columns = []
    if "Time" in renamed.columns:
        columns.append("Time")
    columns.extend([*V_COLUMNS, "Amount", "Class"])
    out = renamed.loc[:, columns].copy()
    for column in columns:
        if column == "Class":
            out[column] = _as_binary_class(out[column])
        else:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if out["Class"].isna().any() or not set(out["Class"].dropna().unique()).issubset({0, 1}):
        raise DataLoadError("Class must be binary 0/1")
    out["Class"] = out["Class"].astype(int)
    return out


def _as_binary_class(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype(str)
    return pd.to_numeric(series, errors="coerce")


def fetch_openml_frame(*, data_home: Path) -> tuple[pd.DataFrame, dict]:
    data_home.mkdir(parents=True, exist_ok=True)
    bunch = fetch_openml(
        data_id=OPENML_DATA_ID,
        as_frame=True,
        parser="auto",
        data_home=str(data_home),
    )
    assert_openml_identity(dict(bunch.details))
    if bunch.frame is None:
        raise DataLoadError("OpenML response did not include a dataframe")
    details = {
        "openml_name": bunch.details.get("name"),
        "openml_version": str(bunch.details.get("version")),
        "openml_data_id": OPENML_DATA_ID,
        "licence": bunch.details.get("licence"),
    }
    return bunch.frame.copy(), details


def prepare_openml_frame(
    raw: pd.DataFrame,
    *,
    usd_per_eur: float,
) -> tuple[pd.DataFrame, dict]:
    frame = normalize_frame(raw)
    nulls = int(frame.isna().sum().sum())
    fingerprint_ok = nulls == 0 and matches_public_fingerprint(frame)
    info: dict = {
        "fingerprint_ok": fingerprint_ok,
        "fingerprint_rows": PUBLIC_ROWS,
        "fingerprint_frauds": PUBLIC_FRAUDS,
        "nulls": nulls,
        "duplicates_removed": 0,
        "usd_per_eur": float(usd_per_eur),
    }
    if not fingerprint_ok:
        info["rows"] = int(len(frame))
        info["fraud_count"] = int(frame["Class"].sum()) if "Class" in frame.columns else None
        return frame, info
    info["duplicates_removed"] = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().reset_index(drop=True)
    frame["Amount"] = frame["Amount"].astype(float) * float(usd_per_eur)
    info["rows"] = int(len(frame))
    info["fraud_count"] = int(frame["Class"].sum())
    info["fraud_rate"] = float(frame["Class"].mean())
    info["has_time"] = "Time" in frame.columns
    return frame, info


def load_dataset(
    *,
    source: str = "auto",
    n_rows: int = PUBLIC_ROWS,
    fraud_rate: float | None = None,
    random_state: int = 42,
    usd_per_eur: float = DEFAULT_USD_PER_EUR,
    data_home: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    if source not in {"auto", "openml", "synthetic"}:
        raise ValueError("source must be auto, openml, or synthetic")
    if usd_per_eur <= 0:
        raise ValueError("usd_per_eur must be positive")
    if source == "synthetic":
        return _synthetic_result(
            n_rows=n_rows,
            fraud_rate=PUBLIC_FRAUD_RATE if fraud_rate is None else fraud_rate,
            random_state=random_state,
            fallback_reason="requested",
            detail="Synthetic data was requested. OpenML was not contacted.",
        )
    home = Path(data_home) if data_home is not None else Path("data") / "openml_cache"
    try:
        raw, remote = fetch_openml_frame(data_home=home)
        frame, prepared = prepare_openml_frame(raw, usd_per_eur=usd_per_eur)
    except Exception as exc:
        # A failed public load must not train on a half-parsed table.
        if source == "openml":
            raise DataLoadError(str(exc)) from exc
        return _synthetic_result(
            n_rows=n_rows,
            fraud_rate=PUBLIC_FRAUD_RATE if fraud_rate is None else fraud_rate,
            random_state=random_state,
            fallback_reason="download_or_schema",
            detail=f"OpenML load failed ({type(exc).__name__}: {exc}).",
        )
    if not prepared["fingerprint_ok"]:
        reason = (
            f"OpenML fingerprint mismatch or nulls "
            f"(rows={prepared['rows']}, frauds={prepared['fraud_count']}, nulls={prepared['nulls']})."
        )
        if source == "openml":
            raise DataLoadError(reason)
        return _synthetic_result(
            n_rows=n_rows,
            fraud_rate=PUBLIC_FRAUD_RATE if fraud_rate is None else fraud_rate,
            random_state=random_state,
            fallback_reason="fingerprint",
            detail=reason,
        )
    ignored = ""
    if fraud_rate is not None:
        ignored = " The fraud_rate argument was ignored because the public file was accepted."
    info = {
        "kind": "openml",
        "chart_tag": TAG_OPENML,
        "fallback_reason": None,
        "detail": (
            "Public OpenML creditcard extract, verified at "
            f"{PUBLIC_ROWS:,} rows and {PUBLIC_FRAUDS:,} frauds before dropping "
            f"{prepared['duplicates_removed']:,} exact duplicate rows. "
            "Amount was converted from EUR to USD at the planning rate. "
            "Time is not in this OpenML file, so no hour features were used." + ignored
        ),
        "amount_unit": "USD",
        "source_amount_currency": "EUR",
        "columns": [column for column in frame.columns if column != "Class"],
        **remote,
        **prepared,
    }
    return frame, info


def _synthetic_result(
    *,
    n_rows: int,
    fraud_rate: float,
    random_state: int,
    fallback_reason: str,
    detail: str,
) -> tuple[pd.DataFrame, dict]:
    frame = make_synthetic(n_rows, fraud_rate=fraud_rate, random_state=random_state)
    duplicates = int(frame.duplicated().sum())
    if duplicates:
        frame = frame.drop_duplicates().reset_index(drop=True)
    info = {
        "kind": "synthetic",
        "chart_tag": TAG_SYNTHETIC,
        "fallback_reason": fallback_reason,
        "detail": (
            detail
            + " Rows were simulated in USD by fraudlab.synthetic. "
            + "They are not the ULB / OpenML transactions."
        ),
        "openml_data_id": OPENML_DATA_ID,
        "openml_name": OPENML_NAME,
        "fingerprint_ok": False,
        "duplicates_removed": duplicates,
        "rows": int(len(frame)),
        "fraud_count": int(frame["Class"].sum()),
        "fraud_rate": float(frame["Class"].mean()),
        "has_time": "Time" in frame.columns,
        "amount_unit": "USD",
        "source_amount_currency": "USD",
        "usd_per_eur": None,
        "licence": None,
        "columns": [column for column in frame.columns if column != "Class"],
        "nulls": 0,
    }
    return frame, info
