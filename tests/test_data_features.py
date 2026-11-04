import numpy as np
import pandas as pd
import pytest

from fraudlab.constants import TAG_SYNTHETIC, V_COLUMNS
from fraudlab.data import (
    DataLoadError,
    assert_openml_identity,
    load_dataset,
    normalize_frame,
    prepare_openml_frame,
)
from fraudlab.features import DesignMatrix, feature_frame
from fraudlab.synthetic import make_synthetic


def test_synthetic_is_labeled_reproducible_and_separated():
    frame = make_synthetic(20_000, n_fraud=400, random_state=0)
    again = make_synthetic(20_000, n_fraud=400, random_state=0)
    pd.testing.assert_frame_equal(frame, again)
    assert int(frame["Class"].sum()) == 400
    assert set(frame.columns) == {"Time", *V_COLUMNS, "Amount", "Class"}
    assert (frame["Amount"] >= 0).all()
    assert frame.loc[frame["Class"] == 1, "V14"].mean() < -1
    assert frame.loc[frame["Class"] == 0, "V14"].mean() == pytest.approx(0, abs=0.15)
    with pytest.raises(ValueError):
        make_synthetic(10, n_fraud=2)


def test_normalize_accepts_lowercase_and_string_labels_without_time():
    raw = make_synthetic(120, n_fraud=8, random_state=2).drop(columns=["Time"])
    raw = raw.rename(columns={"Amount": "amount", "Class": "class"})
    raw["class"] = raw["class"].astype(str)
    frame = normalize_frame(raw)
    assert "Time" not in frame.columns
    assert frame["Class"].dtype == np.int64 or str(frame["Class"].dtype) == "int64"
    assert set(frame["Class"].unique()).issubset({0, 1})


def test_openml_paths(monkeypatch, tmp_path):
    raw = make_synthetic(300, n_fraud=15, random_state=3)
    duplicated = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)

    def fake_fetch(**kwargs):
        return duplicated, {
            "openml_name": "creditcard",
            "openml_version": "1",
            "openml_data_id": 1597,
            "licence": "Public",
        }

    monkeypatch.setattr("fraudlab.data.fetch_openml_frame", fake_fetch)
    monkeypatch.setattr("fraudlab.data.matches_public_fingerprint", lambda *args, **kwargs: True)
    frame, info = load_dataset(source="openml", usd_per_eur=2.0, data_home=tmp_path, fraud_rate=0.2)
    assert info["kind"] == "openml"
    assert info["duplicates_removed"] == 1
    assert info["usd_per_eur"] == 2.0
    assert frame["Amount"].mean() == pytest.approx(raw["Amount"].mean() * 2)
    assert "ignored" in info["detail"]

    def boom(**kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr("fraudlab.data.fetch_openml_frame", boom)
    fallen, fallback = load_dataset(
        source="auto",
        n_rows=500,
        fraud_rate=0.05,
        random_state=1,
        data_home=tmp_path,
    )
    assert fallback["kind"] == "synthetic"
    assert fallback["chart_tag"] == TAG_SYNTHETIC
    assert "offline" in fallback["detail"]
    assert int(fallen["Class"].sum()) == 25

    monkeypatch.setattr("fraudlab.data.fetch_openml_frame", fake_fetch)
    monkeypatch.setattr("fraudlab.data.matches_public_fingerprint", lambda *args, **kwargs: False)
    with pytest.raises(DataLoadError):
        load_dataset(source="openml", data_home=tmp_path)
    mismatch, mismatch_info = load_dataset(
        source="auto",
        n_rows=400,
        fraud_rate=0.05,
        random_state=2,
        data_home=tmp_path,
    )
    assert mismatch_info["fallback_reason"] == "fingerprint"
    assert "Class" in mismatch.columns

    calls = {"n": 0}

    def count(**kwargs):
        calls["n"] += 1
        raise AssertionError("synthetic must not fetch")

    monkeypatch.setattr("fraudlab.data.fetch_openml_frame", count)
    load_dataset(source="synthetic", n_rows=200, fraud_rate=0.1, random_state=0)
    assert calls["n"] == 0

    prepared, meta = prepare_openml_frame(raw, usd_per_eur=3)
    assert meta["fingerprint_ok"] is False
    assert prepared["Amount"].iloc[0] == pytest.approx(raw["Amount"].iloc[0])


def test_identity_and_bad_schema(monkeypatch, tmp_path):
    assert_openml_identity({"name": "creditcard", "id": 1597})
    with pytest.raises(DataLoadError):
        assert_openml_identity({"name": "iris", "id": "1597"})

    def bad(**kwargs):
        return pd.DataFrame({"a": [1, 2, 3]}), {"openml_name": "creditcard"}

    monkeypatch.setattr("fraudlab.data.fetch_openml_frame", bad)
    _frame, info = load_dataset(source="auto", n_rows=220, fraud_rate=0.1, random_state=4)
    assert info["fallback_reason"] == "download_or_schema"
    with pytest.raises(DataLoadError):
        load_dataset(source="openml")
    with pytest.raises(ValueError):
        load_dataset(source="nope")


def test_design_matrix_scales_train_amount_only():
    frame = make_synthetic(150, n_fraud=12, random_state=5)
    design = DesignMatrix().fit(frame.drop(columns=["Class"]))
    transformed = design.transform(frame.drop(columns=["Class"]))
    assert transformed.shape == (150, 32)
    assert transformed[:, 28].mean() == pytest.approx(0, abs=1e-6)
    no_time = frame.drop(columns=["Time", "Class"])
    narrow = DesignMatrix().fit(no_time)
    assert narrow.transform(no_time).shape == (150, 29)
    with pytest.raises(TypeError):
        feature_frame(np.ones((3, 3)))
    with pytest.raises(ValueError):
        feature_frame(frame.drop(columns=["Amount"]))
    names = design.get_feature_names_out()
    assert names[0] == "V1"
    assert names[-1] == "hour_cos"
