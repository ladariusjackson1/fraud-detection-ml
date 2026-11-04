import json
from pathlib import Path

from fraudlab.constants import TAG_OPENML, TAG_SYNTHETIC
from fraudlab.plots import FIGURE_FILES
from fraudlab.reporting import money

ROOT = Path(__file__).resolve().parents[1]


def test_readme_quotes_the_public_scorecard():
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    source = metrics["data_source"]
    assert source["kind"] == "openml"
    assert source["chart_tag"] == TAG_OPENML
    assert TAG_OPENML in readme
    assert TAG_SYNTHETIC not in readme
    selected = next(row for row in metrics["models"] if row["key"] == metrics["selected_model"])
    assert money(selected["test_cost"]) in readme
    assert f"{selected['test_average_precision']:.3f}" in readme
    assert money(metrics["approve_all_test_cost"]) in readme
    note = (ROOT / "reports" / "DATA_SOURCE.txt").read_text(encoding="utf-8")
    assert note.startswith("OPENML CREDITCARD")
    for name in FIGURE_FILES:
        figure = ROOT / "reports" / "figures" / name
        assert figure.read_bytes().startswith(b"\x89PNG")
        assert figure.stat().st_size > 1000
        assert name in readme
    assert "https://fraud-detection-ml.streamlit.app" in readme
    assert "Live demo" in readme
    assert "docs/review-desk.gif" in readme
    gif = ROOT / "docs" / "review-desk.gif"
    assert gif.read_bytes()[:6] in {b"GIF87a", b"GIF89a"}
    assert gif.stat().st_size > 10_000
