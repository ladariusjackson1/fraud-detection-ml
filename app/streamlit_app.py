"""Review desk: tune the queue and score a transaction.

Run from the repository root after training:

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import FuncFormatter

from fraudlab.app_cache import (
    SLIDER_MAX,
    best_grid_queue,
    grid_costs,
    grid_decision,
    saved_decision,
    startup_note,
)
from fraudlab.constants import V_COLUMNS
from fraudlab.costs import CostConfig, realized_unit_cost
from fraudlab.dashboard import (
    comparison_frame,
    decode_saved_threshold,
    format_threshold,
    frame_from_mapping,
    load_desk,
    score_frame,
    single_ticket_costs,
    source_banner,
)
from fraudlab.explain import COMPONENT_NOTE, explain_ticket
from fraudlab.plots import apply_style, focus_queue
from fraudlab.reporting import money, percent

ROOT = Path(__file__).resolve().parents[1]
PROFILE_LABELS = {
    "typical_legitimate": "Typical legitimate (training median)",
    "smaller_fraud_profile": "Smaller-fraud profile (training median)",
    "larger_fraud_profile": "Larger-fraud profile (training median)",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; max-width: 1180px;}
        div[data-testid="stMetric"] {
            background: #f4f1eb;
            padding: 0.7rem 0.85rem;
            border-radius: 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def cached_assets():
    return load_desk(ROOT)


def main() -> None:
    st.set_page_config(page_title="Fraud review desk", layout="wide")
    inject_css()
    st.title("Fraud review desk")
    st.caption("Business analytics case study · Ladarius Jackson")
    try:
        assets = cached_assets()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.write("Train the scorecard from the repository root, then reload this page.")
        st.code("python -m fraudlab --source auto", language="bash")
        return

    level, text = source_banner(assets.metadata["data_source"])
    getattr(st, level)(text)
    st.caption(
        startup_note(
            validation_rows=int(assets.curve.n),
            demo_rows=int(len(assets.demo)),
            grid_points=int(len(assets.curve.reviews_per_10k)),
        )
    )
    _header(assets)
    cutoff, score, compare = st.tabs(["Queue cutoff", "Score a transaction", "Model comparison"])
    with cutoff:
        _cutoff_tab(assets)
    with score:
        _score_tab(assets)
    with compare:
        _compare_tab(assets)


def _header(assets) -> None:
    summary = assets.metadata["test_summary"]
    selected = next(
        row for row in assets.metrics["models"] if row["key"] == assets.metadata["model_key"]
    )
    columns = st.columns(4)
    columns[0].metric("Test PR-AUC", f"{summary['average_precision']:.3f}")
    precision = "n/a" if summary["precision"] is None else f"{summary['precision']:.3f}"
    columns[1].metric("Test precision", precision)
    columns[2].metric("Test recall", f"{summary['recall']:.3f}")
    columns[3].metric("Savings vs approve-all", money(summary["savings_vs_approve_all"]))
    st.caption(
        f"Published operating point for {selected['display_name']}. "
        "The cutoff was chosen on validation cost. These four numbers are the untouched test set. "
        f"Score type: {selected['score_kind'].replace('_', ' ')}."
    )


def _cutoff_tab(assets) -> None:
    st.subheader("What the queue costs")
    st.write(
        "Class-weighted scores are ranking tools, not calibrated probabilities. "
        "The control that matches the review desk is how many tickets you pull per 10,000 transactions. "
        "Moving the slider replays a cached grid scored on the validation book before this app was packaged. "
        "It does not rewrite the published test result, and it does not load the public file."
    )
    base = CostConfig.from_dict(assets.metadata["cost_config"])
    inputs = st.columns(3)
    review = inputs[0].number_input(
        "Review cost (USD)",
        min_value=0.0,
        value=float(base.review_cost),
        step=1.0,
        help="Analyst time. At $8 and 4 minutes, that is $120 an hour.",
    )
    decline = inputs[1].number_input(
        "False-decline cost (USD)",
        min_value=0.0,
        value=float(base.false_decline_cost),
        step=1.0,
        help="Goodwill and lost margin when a legitimate ticket is pulled.",
    )
    fee = inputs[2].number_input(
        "Chargeback fee (USD)",
        min_value=0.0,
        value=float(base.chargeback_fee),
        step=1.0,
        help="Extra operations cost on an approved fraud, on top of the ticket amount.",
    )
    cost = CostConfig(review_cost=review, false_decline_cost=decline, chargeback_fee=fee)
    recommended_per_10k, _recommended_threshold, recommended = best_grid_queue(assets.curve, cost)
    if "per_10k" not in st.session_state:
        st.session_state.per_10k = int(recommended_per_10k)
    if st.button("Use cost-minimizing queue"):
        st.session_state.per_10k = int(recommended_per_10k)
    per_10k = st.slider(
        "Reviews per 10,000 transactions",
        min_value=0,
        max_value=SLIDER_MAX,
        step=1,
        key="per_10k",
    )
    threshold, decision = grid_decision(assets.curve, float(per_10k), cost)
    kind = assets.metadata["score_kind"]
    st.write(
        f"Score cutoff: **{format_threshold(threshold, kind)}**. "
        f"Realized validation queue: **{decision.reviews_per_10k:.1f}** reviews per 10,000 "
        f"({decision.analyst_hours:.1f} analyst hours on this validation slice)."
    )
    cards = st.columns(4)
    precision = "n/a" if not np.isfinite(decision.precision) else f"{decision.precision:.3f}"
    cards[0].metric("Validation cost", money(decision.total_cost))
    cards[1].metric("Validation precision", precision)
    cards[2].metric("Validation recall", f"{decision.recall:.3f}")
    saved = saved_decision(assets.curve, cost)
    cards[3].metric("Cost-minimizing validation cost", money(recommended.total_cost))
    st.caption(
        "Published test cost at the frozen cutoff, under the training cost assumptions: "
        f"{money(assets.metadata['test_summary']['cost'])}. "
        f"Saved cutoff: {format_threshold(decode_saved_threshold(assets.metadata), kind)}. "
        "Replayed validation cost at the saved cutoff with the costs currently on screen: "
        f"{money(saved.total_cost)}."
    )
    _plot_curve(assets, cost, decision, int(per_10k))
    if recommended_per_10k >= SLIDER_MAX:
        st.info(
            "The cheapest queue on this grid is at the slider maximum. "
            "A cost assumption that wants a longer queue needs a new training run to see past "
            f"{SLIDER_MAX:,} reviews per 10,000."
        )


def _plot_curve(assets, cost: CostConfig, decision, per_10k: int) -> None:
    costs = grid_costs(assets.curve, cost)
    reviews = assets.curve.reviews_per_10k.astype(float)
    apply_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    shown_reviews, shown_costs = focus_queue(reviews, costs, float(per_10k))
    order = np.argsort(shown_reviews)
    ax.plot(shown_reviews[order], shown_costs[order], color="#1F4E79", lw=2)
    ax.scatter([per_10k], [decision.total_cost], color="#B42318", s=42, zorder=3)
    ax.set_xlim(left=0)
    ceiling = float(np.nanmax(shown_costs)) if len(shown_costs) else 1.0
    ax.set_ylim(0, ceiling * 1.08 if ceiling > 0 else 1)
    ax.set_xlabel("Reviews per 10,000 validation transactions")
    ax.set_ylabel("Validation cost (USD)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"${value:,.0f}"))
    ax.set_title("Validation cost near the current queue", loc="left")
    ax.grid(color="#E6E2DC", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _score_tab(assets) -> None:
    st.subheader("Score one ticket")
    st.write(
        "V1–V28 are principal components. They do not have merchant names, ages, or countries. "
        "Profiles are medians from the training fold, not people."
    )
    mode = st.radio("Source", ["Held-out example", "Training profile"], horizontal=True)
    if mode == "Held-out example":
        frame, label, amount = _demo_choice(assets)
    else:
        frame, label, amount = _profile_choice(assets)
    threshold = _active_threshold(assets)
    scores = score_frame(assets.model, frame, assets.metadata["model_key"])
    score = float(scores[0])
    flagged = bool(np.isfinite(threshold) and score >= threshold)
    if flagged:
        st.error(
            f"Review this ticket. Score {score:.4f}. Cutoff {format_threshold(threshold, assets.metadata['score_kind'])}."
        )
    else:
        st.success(
            f"Approve this ticket. Score {score:.4f}. Cutoff {format_threshold(threshold, assets.metadata['score_kind'])}."
        )
    _explain(assets, frame)
    costs = single_ticket_costs(amount, CostConfig.from_dict(assets.metadata["cost_config"]))
    grid = st.columns(4)
    grid[0].metric(
        "Cost if approved and legitimate", money(costs["approve_if_legitimate"], cents=True)
    )
    grid[1].metric("Cost if approved and fraud", money(costs["approve_if_fraud"], cents=True))
    grid[2].metric(
        "Cost if reviewed and legitimate", money(costs["review_if_legitimate"], cents=True)
    )
    grid[3].metric("Cost if reviewed and fraud", money(costs["review_if_fraud"], cents=True))
    st.caption(
        "These four numbers do not use the score as a probability. The label is still unknown."
    )
    if label is not None and st.checkbox("Reveal held-out label"):
        name = "Fraud" if label == 1 else "Legitimate"
        realized = realized_unit_cost(
            label, flagged, amount, CostConfig.from_dict(assets.metadata["cost_config"])
        )
        st.write(
            f"Held-out label: **{name}**. Realized cost of this decision: **{money(realized, cents=True)}**."
        )


def _active_threshold(assets) -> float:
    """Use the queue slider when the cutoff tab has set it; otherwise the saved cutoff."""
    if "per_10k" in st.session_state:
        threshold, _decision = grid_decision(
            assets.curve, float(st.session_state.per_10k), CostConfig()
        )
        return threshold
    return decode_saved_threshold(assets.metadata)


def _explain(assets, frame: pd.DataFrame) -> None:
    reference = assets.scenarios.get("typical_legitimate")
    if reference is None:
        st.info("No typical legitimate profile was saved, so this ticket is not compared.")
        return
    explanation = explain_ticket(
        assets.model,
        frame,
        reference,
        model_key=assets.metadata["model_key"],
        feature_columns=list(assets.metadata["feature_columns"]),
        score_kind=assets.metadata["score_kind"],
    )
    st.subheader("What pushed the score")
    st.write(explanation.summary)
    for sentence in explanation.sentences:
        st.markdown(f"- {sentence}")
    st.caption(COMPONENT_NOTE)


def _demo_choice(assets) -> tuple[pd.DataFrame, int, float]:
    demo = assets.demo
    choice = st.selectbox("Held-out example", demo["demo_id"].tolist())
    row = demo.loc[demo["demo_id"] == choice].iloc[0]
    columns = assets.metadata["feature_columns"]
    frame = pd.DataFrame([{column: float(row[column]) for column in columns}])
    st.write(f"Amount on the ticket: **{money(float(row['Amount']), cents=True)}**.")
    return frame, int(row["Class"]), float(row["Amount"])


def _profile_choice(assets) -> tuple[pd.DataFrame, None, float]:
    names = [name for name in PROFILE_LABELS if name in assets.scenarios]
    choice = st.selectbox("Profile", names, format_func=lambda key: PROFILE_LABELS[key])
    profile = dict(assets.scenarios[choice])
    columns = assets.metadata["feature_columns"]
    amount = st.number_input(
        "Amount (USD)", min_value=0.0, value=float(profile["Amount"]), step=1.0
    )
    profile["Amount"] = amount
    if "Time" in columns:
        profile["Time"] = st.number_input(
            "Seconds from the start of the window",
            min_value=0.0,
            value=float(profile.get("Time", 0.0)),
            step=60.0,
        )
    if st.checkbox("Edit PCA components"):
        grid = st.columns(4)
        for index, column in enumerate(V_COLUMNS):
            profile[column] = grid[index % 4].number_input(
                column,
                value=float(profile[column]),
                key=f"{choice}-{column}",
                step=0.1,
                format="%.3f",
            )
    return frame_from_mapping(profile, columns), None, float(amount)


def _compare_tab(assets) -> None:
    st.subheader("Held-out scorecard")
    st.dataframe(comparison_frame(assets.metrics), hide_index=True, width="stretch")
    st.caption(
        "Selection used validation cost, then validation PR-AUC. "
        f"Test-cost leader: `{assets.metrics['test_cost_leader']}`."
    )
    figures = ROOT / "reports" / "figures"
    captions = {
        "01_class_balance.png": "Legitimate volume dwarfs fraud. Accuracy is the wrong headline.",
        "02_amount_by_class.png": "Ticket size overlaps. A large-ticket rule is a weak net.",
        "03_precision_recall.png": "Test precision–recall. The dashed line is the fraud rate.",
        "04_cost_vs_threshold.png": "Validation dollars against queue size for the chosen model.",
        "05_model_comparison.png": "Test PR-AUC and savings. The dark bar is the validation choice.",
        "06_confusion_matrix.png": "Test decisions at the frozen cutoff.",
        "07_score_distributions.png": "Test scores for legitimate tickets and frauds.",
    }
    for name, caption in captions.items():
        path = figures / name
        if path.exists():
            st.image(str(path), caption=caption)
    approve = assets.metrics["approve_all_test_cost"]
    st.write(
        f"Approving every test transaction costs {money(approve)}. "
        f"Flag rate at the published cutoff: {percent(assets.metadata['test_summary']['flag_rate'])}."
    )


if __name__ == "__main__":
    main()
