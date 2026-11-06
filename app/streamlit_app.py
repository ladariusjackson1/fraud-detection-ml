"""Review desk: compare models, set an alert budget, replay a stream, score a CSV.

Run from the repository root after training:

    streamlit run streamlit_app.py
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

from fraudlab.alert_budget import (
    budget_series,
    evaluate_budget,
    planning_daily_volume,
    published_alert_count,
    queue_precision_recall,
)
from fraudlab.app_cache import SLIDER_MAX, best_grid_queue, startup_note
from fraudlab.batch import score_batch, template_from_demo, to_csv_bytes
from fraudlab.constants import V_COLUMNS
from fraudlab.costs import CostConfig, realized_unit_cost
from fraudlab.dashboard import (
    comparison_frame,
    decode_saved_threshold,
    format_threshold,
    frame_from_mapping,
    headline_comparison,
    load_desk,
    overview_blurb,
    score_frame,
    single_ticket_costs,
    source_banner,
)
from fraudlab.explain import COMPONENT_NOTE, explain_ticket
from fraudlab.monitoring import batch_psi, profile_table
from fraudlab.plots import apply_style
from fraudlab.reporting import format_ci, money, percent
from fraudlab.stream import replay_tally, score_demo_row

ROOT = Path(__file__).resolve().parents[1]
PROFILE_LABELS = {
    "typical_legitimate": "Typical legitimate (training median)",
    "smaller_fraud_profile": "Smaller-fraud profile (training median)",
    "larger_fraud_profile": "Larger-fraud profile (training median)",
}
INK = "#1F4E79"
MARK = "#B42318"


def plain(text: str) -> str:
    """Streamlit markdown treats ``$...$`` as math. Dollar amounts have to be escaped."""
    return text.replace("$", r"\$")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.15rem; padding-bottom: 2.5rem; max-width: 1180px;}
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #E4DFD6;
            border-top: 3px solid #1F4E79;
            padding: 0.72rem 0.85rem 0.55rem;
            border-radius: 0.65rem;
        }
        div[data-testid="stMetric"] label {color: #627D98;}
        .desk-kicker {
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #627D98;
            font-size: 0.75rem;
            margin-bottom: 0.15rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def cached_assets():
    return load_desk(ROOT)


@st.cache_data
def cached_reference_scores(path: str) -> np.ndarray:
    return np.load(path)["score"].astype(float)


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
    _seed_state(assets)
    st.caption(
        startup_note(
            validation_rows=int(assets.curve.n),
            demo_rows=int(len(assets.demo)),
            grid_points=int(len(assets.curve.reviews_per_10k)),
        )
    )
    overview, budget, stream, score, batch, models, monitor = st.tabs(
        [
            "Overview",
            "Alert budget",
            "Live stream",
            "Score one",
            "Batch CSV",
            "Models",
            "Monitoring",
        ]
    )
    with overview:
        _overview_tab(assets)
    with budget:
        _budget_tab(assets)
    with stream:
        _stream_tab(assets)
    with score:
        _score_tab(assets)
    with batch:
        _batch_tab(assets)
    with models:
        _models_tab(assets)
    with monitor:
        _monitor_tab(assets)


def _seed_state(assets) -> None:
    base = assets.metadata["cost_config"]
    st.session_state.setdefault("desk_review", float(base["review_cost"]))
    st.session_state.setdefault("desk_decline", float(base["false_decline_cost"]))
    st.session_state.setdefault("desk_fee", float(base["chargeback_fee"]))
    daily = planning_daily_volume(int(assets.metrics["data_source"]["rows"]))
    st.session_state.setdefault("tx_per_day", float(daily))
    if "alerts_per_day" not in st.session_state:
        st.session_state.alerts_per_day = published_alert_count(
            assets.curve, float(st.session_state.tx_per_day), _cost_from_state()
        )
    st.session_state.setdefault("use_budget_cutoff", False)
    st.session_state.setdefault("stream_pos", 1)
    st.session_state.setdefault("stream_seen", [])
    st.session_state.setdefault("explain_cache", {})


def _cost_from_state() -> CostConfig:
    return CostConfig(
        review_cost=float(st.session_state["desk_review"]),
        false_decline_cost=float(st.session_state["desk_decline"]),
        chargeback_fee=float(st.session_state["desk_fee"]),
    )


def _saved_threshold(assets) -> float:
    return decode_saved_threshold(assets.metadata)


def _budget_threshold(assets) -> float:
    daily = float(st.session_state["tx_per_day"])
    alerts = int(st.session_state["alerts_per_day"])
    return evaluate_budget(assets.curve, alerts, daily, _cost_from_state()).threshold


def _active_threshold(assets) -> float:
    if st.session_state.get("use_budget_cutoff"):
        return _budget_threshold(assets)
    return _saved_threshold(assets)


def _selected(assets) -> dict:
    return next(
        row for row in assets.metrics["models"] if row["key"] == assets.metadata["model_key"]
    )


def _overview_tab(assets) -> None:
    st.markdown(
        '<p class="desk-kicker">Held-out test set · cutoff frozen on validation</p>',
        unsafe_allow_html=True,
    )
    selected = _selected(assets)
    summary = assets.metadata["test_summary"]
    caught = int(selected["test_true_positives"])
    missed = int(selected["test_false_negatives"])
    cards = st.columns(5)
    cards[0].metric(
        "Test PR-AUC",
        f"{summary['average_precision']:.3f}",
        help=format_ci(summary["average_precision"], summary["average_precision_ci95"]),
    )
    cards[1].metric("Test cost", money(summary["cost"]))
    cards[2].metric("Savings vs approve-all", money(summary["savings_vs_approve_all"]))
    cards[3].metric("Frauds caught", f"{caught} of {caught + missed}")
    cards[4].metric("Reviews / 10k", f"{summary['reviews_per_10k']:.1f}")
    st.markdown(plain(overview_blurb(assets.metrics)))
    st.caption(
        plain(
            "PR-AUC is average precision. The interval resamples the test rows and does not retune "
            f"the model. Test cost interval: {money(summary['cost_ci95'][0])}–{money(summary['cost_ci95'][1])}. "
            "The alert-budget tab scales the validation book to a planning day. It does not replace these numbers."
        )
    )
    st.subheader("Class-weighted logistic, random forest, gradient boosting")
    st.dataframe(headline_comparison(assets.metrics), hide_index=True, width="stretch")
    st.caption(
        "Validation cost selected the model. Test PR-AUC is the ranking metric. "
        "The full scorecard, including the model that was not selected, is on the Models tab."
    )


def _budget_tab(assets) -> None:
    st.subheader("If analysts can review N alerts a day")
    st.write(
        "The slider is a staffing cap. The desk snaps it to the cached validation queue "
        "and multiplies those rates by the planning daily volume. "
        "The public extract has no timestamp. The default volume is the modeling table divided by two, "
        "because the source study covers two days."
    )
    inputs = st.columns(3)
    inputs[0].number_input("Review cost (USD)", min_value=0.0, step=1.0, key="desk_review")
    inputs[1].number_input("False-decline cost (USD)", min_value=0.0, step=1.0, key="desk_decline")
    inputs[2].number_input("Chargeback fee (USD)", min_value=0.0, step=1.0, key="desk_fee")
    cost = _cost_from_state()
    daily_default = planning_daily_volume(int(assets.metrics["data_source"]["rows"]))
    st.number_input(
        "Transactions per day",
        min_value=1.0,
        step=100.0,
        key="tx_per_day",
        help=f"Default is the modeling table / 2 = {daily_default:,.1f}.",
    )
    daily = float(st.session_state.tx_per_day)
    max_alerts = max(1, int(np.floor(SLIDER_MAX / 10_000.0 * daily)))
    if int(st.session_state.alerts_per_day) > max_alerts:
        st.session_state.alerts_per_day = max_alerts
    jumps = st.columns(2)
    if jumps[0].button("Use the published queue"):
        st.session_state.alerts_per_day = min(
            max_alerts, published_alert_count(assets.curve, daily, cost)
        )
        st.rerun()
    if jumps[1].button("Use the cost-minimizing queue"):
        _per_10k, _threshold, decision = best_grid_queue(assets.curve, cost)
        st.session_state.alerts_per_day = min(max_alerts, int(round(decision.flag_rate * daily)))
        st.rerun()
    alerts = st.slider(
        "Alerts analysts can review per day",
        min_value=0,
        max_value=max_alerts,
        step=1,
        key="alerts_per_day",
    )
    budget = evaluate_budget(assets.curve, int(alerts), daily, cost)
    if budget.clamped:
        st.info(
            f"The cached grid stops at {SLIDER_MAX:,} reviews per 10,000. "
            "This request was evaluated at that cap."
        )
    precision = (
        "n/a"
        if not np.isfinite(budget.validation_precision)
        else f"{budget.validation_precision:.3f}"
    )
    cards = st.columns(4)
    cards[0].metric("Frauds caught / day", f"{budget.expected_caught:.1f}")
    cards[1].metric("Frauds missed / day", f"{budget.expected_missed:.1f}")
    cards[2].metric("Expected loss / day", money(budget.expected_cost))
    cards[3].metric("Precision at this queue", precision)
    st.markdown(
        plain(
            f"Requested **{budget.alerts_requested:,}** slots. "
            f"The grid realizes about **{budget.expected_alerts:.1f}** alerts a day "
            f"({budget.reviews_per_10k_grid} per 10,000), "
            f"cutoff **{format_threshold(budget.threshold, assets.metadata['score_kind'])}**, "
            f"recall **{budget.validation_recall:.3f}**, "
            f"about **{budget.analyst_hours:.1f}** analyst hours."
        )
    )
    st.caption(budget.basis)
    pieces = st.columns(4)
    pieces[0].metric("Missed principal", money(budget.expected_missed_principal))
    pieces[1].metric("Chargeback fees", money(budget.expected_chargeback_fees))
    pieces[2].metric("Analyst time", money(budget.expected_review_spend))
    pieces[3].metric("False-decline cost", money(budget.expected_false_decline_spend))
    series = budget_series(assets.curve, daily, cost)
    _plot_budget(series, budget)
    _plot_pr(assets, budget)
    st.caption(
        "Both charts use the validation book. The test-set precision-recall curve is on the Models tab. "
        "Score one ticket and the live stream keep the published cutoff unless you opt in below."
    )
    st.checkbox(
        "Use this alert-budget cutoff when scoring tickets",
        key="use_budget_cutoff",
        help="Off means the published validation cutoff. On means the queue currently on this slider.",
    )


def _plot_budget(series: pd.DataFrame, budget) -> None:
    apply_style()
    cap = max(400.0, float(budget.alerts_requested) * 3.0)
    shown = series[series["alerts_per_day"] <= cap]
    if shown.empty:
        shown = series
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.15))
    axes[0].plot(shown["alerts_per_day"], shown["expected_caught"], color=INK, lw=2)
    axes[0].scatter([budget.expected_alerts], [budget.expected_caught], color=MARK, s=42, zorder=3)
    axes[0].set_xlabel("Alerts a day")
    axes[0].set_ylabel("Expected frauds caught")
    axes[0].set_title("Frauds caught", loc="left")
    axes[1].plot(shown["alerts_per_day"], shown["expected_cost"], color=INK, lw=2)
    axes[1].scatter([budget.expected_alerts], [budget.expected_cost], color=MARK, s=42, zorder=3)
    axes[1].set_xlabel("Alerts a day")
    axes[1].set_ylabel("Expected loss (USD)")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"${value:,.0f}"))
    axes[1].set_title("Loss", loc="left")
    for ax in axes:
        ax.set_xlim(left=0)
        ax.grid(color="#E6E2DC", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _plot_pr(assets, budget) -> None:
    recall, precision = queue_precision_recall(assets.curve)
    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.plot(recall, precision, color=INK, lw=2, label="Validation queue")
    if np.isfinite(budget.validation_precision):
        ax.scatter(
            [budget.validation_recall],
            [budget.validation_precision],
            color=MARK,
            s=42,
            zorder=3,
            label="This budget",
        )
        ax.legend(loc="upper right")
    prevalence = float(assets.curve.true_positives[-1] + assets.curve.false_negatives[-1]) / float(
        assets.curve.n
    )
    ax.axhline(prevalence, color="#627D98", ls="--", lw=1, label=None)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Validation precision–recall, selected model", loc="left")
    ax.grid(color="#E6E2DC", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _stream_tab(assets) -> None:
    st.subheader("Held-out stream")
    st.write(
        "Each step scores one ticket from the 65-row holdout sample with the saved model, "
        "then says which fields pushed the score. This is a replay, not a processor feed. "
        "The running tally covers only the tickets you have stepped through."
    )
    n = len(assets.demo)
    controls = st.columns(2)
    if controls[0].button("Previous ticket", disabled=int(st.session_state.stream_pos) <= 1):
        st.session_state.stream_pos -= 1
        st.rerun()
    if controls[1].button("Next ticket", disabled=int(st.session_state.stream_pos) >= n):
        st.session_state.stream_pos += 1
        st.rerun()
    position = st.slider("Ticket in the sample", min_value=1, max_value=n, key="stream_pos")
    threshold = _active_threshold(assets)
    ticket = _cached_ticket(assets, int(position) - 1).at_threshold(threshold)
    seen = st.session_state.stream_seen
    if ticket.demo_id not in seen:
        seen.append(ticket.demo_id)
    _decision_banner(ticket.flagged, ticket.score, threshold, assets.metadata["score_kind"])
    st.markdown(plain(f"**{ticket.demo_id}** · amount {money(ticket.amount, cents=True)}"))
    st.markdown(plain(ticket.explanation.summary))
    for sentence in ticket.explanation.sentences:
        st.markdown(plain(f"- {sentence}"))
    st.caption(COMPONENT_NOTE)
    if ticket.label is not None:
        name = "Fraud" if ticket.label == 1 else "Legitimate"
        outcome = _outcome_phrase(ticket.label, ticket.flagged)
        realized = realized_unit_cost(
            ticket.label, ticket.flagged, ticket.amount, _cost_from_state()
        )
        st.markdown(
            plain(
                f"Held-out label, shown after the call: **{name}**. {outcome} "
                f"Realized cost {money(realized, cents=True)}."
            )
        )
    tally = _tally_seen(assets, threshold)
    cards = st.columns(4)
    cards[0].metric("Tickets seen", str(tally.seen))
    cards[1].metric("Frauds caught", str(tally.frauds_caught))
    cards[2].metric("Frauds missed", str(tally.frauds_missed))
    cards[3].metric("Replay cost", money(tally.realized_cost))
    st.caption(
        f"False alerts in the replay: {tally.false_alerts}. "
        "This tally is the sample you have viewed. It is not the test-set scorecard."
    )
    if st.button("Reset replay tally"):
        st.session_state.stream_seen = []
        st.rerun()


def _cached_ticket(assets, position: int):
    cache = st.session_state.explain_cache
    demo_id = str(assets.demo.iloc[position]["demo_id"])
    if demo_id not in cache:
        cache[demo_id] = score_demo_row(
            assets.demo,
            position,
            assets.model,
            model_key=assets.metadata["model_key"],
            feature_columns=list(assets.metadata["feature_columns"]),
            score_kind=assets.metadata["score_kind"],
            reference=assets.scenarios["typical_legitimate"],
            threshold=_saved_threshold(assets),
        )
    return cache[demo_id]


def _tally_seen(assets, threshold: float):
    rows = []
    cache = st.session_state.explain_cache
    for demo_id in st.session_state.stream_seen:
        ticket = cache[demo_id].at_threshold(threshold)
        rows.append({"flagged": ticket.flagged, "label": ticket.label, "amount": ticket.amount})
    return replay_tally(rows, _cost_from_state())


def _outcome_phrase(label: int, flagged: bool) -> str:
    if label == 1 and flagged:
        return "The review caught it."
    if label == 1 and not flagged:
        return "The ticket was approved and the fraud was missed."
    if label == 0 and flagged:
        return "A legitimate ticket was pulled."
    return "A legitimate ticket was approved."


def _score_tab(assets) -> None:
    st.subheader("Score one ticket")
    st.write(
        "V1–V28 are principal components. They do not have merchant names, ages, or countries. "
        "Profiles are medians from the training fold, not people."
    )
    threshold = _active_threshold(assets)
    st.caption(
        "Cutoff in use: "
        + format_threshold(threshold, assets.metadata["score_kind"])
        + (
            " (alert-budget queue)."
            if st.session_state.get("use_budget_cutoff")
            else " (published validation cutoff)."
        )
    )
    mode = st.radio("Source", ["Held-out example", "Training profile"], horizontal=True)
    if mode == "Held-out example":
        frame, label, amount = _demo_choice(assets)
    else:
        frame, label, amount = _profile_choice(assets)
    score = float(score_frame(assets.model, frame, assets.metadata["model_key"])[0])
    flagged = bool(np.isfinite(threshold) and score >= threshold)
    _decision_banner(flagged, score, threshold, assets.metadata["score_kind"])
    _explain(assets, frame)
    costs = single_ticket_costs(amount, _cost_from_state())
    grid = st.columns(4)
    grid[0].metric("If approved and legitimate", money(costs["approve_if_legitimate"], cents=True))
    grid[1].metric("If approved and fraud", money(costs["approve_if_fraud"], cents=True))
    grid[2].metric("If reviewed and legitimate", money(costs["review_if_legitimate"], cents=True))
    grid[3].metric("If reviewed and fraud", money(costs["review_if_fraud"], cents=True))
    st.caption("These four numbers do not use the score as a probability.")
    if label is not None and st.checkbox("Reveal held-out label", key="reveal_score_label"):
        name = "Fraud" if label == 1 else "Legitimate"
        realized = realized_unit_cost(label, flagged, amount, _cost_from_state())
        st.markdown(
            plain(
                f"Held-out label: **{name}**. Realized cost of this decision: "
                f"**{money(realized, cents=True)}**."
            )
        )


def _batch_tab(assets) -> None:
    st.subheader("Score a CSV")
    st.write(
        "Upload tickets with V1–V28 and Amount. The desk scores at most 2,000 rows, "
        "flags them at the cutoff in use, and writes a reason for the first flagged rows. "
        "Download the scored file when you want to keep it."
    )
    columns = list(assets.metadata["feature_columns"])
    template = template_from_demo(assets.demo, columns, n=8)
    st.download_button(
        "Download a sample CSV",
        data=to_csv_bytes(template),
        file_name="sample_transactions.csv",
        mime="text/csv",
    )
    upload = st.file_uploader("Transaction CSV", type=["csv"])
    if upload is None:
        st.caption(
            "The sample file has no labels. Add a Class column of 0/1 if you want realized cost."
        )
        return
    try:
        frame = pd.read_csv(upload)
        scored = score_batch(
            frame,
            assets.model,
            model_key=assets.metadata["model_key"],
            feature_columns=columns,
            score_kind=assets.metadata["score_kind"],
            threshold=_active_threshold(assets),
            reference=assets.scenarios.get("typical_legitimate"),
            cost=_cost_from_state(),
        )
    except (ValueError, TypeError, RuntimeError) as exc:
        st.error(str(exc))
        return
    flagged = int((scored["decision"] == "review").sum())
    cards = st.columns(3)
    cards[0].metric("Rows scored", f"{len(scored):,}")
    cards[1].metric("Flagged for review", f"{flagged:,}")
    cards[2].metric("Flag rate", percent(flagged / len(scored)))
    if "realized_cost" in scored.columns:
        st.metric("Realized cost of this file", money(float(scored["realized_cost"].sum())))
    path = ROOT / "artifacts" / "validation_scores.npz"
    psi = None
    if path.exists():
        psi = batch_psi(cached_reference_scores(str(path)), scored["fraud_score"].to_numpy())
    if psi is None:
        st.caption(
            "Score PSI is withheld under 30 rows. It would compare this file with the validation scores, "
            "and a short file is too noisy."
        )
    else:
        st.write(
            f"Score PSI against the validation reference: **{psi:.3f}**. "
            "This is this file versus the validation book, not a calendar-drift alarm. "
            "A PSI above 0.25 is a common review trigger in credit scorecards, and that cutoff is a convention."
        )
    preview_columns = [
        column
        for column in ("row_id", "Amount", "fraud_score", "decision", "top_reason", "realized_cost")
        if column in scored.columns
    ]
    st.dataframe(scored[preview_columns].head(20), hide_index=True, width="stretch")
    st.download_button(
        "Download scored CSV",
        data=to_csv_bytes(scored),
        file_name="scored_transactions.csv",
        mime="text/csv",
    )


def _models_tab(assets) -> None:
    st.subheader("Held-out scorecard")
    st.write(
        "Primary metric: precision-recall AUC, with a 95% bootstrap interval. "
        "The interval does not retune the cutoff. The dark selection was validation cost, "
        "not the best test PR-AUC."
    )
    st.dataframe(headline_comparison(assets.metrics), hide_index=True, width="stretch")
    st.dataframe(comparison_frame(assets.metrics), hide_index=True, width="stretch")
    st.caption(f"Test-cost leader: `{assets.metrics['test_cost_leader']}`.")
    figures = ROOT / "reports" / "figures"
    captions = {
        "03_precision_recall.png": "Test precision–recall. The dashed line is the fraud rate.",
        "05_model_comparison.png": "Test PR-AUC and savings. The dark bar is the validation choice.",
        "06_confusion_matrix.png": "Test decisions at the frozen cutoff.",
        "07_score_distributions.png": "Test scores for legitimate tickets and frauds.",
        "04_cost_vs_threshold.png": "Validation dollars against queue size for the chosen model.",
        "01_class_balance.png": "Legitimate volume dwarfs fraud. Accuracy is the wrong headline.",
        "02_amount_by_class.png": "Ticket size overlaps. A large-ticket rule is a weak net.",
    }
    for name, caption in captions.items():
        path = figures / name
        if path.exists():
            st.image(str(path), caption=caption)
    st.markdown(
        plain(
            f"Approving every test transaction costs {money(assets.metrics['approve_all_test_cost'])}. "
            f"Flag rate at the published cutoff: {percent(assets.metadata['test_summary']['flag_rate'])}."
        )
    )


def _monitor_tab(assets) -> None:
    st.subheader("What to watch on the next book")
    for note in assets.monitoring["notes"]:
        st.write(note)
    psi = float(assets.monitoring["validation_half_psi"])
    flag_rate = float(assets.monitoring["flag_rate"])
    cards = st.columns(3)
    cards[0].metric("Validation-half PSI", f"{psi:.4f}")
    cards[1].metric("Validation flag rate", percent(flag_rate, digits=3))
    cards[2].metric(
        "Validation recall",
        f"{float(assets.monitoring['recall']):.3f}",
    )
    st.caption(
        "The half-split PSI is two random halves of the same validation scores. "
        "It is a stability check. It is not evidence that production drifted."
    )
    table = profile_table(assets.monitoring)
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        "Score a CSV of at least 30 rows on the Batch tab to compare its score distribution "
        "with this reference. The model card in the repository restates these notes."
    )


def _decision_banner(flagged: bool, score: float, threshold: float, score_kind: str) -> None:
    text = plain(f"Score {score:.4f}. Cutoff {format_threshold(threshold, score_kind)}.")
    if flagged:
        st.error(f"Review this ticket. {text}")
    else:
        st.success(f"Approve this ticket. {text}")


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
    st.markdown(plain(explanation.summary))
    for sentence in explanation.sentences:
        st.markdown(plain(f"- {sentence}"))
    st.caption(COMPONENT_NOTE)


def _demo_choice(assets) -> tuple[pd.DataFrame, int, float]:
    demo = assets.demo
    choice = st.selectbox("Held-out example", demo["demo_id"].tolist())
    row = demo.loc[demo["demo_id"] == choice].iloc[0]
    columns = assets.metadata["feature_columns"]
    frame = pd.DataFrame([{column: float(row[column]) for column in columns}])
    st.markdown(plain(f"Amount on the ticket: **{money(float(row['Amount']), cents=True)}**."))
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


if __name__ == "__main__":
    main()
