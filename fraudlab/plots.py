"""Charts for the scorecard. Every title carries the data-source tag."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

FIGURE_FILES = (
    "01_class_balance.png",
    "02_amount_by_class.png",
    "03_precision_recall.png",
    "04_cost_vs_threshold.png",
    "05_model_comparison.png",
    "06_confusion_matrix.png",
    "07_score_distributions.png",
)

INK = "#1F2933"
MUTED = "#627D98"
LEGIT = "#1F4E79"
FRAUD = "#B42318"
SELECTED = "#1F4E79"
OTHER = "#98A8B8"
GRID = "#E6E2DC"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D5D0C8",
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", color=INK, pad=10)


def _footer(fig: plt.Figure, source_tag: str) -> None:
    fig.text(0.0, -0.02, source_tag, color=MUTED, fontsize=8, ha="left")
    fig.tight_layout()


def save_class_balance(path: Path, *, legitimate: int, fraud: int, source_tag: str) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    counts = [legitimate, fraud]
    labels = ["Legitimate", "Fraud"]
    colors = [LEGIT, FRAUD]
    bars = ax.bar(labels, counts, color=colors, width=0.62)
    ax.set_yscale("log")
    ax.set_ylabel("Transactions (log scale)")
    rate = fraud / max(legitimate + fraud, 1)
    _title(ax, f"Class balance  ·  fraud rate {rate:.3%}")
    for bar, count in zip(bars, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}",
            ha="center",
            va="bottom",
            color=INK,
            fontsize=10,
        )
    ax.set_ylim(min(counts) * 0.6, max(counts) * 3)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _footer(fig, source_tag)
    save_figure(fig, path)


def save_amount_by_class(
    path: Path,
    *,
    legitimate_amounts: np.ndarray,
    fraud_amounts: np.ndarray,
    source_tag: str,
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    legit = np.log10(np.asarray(legitimate_amounts, dtype=float) + 1.0)
    fraud = np.log10(np.asarray(fraud_amounts, dtype=float) + 1.0)
    hi = max(float(legit.max()), float(fraud.max()), 1.0)
    bins = np.linspace(0, hi, 36)
    ax.hist(legit, bins=bins, density=True, alpha=0.55, color=LEGIT, label="Legitimate")
    ax.hist(fraud, bins=bins, density=True, alpha=0.65, color=FRAUD, label="Fraud")
    ax.set_xlabel("log10(amount in USD + 1)")
    ax.set_ylabel("Density")
    _title(ax, "Ticket size by class")
    ax.legend(loc="upper right")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _footer(fig, source_tag)
    save_figure(fig, path)


def save_pr_curves(
    path: Path,
    *,
    curves: list[dict],
    prevalence: float,
    source_tag: str,
    selected_key: str,
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    palette = ["#1F4E79", "#0F6E56", "#B54708", "#7A3E9D", "#B42318", "#5C6B7A"]
    for index, curve in enumerate(curves):
        color = palette[index % len(palette)]
        width = 2.4 if curve["key"] == selected_key else 1.6
        ax.plot(curve["recall"], curve["precision"], color=color, lw=width, label=curve["name"])
    ax.axhline(prevalence, color=MUTED, ls="--", lw=1, label=f"Prevalence ({prevalence:.3%})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    _title(ax, "Precision–recall on the held-out test set")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _footer(fig, source_tag)
    save_figure(fig, path)


def focus_queue(
    reviews_per_10k: np.ndarray,
    costs: np.ndarray,
    chosen_reviews: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop the flag-everyone tail so the cost minimum is visible.

    Reviewing every transaction costs orders of magnitude more than the
    operating region. A full-range axis hides the dip the cutoff is chosen from.
    """
    reviews = np.asarray(reviews_per_10k, dtype=float)
    totals = np.asarray(costs, dtype=float)
    cap = max(100.0, float(chosen_reviews) * 6.0)
    mask = np.isfinite(reviews) & (reviews <= cap)
    if int(np.sum(mask)) < 8:
        return reviews, totals
    return reviews[mask], totals[mask]


def save_cost_curve(
    path: Path,
    *,
    reviews_per_10k: np.ndarray,
    costs: np.ndarray,
    chosen_reviews: float,
    chosen_cost: float,
    model_name: str,
    source_tag: str,
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    shown_reviews, shown_costs = focus_queue(reviews_per_10k, costs, chosen_reviews)
    order = np.argsort(shown_reviews)
    ax.plot(shown_reviews[order], shown_costs[order], color=LEGIT, lw=2)
    ax.scatter([chosen_reviews], [chosen_cost], color=FRAUD, s=42, zorder=3, label="Chosen cutoff")
    ax.set_xlim(left=0)
    ceiling = float(np.nanmax(shown_costs))
    ax.set_ylim(0, ceiling * 1.08 if ceiling > 0 else 1)
    ax.set_xlabel("Reviews per 10,000 validation transactions")
    ax.set_ylabel("Validation cost (USD)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"${value:,.0f}"))
    _title(ax, f"Validation cost near the chosen queue\n{model_name}")
    ax.legend(loc="upper right")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _footer(fig, source_tag)
    save_figure(fig, path)


def save_model_comparison(
    path: Path,
    *,
    names: list[str],
    keys: list[str],
    pr_auc: list[float],
    savings: list[float],
    selected_key: str,
    source_tag: str,
) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0))
    colors = [SELECTED if key == selected_key else OTHER for key in keys]
    y = np.arange(len(names))
    axes[0].barh(y, pr_auc, color=colors)
    axes[0].set_yticks(y, names)
    axes[0].set_xlabel("Test PR-AUC")
    axes[0].invert_yaxis()
    _title(axes[0], "Ranking quality")
    axes[1].barh(y, savings, color=colors)
    axes[1].axvline(0, color=MUTED, lw=1)
    axes[1].set_yticks(y, [])
    axes[1].set_xlabel("Test savings versus approve-all")
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[1].xaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _pos: f"${value / 1000:.0f}k" if abs(value) >= 1000 else f"${value:.0f}"
        )
    )
    axes[1].invert_yaxis()
    _title(axes[1], "Dollar outcome")
    for ax in axes:
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Held-out comparison  ·  highlighted bar is the validation choice",
        color=INK,
        x=0.01,
        ha="left",
    )
    _footer(fig, source_tag)
    save_figure(fig, path)


def save_confusion(
    path: Path,
    *,
    true_negatives: int,
    false_positives: int,
    false_negatives: int,
    true_positives: int,
    false_negative_amount: float,
    model_name: str,
    source_tag: str,
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    counts = [
        [true_negatives, false_positives],
        [false_negatives, true_positives],
    ]
    colors = [["#E6E9EE", "#F6D6A8"], ["#F4C7C3", "#C7E6D5"]]
    notes = [
        ["Approved legitimate", "Reviewed legitimate"],
        [f"Missed fraud\n${false_negative_amount:,.0f} principal", "Reviewed fraud"],
    ]
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    for row in range(2):
        for col in range(2):
            ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, color=colors[row][col]))
            ax.text(
                col,
                row - 0.08,
                f"{counts[row][col]:,}",
                ha="center",
                va="center",
                fontsize=16,
                color=INK,
            )
            ax.text(
                col, row + 0.22, notes[row][col], ha="center", va="center", fontsize=8, color=MUTED
            )
    ax.set_xticks([0, 1], ["Predicted approve", "Predicted review"])
    ax.set_yticks([0, 1], ["Actual legitimate", "Actual fraud"])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _title(ax, f"Test decisions\n{model_name}")
    _footer(fig, source_tag)
    save_figure(fig, path)


def save_score_distributions(
    path: Path,
    *,
    legitimate_scores: np.ndarray,
    fraud_scores: np.ndarray,
    threshold: float,
    score_label: str,
    model_name: str,
    source_tag: str,
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.hist(legitimate_scores, bins=40, density=True, alpha=0.55, color=LEGIT, label="Legitimate")
    ax.hist(fraud_scores, bins=40, density=True, alpha=0.65, color=FRAUD, label="Fraud")
    if np.isfinite(threshold):
        ax.axvline(threshold, color=INK, ls="--", lw=1.2, label="Validation cutoff")
    ax.set_xlabel(score_label)
    ax.set_ylabel("Density")
    _title(ax, f"Test score distributions\n{model_name}")
    ax.legend(loc="best")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _footer(fig, source_tag)
    save_figure(fig, path)
