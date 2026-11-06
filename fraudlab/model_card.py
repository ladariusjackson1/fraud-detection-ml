"""Model card rendered from the scorecard. Numbers are not typed in by hand."""

from __future__ import annotations

from fraudlab.monitoring import profile_table
from fraudlab.reporting import (
    approve_all_composition_note,
    format_ci,
    independent_rounding_note,
    money,
    savings_column_note,
)

HEADLINE_KEYS = ("logistic_balanced", "random_forest", "gradient_boosting")


def _ci_money(interval: list[float | None] | None) -> str:
    if not interval or interval[0] is None or interval[1] is None:
        return "n/a"
    return f"{money(float(interval[0]))}–{money(float(interval[1]))}"


def _threshold_text(row: dict) -> str:
    if row.get("threshold") is None:
        return "approve all"
    if row.get("score_kind") == "amount_usd":
        return money(float(row["threshold"]), cents=True)
    return f"{float(row['threshold']):.6f}"


def _row(models: list[dict], key: str) -> dict:
    found = [row for row in models if row["key"] == key]
    if not found:
        raise KeyError(f"Scorecard is missing {key}")
    return found[0]


def render_model_card(metrics: dict, monitoring: dict) -> str:
    models = list(metrics["models"])
    selected = _row(models, metrics["selected_model"])
    source = metrics["data_source"]
    cost = metrics["cost_config"]
    train = metrics["split"]["train"]
    validation = metrics["split"]["validation"]
    test = metrics["split"]["test"]
    features = "V1–V28 and Amount"
    if monitoring.get("has_time"):
        features += " and Time"
    lines: list[str] = [
        "# Model card",
        "",
        "Generated from `reports/metrics.json` and the validation-score reference. Do not hand-edit.",
        "",
        "## Model",
        "",
        f"- **Selected model:** {selected['display_name']} (`{selected['key']}`).",
        f"- **Selection rule:** {metrics['selection_rule']}",
        "- **Score type:** ranking score. Class-weighted scores are not calibrated probabilities.",
        f"- **Frozen cutoff:** {_threshold_text(selected)}.",
        "- **Primary metric:** precision-recall AUC (average precision). Accuracy is not used.",
        (
            "- **Test PR-AUC:** "
            f"{format_ci(selected['test_average_precision'], selected['test_average_precision_ci95'])}."
        ),
        (
            f"- **Test cost:** {money(selected['test_cost'])} "
            f"(95% bootstrap CI {_ci_money(selected['test_cost_ci95'])})."
        ),
        f"- **Savings versus approve-all:** {money(selected['savings_vs_approve_all'])}.",
        f"- **Rounding:** {independent_rounding_note(metrics['approve_all_test_cost'], selected['test_cost'])}",
        (
            f"- **Test-cost leader:** `{metrics['test_cost_leader']}`. "
            f"Selection matches that leader: {str(metrics['selection_matches_test_leader']).lower()}."
        ),
        "",
        "## Intended use",
        "",
        "Rank card transactions into a manual review queue. Each ticket is approved or reviewed. "
        "A review is assumed to save the full principal. This is not an automatic decline, "
        "not a chargeback model, and not a customer-level loss forecast.",
        "",
        "## Data",
        "",
        f"- **Source:** {source['chart_tag']}.",
        f"- **Detail:** {source['detail']}",
        (
            f"- **Modeling rows:** {int(source['rows']):,}. "
            f"Frauds: {int(source['fraud_count']):,} ({float(source['fraud_rate']):.4%})."
        ),
        (
            f"- **Split, seed {metrics['random_state']}:** "
            f"train {int(train['rows']):,} ({int(train['frauds'])} frauds), "
            f"validation {int(validation['rows']):,} ({int(validation['frauds'])} frauds), "
            f"test {int(test['rows']):,} ({int(test['frauds'])} frauds)."
        ),
        f"- **Features:** {features}.",
        "",
        "## Headline comparison",
        "",
        "Precision-recall AUC is the ranking metric because fraud is rare. "
        "The interval resamples the test rows and does not retune the model or the cutoff. "
        "The three rows are the class-weighted logistic baseline, the random forest, "
        "and gradient boosting.",
        "",
        "| Model | Val PR-AUC | Test PR-AUC (95% CI) | Test cost | Validation cost |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key in HEADLINE_KEYS:
        row = _row(models, key)
        mark = "**" if row["key"] == metrics["selected_model"] else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{mark}{row['display_name']}{mark}",
                    f"{row['val_average_precision']:.3f}",
                    format_ci(row["test_average_precision"], row["test_average_precision_ci95"]),
                    money(row["test_cost"]),
                    money(row["val_cost"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The random forest is deployed only because it won validation cost. "
            "The full scorecard also fits an undersampled logistic regression, an isolation forest, "
            "and an amount-only rule. Those rows are below. Hiding them would hide the test-cost leader "
            "when it is not the selected model.",
            "",
            "## Full test scorecard",
            "",
            "| Model | Test PR-AUC (95% CI) | Precision | Recall | Test cost | Selected |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in models:
        precision = "n/a" if row["test_precision"] is None else f"{row['test_precision']:.3f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["display_name"],
                    format_ci(row["test_average_precision"], row["test_average_precision_ci95"]),
                    precision,
                    f"{row['test_recall']:.3f}",
                    money(row["test_cost"]),
                    "yes" if row["key"] == metrics["selected_model"] else "",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Operating point on the test set",
            "",
            f"- Cutoff frozen on validation: {_threshold_text(selected)}.",
            f"- Reviews per 10,000 test transactions: {selected['test_reviews_per_10k']:.1f}.",
            (
                f"- Caught frauds: {int(selected['test_true_positives'])}. "
                f"Missed frauds: {int(selected['test_false_negatives'])}. "
                f"Legitimate tickets pulled: {int(selected['test_false_positives'])}."
            ),
            f"- Missed-fraud principal: {money(selected['test_false_negative_amount'])}.",
            f"- Approve-all cost on the same rows: {money(metrics['approve_all_test_cost'])}.",
            "",
        ]
    )
    composition = approve_all_composition_note(metrics)
    if composition:
        lines.append(composition)
        lines.append("")
    lines.extend(
        [
            savings_column_note(metrics),
            "",
            "## Cost assumptions",
            "",
            f"- Review: {money(cost['review_cost'], cents=True)} ({cost['minutes_per_review']:g} minutes).",
            f"- False decline, on top of the review: {money(cost['false_decline_cost'], cents=True)}.",
            f"- Chargeback fee on an approved fraud, on top of principal: {money(cost['chargeback_fee'], cents=True)}.",
            "- Approving a legitimate ticket costs nothing in this loss account. Interchange is not credited.",
            "- These are planning inputs. They are not estimated from the file.",
            "",
            "## Monitoring",
            "",
        ]
    )
    notes = monitoring.get("notes") or []
    if not notes:
        raise ValueError("Monitoring reference has no notes")
    for note in notes:
        lines.append(note)
        lines.append("")
    lines.extend(
        [
            "Validation reference quantiles:",
            "",
            "| Slice | p50 | p90 | p99 | p999 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    table = profile_table(monitoring)
    for record in table.to_dict(orient="records"):
        slice_name = str(record["Slice"])
        if slice_name.startswith("Amount"):
            cells = [money(float(record[key]), cents=True) for key in ("p50", "p90", "p99", "p999")]
        else:
            cells = [f"{float(record[key]):.4f}" for key in ("p50", "p90", "p99", "p999")]
        lines.append("| " + " | ".join([slice_name, *cells]) + " |")
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- One stratified split. The bootstrap resamples that test slice. It does not refit.",
            "- V1–V28 are principal components. This card does not name the original card fields.",
            "- A review is assumed to recover the whole principal. Partial recovery is out of scope.",
            "- The public amounts were converted at a planning FX rate when the source was euros.",
            "- Population stability on a later book is a comparison to this reference, not a measured outage.",
            "",
            "## Reproduce",
            "",
            "`python -m fraudlab --source auto` rewrites `reports/metrics.json`, this card, and the validation reference.",
            "",
        ]
    )
    return "\n".join(lines)
