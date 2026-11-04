"""Command line entry for the scorecard run."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fraudlab.constants import (
    DEFAULT_CHARGEBACK_FEE,
    DEFAULT_FALSE_DECLINE_COST,
    DEFAULT_REVIEW_COST,
    DEFAULT_SEED,
    DEFAULT_USD_PER_EUR,
    PUBLIC_ROWS,
)
from fraudlab.costs import CostConfig
from fraudlab.experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit the fraud review scorecard and write reports/ plus artifacts/."
    )
    parser.add_argument("--source", choices=["auto", "openml", "synthetic"], default="auto")
    parser.add_argument("--synthetic-rows", type=int, default=PUBLIC_ROWS)
    parser.add_argument("--fraud-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--usd-per-eur", type=float, default=DEFAULT_USD_PER_EUR)
    parser.add_argument("--review-cost", type=float, default=DEFAULT_REVIEW_COST)
    parser.add_argument("--false-decline-cost", type=float, default=DEFAULT_FALSE_DECLINE_COST)
    parser.add_argument("--chargeback-fee", type=float, default=DEFAULT_CHARGEBACK_FEE)
    parser.add_argument("--bootstraps", type=int, default=400)
    parser.add_argument("--output", type=Path, default=Path("."))
    parser.add_argument("--fast", action="store_true", help="Smaller models for a smoke run")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cost = CostConfig(
        review_cost=args.review_cost,
        false_decline_cost=args.false_decline_cost,
        chargeback_fee=args.chargeback_fee,
    )
    metrics = run_experiment(
        output_dir=args.output,
        source=args.source,
        synthetic_rows=args.synthetic_rows,
        fraud_rate=args.fraud_rate,
        random_state=args.seed,
        usd_per_eur=args.usd_per_eur,
        cost=cost,
        fast=args.fast,
        n_boot=args.bootstraps,
    )
    selected_cost = next(
        row["test_cost"] for row in metrics["models"] if row["key"] == metrics["selected_model"]
    )
    logging.info(
        "Selected %s. Test cost %.2f. Artifacts written under %s",
        metrics["selected_model"],
        selected_cost,
        args.output,
    )
    return 0
