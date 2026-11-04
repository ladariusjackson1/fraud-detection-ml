"""Locked case-study constants.

Amounts on the public file are euros. The case study converts them to dollars
with USD_PER_EUR before any model sees the data. Review costs are planning
assumptions, not estimated from the file.
"""

OPENML_DATA_ID = 1597
OPENML_NAME = "creditcard"
PUBLIC_ROWS = 284_807
PUBLIC_FRAUDS = 492
PUBLIC_FRAUD_RATE = PUBLIC_FRAUDS / PUBLIC_ROWS

DEFAULT_USD_PER_EUR = 1.10
DEFAULT_REVIEW_COST = 8.0
DEFAULT_FALSE_DECLINE_COST = 12.0
DEFAULT_CHARGEBACK_FEE = 25.0
MINUTES_PER_REVIEW = 4.0
DEFAULT_SEED = 42
NEGATIVE_PER_POSITIVE = 25
PRECISION_TARGETS = (0.50, 0.80)

V_COLUMNS = [f"V{i}" for i in range(1, 29)]

TAG_OPENML = "OpenML creditcard · data id 1597"
TAG_SYNTHETIC = "SYNTHETIC DATA — not OpenML transactions"
