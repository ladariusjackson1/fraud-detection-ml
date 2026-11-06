# Model card

Generated from `reports/metrics.json` and the validation-score reference. Do not hand-edit.

## Model

- **Selected model:** Random forest (`random_forest`).
- **Selection rule:** Minimum validation dollar cost. Ties break toward higher validation PR-AUC, then the model key. The test set is not used to choose the model or the cutoff.
- **Score type:** ranking score. Class-weighted scores are not calibrated probabilities.
- **Frozen cutoff:** 0.413574.
- **Primary metric:** precision-recall AUC (average precision). Accuracy is not used.
- **Test PR-AUC:** 0.679 (0.579–0.788).
- **Test cost:** $6,411 (95% bootstrap CI $2,102–$12,180).
- **Savings versus approve-all:** $10,903.
- **Rounding:** Each displayed dollar amount is rounded to the nearest dollar on its own. Exact approve-all cost is $17,314.52. Exact policy cost is $6,411.16. Exact savings are $10,903.35, which rounds to $10,903. Subtracting the already-rounded headlines gives $17,315 − $6,411 = $10,904.
- **Test-cost leader:** `logistic_undersampled`. Selection matches that leader: false.

## Intended use

Rank card transactions into a manual review queue. Each ticket is approved or reviewed. A review is assumed to save the full principal. This is not an automatic decline, not a chargeback model, and not a customer-level loss forecast.

## Data

- **Source:** OpenML creditcard · data id 1597.
- **Detail:** Public OpenML creditcard extract, verified at 284,807 rows and 492 frauds before dropping 9,144 exact duplicate rows. Amount was converted from EUR to USD at the planning rate. Time is not in this OpenML file, so no hour features were used.
- **Modeling rows:** 275,663. Frauds: 473 (0.1716%).
- **Split, seed 42:** train 165,397 (283 frauds), validation 55,133 (95 frauds), test 55,133 (95 frauds).
- **Features:** V1–V28 and Amount.

## Headline comparison

Precision-recall AUC is the ranking metric because fraud is rare. The interval resamples the test rows and does not retune the model or the cutoff. The three rows are the class-weighted logistic baseline, the random forest, and gradient boosting.

| Model | Val PR-AUC | Test PR-AUC (95% CI) | Test cost | Validation cost |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.652 | 0.689 (0.592–0.787) | $6,105 | $7,899 |
| **Random forest** | 0.640 | 0.679 (0.579–0.788) | $6,411 | $7,856 |
| Gradient boosting | 0.675 | 0.708 (0.607–0.799) | $6,801 | $8,335 |

The random forest is deployed only because it won validation cost. The full scorecard also fits an undersampled logistic regression, an isolation forest, and an amount-only rule. Those rows are below. Hiding them would hide the test-cost leader when it is not the selected model.

## Full test scorecard

| Model | Test PR-AUC (95% CI) | Precision | Recall | Test cost | Selected |
| --- | ---: | ---: | ---: | ---: | --- |
| Logistic regression | 0.689 (0.592–0.787) | 0.789 | 0.747 | $6,105 |  |
| Logistic regression, undersampled | 0.713 (0.625–0.806) | 0.721 | 0.789 | $5,728 |  |
| Random forest | 0.679 (0.579–0.788) | 0.752 | 0.800 | $6,411 | yes |
| Gradient boosting | 0.708 (0.607–0.799) | 0.714 | 0.789 | $6,801 |  |
| Isolation forest | 0.197 (0.141–0.288) | 0.132 | 0.674 | $14,346 |  |
| Amount-only rule | 0.002 (0.001–0.004) | n/a | 0.000 | $17,315 |  |

## Operating point on the test set

- Cutoff frozen on validation: 0.413574.
- Reviews per 10,000 test transactions: 18.3.
- Caught frauds: 76. Missed frauds: 19. Legitimate tickets pulled: 25.
- Missed-fraud principal: $4,828.
- Approve-all cost on the same rows: $17,315.

Approve-all pays the principal on all 95 test frauds, $14,939.52, plus $25.00 on each ($2,375.00), and no review cost: $17,314.52. The $4,828 line is the principal of the 19 missed frauds only.

The savings column is the exact cost difference, rounded to the nearest dollar. It is not the difference of the rounded cost figures. Those two procedures differ by $1 for: Logistic regression, undersampled; Random forest; Isolation forest.

## Cost assumptions

- Review: $8.00 (4 minutes).
- False decline, on top of the review: $12.00.
- Chargeback fee on an approved fraud, on top of principal: $25.00.
- Approving a legitimate ticket costs nothing in this loss account. Interchange is not credited.
- These are planning inputs. They are not estimated from the file.

## Monitoring

This file has no Time column, so calendar drift was not measured. The figures below are the validation-book reference a later extract would be compared with.

At the frozen cutoff the validation book flags 0.172% of tickets (17.2 per 10,000), with precision 0.779 and recall 0.779.

A random split of the validation scores has PSI 0.0002. That number checks whether two halves of the same book look alike. It is not a drift event. Compare a new batch with population_stability_index against these validation scores. A PSI above 0.25 is a common credit-scorecard review trigger; that cutoff is a convention, not a result estimated from this file.

When a new book arrives, compare four things with this reference before refitting: the flag rate at the frozen cutoff, the score PSI, the precision of tickets that were actually reviewed once labels arrive, and the amount median. A one-day move inside the bootstrap interval on test cost is not a reason to refit.

Validation reference quantiles:

| Slice | p50 | p90 | p99 | p999 |
| --- | ---: | ---: | ---: | ---: |
| Score, all | 0.0000 | 0.0100 | 0.0870 | 0.8877 |
| Score, legitimate | 0.0000 | 0.0100 | 0.0779 | 0.2048 |
| Score, fraud | 0.8778 | 0.9995 | 0.9996 | 0.9996 |
| Amount USD, legitimate | $25.84 | $226.60 | $1,100.00 | $3,213.37 |
| Amount USD, fraud | $7.41 | $651.40 | $1,675.54 | $1,959.14 |

## Limits

- One stratified split. The bootstrap resamples that test slice. It does not refit.
- V1–V28 are principal components. This card does not name the original card fields.
- A review is assumed to recover the whole principal. Partial recovery is out of scope.
- The public amounts were converted at a planning FX rate when the source was euros.
- Population stability on a later book is a comparison to this reference, not a measured outage.

## Reproduce

`python -m fraudlab --source auto` rewrites `reports/metrics.json`, this card, and the validation reference.
