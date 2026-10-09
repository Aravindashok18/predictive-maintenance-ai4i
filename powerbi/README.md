# Power BI report

These files are the data behind the dashboard. The `.pbix` file itself is not in the repository.

| file | grain | use |
|---|---|---|
| `fact_machines.csv` | one row per machine (10,000) | measures, slicers, scatter and band charts |
| `fact_failure_modes.csv` | one row per failure event and mode | failure-mode breakdown |
| `feature_importance.csv` | one row per feature | driver bar chart |
| `model_comparison.csv` | one row per model | model comparison table |
| `pr_curve.csv` | one row per threshold | precision-recall line chart |

`failure_risk` is an out-of-fold probability: each machine is scored by a model that was trained without it.
`flagged` applies the threshold chosen on the training folds (see `reports/final_model.json`).

## Building the report

1. Get data > Text/CSV, load the five files.
2. Model view: relate `fact_failure_modes[product_id]` to `fact_machines[product_id]` (many-to-one, single direction).
3. Paste the measures from `measures.dax` (New measure, one at a time).
4. Suggested pages:
   - **Overview**: cards for Machines, Failures, Failure Rate, Recall, Precision; failure rate by `type`; Mode Events by `failure_mode`.
   - **Drivers**: `feature_importance` bar chart; failure rate by `rpm_band` and by `wear_band`; scatter of `rpm` against `torque_nm` with `failure` as legend.
   - **Model**: `model_comparison` table; `pr_curve` line chart (recall on the x axis, precision on the y axis).
