# Switch the project to a new dataset

The teacher has changed the dataset. Replace `telecom_customer_churn_2500.csv` with
`Telco_customer_churn.xlsx` (IBM Telco churn: 7043 rows, 33 columns, sheet 1) and rebuild the
notebook on it. Keep the same structure, models, pipeline style, random_state=42 and the
leakage-safe workflow. Keep the old CSV in an `old_data/` folder and don't delete anything.

## Facts about the new file (already checked; verify them in the EDA)
- Load with `pd.read_excel("Telco_customer_churn.xlsx")` (needs `openpyxl`, so install it if missing).
- Column names have spaces (e.g. "Tenure Months", "Monthly Charges"). Rename them to CamelCase
  (TenureMonths, MonthlyCharges, ...) right after loading so the rest of the code stays readable.
- Target: `Churn Value` (1 = churn, 0 = stay). Churn rate is 26.5% (more imbalanced than before).
- `Total Charges` is text with 11 blank values. All 11 are customers with Tenure 0, so fill them with 0.
- `Total Charges` correlates 0.9996 with Tenure × MonthlyCharges, which makes it redundant again.
- `Senior Citizen` is Yes/No text here, not 0/1.
- No missing values otherwise, and no duplicate CustomerIDs.

## Columns to DROP, with the reason for each (put this table in a markdown cell)
| Column | Reason |
|---|---|
| CustomerID | Identifier |
| Count | Always 1 |
| Country, State | Constant (United States / California) |
| City | 1129 categories; too sparse and not in the problem statement |
| Zip Code, Lat Long, Latitude, Longitude | Location; not part of the problem statement (demographic/usage/billing/feedback) |
| Churn Label | Duplicate of the target |
| **Churn Score** | **LEAKAGE**: an IBM model score computed with knowledge of churn (corr 0.66 with target) |
| **Churn Reason** | **LEAKAGE**: only filled in for customers who already churned |
| CLTV | A predicted value from another model; drop it to keep the model honest (mention it as optional) |

Explain the leakage drops clearly. This is the most important point in the cleaning section.

## Columns that no longer exist
The new data has NO AvgMonthlyGB, MonthlyCalls, CustomerSupportCalls, LatePayments or
SatisfactionScore. Remove every cell, feature and markdown sentence that depends on them,
including ChargePerGB, LogAvgMonthlyGB, GBMissing, SatisfactionMissing, LowSatisfaction,
HighSupportCalls, SupportCallsPerMonth, LatePaymentRate, HighLatePayments, LowSat_x_HighSupport
and the MonthlyCalls data-quality fix. Also remove the old EDA findings that were specific to the
old data (0.24 max correlation, 83 phone inconsistencies, etc.).

## Problem-statement category mapping (state this explicitly)
- Demographic: Gender, SeniorCitizen, Partner, Dependents
- Usage: TenureMonths, PhoneService, MultipleLines, InternetService, the 6 add-on services
- Billing: Contract, PaperlessBilling, PaymentMethod, MonthlyCharges (TotalCharges is used only to derive features)
- Service feedback: **this dataset has no feedback column.** Say so honestly. Churn Reason is
  feedback, but it is leakage, so it is excluded. As the closest available proxy, use
  service-quality signals: TechSupport / OnlineSecurity uptake (`HasProtectionBundle`). Mention
  the missing category as a limitation in the conclusion. Also show a short bar chart of
  Churn Reason in the EDA ("why churners left"). It is used for insight only, never as a model input.

## Feature engineering (rebuild with what exists)
Keep or rebuild:
- AvgChargePerMonth, ChargeDelta, NumServices, HasProtectionBundle, TenureGroup, IsNewCustomer,
  IsMonthToMonth, AutoPay, MonthToMonth_x_New
Add:
- `IsFiber` and `Fiber_x_MonthToMonth`: fiber churns at 42% vs DSL at 19%, so this is a strong segment
- `NoInternet` flag (only 7% of these customers churn)
- `HasStreaming` (StreamingTV or StreamingMovies)
- `ElectronicCheck` flag (check its churn rate in the EDA first)
- `CustomerSegment` from K-Means on scaled TenureMonths, MonthlyCharges and NumServices, fitted
  inside the pipeline on training folds only. Choose k with elbow and silhouette, and show the
  segment profiles and churn rate per segment.
Run the ablation again and report it honestly, but keep the engineered features and the segment in
the final model, as before (the brief asks for them).

## Modelling and evaluation
- Same 4 models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting), tuned
  with 5-fold StratifiedKFold, class_weight='balanced', with SMOTE only as a comparison.
- Same comparison table: CV mean ± std (F1, ROC-AUC) plus test Accuracy/Precision/Recall/F1/ROC-AUC,
  with a threshold column, and a row for the best model at its tuned threshold.
- Expect higher scores than before (ROC-AUC around 0.84–0.86 is typical for this dataset). If any
  model scores above about 0.95, stop and check for leakage before continuing.
- Rewrite the "why scores are moderate" text to match the new results.

## Finish
1. Update every markdown cell and README.md to match the new data and the new numbers. No
   leftover text from the old dataset should remain.
2. Re-save churn_model.pkl and churn_features.py, and update predict_churn() and its example
   customer to the new column names.
3. Restart the kernel and run all cells, then confirm there are no errors and that the model loads
   and predicts in a fresh Python process.
4. Give me a short summary: columns dropped (and why), the final feature count, the final model
   and threshold, the test Precision/Recall/F1/ROC-AUC, and the top 5 churn drivers.
