OK, proceed with Sections 2–7 with your proposed handling, plus:
1. MonthlyCalls: set to 0 for customers with PhoneService = No (or flag them), and document this as a data-quality issue in the notebook.
2. Add threshold flags from the EDA: HighLatePayments (LatePayments >= 3) and HighSupportCalls (CustomerSupportCalls >= 3).
3. Add interaction features: MonthToMonth_x_New (IsMonthToMonth & IsNewCustomer) and LowSat_x_HighSupport. Keep them only if the ablation supports them.
4. Use log1p(AvgMonthlyGB) as the model input as well, not only for K-Means.
5. When pruning, keep at least one feature from each problem-statement category (demographic, usage, billing, feedback) and say so explicitly.
6. In the evaluation section, note that the weak raw correlations (max 0.24) mean moderate scores are expected, and judge models on churn-class recall/F1 and ROC-AUC.
7. Confirmed: I don't have the report. Proceed without it.
