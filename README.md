# Telecom Customer Churn Prediction

A classification model that predicts whether a telecom customer will churn (leave) or stay,
using demographic, usage and billing data from the IBM Telco customer churn dataset.

## Files

| File | What it is |
|---|---|
| `churn_prediction.ipynb` | The full analysis: EDA, cleaning, feature engineering, models, evaluation, conclusion |
| `Telco_customer_churn.xlsx` | The dataset: 7,043 customers, 33 columns, target `Churn Value` (1 = churn, 0 = stay) |
| `churn_features.py` | Cleaning and feature-engineering code. Written by the notebook (Section 3.2) and needed to load the model |
| `churn_model.pkl` | The saved final model: pipeline, decision threshold and feature list |
| `train.py` | Standalone training script: same features, pipeline, models and grids as the notebook |
| `train.bat`, `train_quick.bat` | Double-click launchers for Windows (full run / quick test run) |
| `requirements.txt` | Package versions used for this project |
| `outputs/` | Created by `train.py`: `churn_model.pkl`, `results.csv`, `training_log.txt` |
| `old_data/` | The previous dataset (`telecom_customer_churn_2500.csv`) and the earlier version of the notebook, model and README, kept for reference |

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn openpyxl jupyter
jupyter notebook churn_prediction.ipynb
```

Then choose **Kernel > Restart & Run All**. The notebook runs top to bottom in a few minutes and
re-creates `churn_features.py` and `churn_model.pkl`. `random_state=42` is fixed everywhere.

## How to train

The notebook explains the analysis; `train.py` trains the model without opening Jupyter.

**Double-click (Windows):**
- `train.bat` - full training run (about 1.5 minutes).
- `train_quick.bat` - fast test run with smaller grids and 3-fold cross-validation (under 20 seconds).

The first time, the launcher creates a private Python environment in `.venv` and installs the
packages from `requirements.txt` (about a minute and a half, needs internet). Later runs reuse it.
Python 3.11 must be installed and on the PATH. The window stays open at the end so you can read the results.

**Standalone `train.exe` (no Python needed):** `dist/train.exe` (about 119 MB) contains Python and
all packages. Copy it into the folder that holds `Telco_customer_churn.xlsx` and double-click it,
or run it from a terminal with the same options as below (`train.exe --quick`). It looks for the
data file and writes `outputs/` next to itself, and keeps the window open until you press Enter.
It takes a few seconds longer to start than `train.py`, because it unpacks itself first.
To rebuild it after changing the code: `python -m PyInstaller train.spec`.

**From a terminal** (in this folder, with the packages installed):

```bash
python train.py --quick          # fast test run
python train.py                  # full run, same settings as the notebook
python train.py --model rf       # only one model: lr, dt, rf, gb (default: all)
python train.py --smote          # SMOTE instead of class weights (default: --no-smote)
python train.py --data other.csv # a different data file (.xlsx or .csv with the same columns)
```

The results go to `outputs/`: `churn_model.pkl` (pipeline + threshold + feature list),
`results.csv` (the comparison table) and `training_log.txt` (everything printed, with a timestamp).

**How the script picks the final model.** It takes the model with the highest cross-validated
ROC-AUC. Models within one standard deviation of it count as a tie, and the tie is broken by the
highest cross-validated F1. On this data Logistic Regression, Random Forest and Gradient Boosting
tie, so the script selects **Random Forest** (CV F1 0.651) with a tuned threshold of 0.48. This is
a different choice from the notebook, which picks Logistic Regression for interpretability. Every
model's scores are identical in the two.

To predict for a new customer with the script's model:

```python
from train import predict_churn
predict_churn({"TenureMonths": 3, "Contract": "Month-to-month", ...})   # all input columns
```

## Approach

1. **EDA** - column names converted to CamelCase, class balance (26.5% churn), leakage checks,
   churn rate against the key drivers, correlations.
2. **Cleaning** - drop identifier, location, duplicate-target and leaking columns; fill the 11 blank
   `TotalCharges` (all tenure 0) with 0; normalise "No internet/phone service" to "No"; then a
   stratified 80/20 split.
3. **Feature engineering** - 15 engineered features, including K-Means customer segments (k = 4),
   built inside the pipeline.
4. **Pipeline** - feature engineering, K-Means, encoding and scaling in one scikit-learn `Pipeline`,
   so nothing is fitted on validation or test data.
5. **Models** - Logistic Regression, Decision Tree, Random Forest and Gradient Boosting, each tuned
   with 5-fold `GridSearchCV` on F1. Class imbalance handled with `class_weight="balanced"`; SMOTE
   was compared and gave no gain.
6. **Evaluation** - test-set comparison, confusion matrices, ROC curves, threshold tuning, churn drivers.

## Columns dropped

| Column | Reason |
|---|---|
| `CustomerID` | Identifier |
| `Count` | Always 1 |
| `Country`, `State` | Constant (United States / California) |
| `City` | 1,129 categories; too sparse and not in the problem statement |
| `Zip Code`, `Lat Long`, `Latitude`, `Longitude` | Location; not part of the problem statement |
| `Churn Label` | Duplicate of the target |
| **`Churn Score`** | **Leakage**: an IBM model score computed with knowledge of churn (correlation 0.66; ROC-AUC 0.94 on its own) |
| **`Churn Reason`** | **Leakage**: only filled in for customers who already churned |
| `CLTV` | A predicted value from another model (optional input, excluded to keep the model honest) |
| `Gender` | No link with churn (chi-square p = 0.89) |
| `Phone Service` | Not significant (p = 0.20) |
| `Total Charges` | Redundant with tenure × monthly charges (correlation 0.9996); used only to derive features |

## Final feature set

31 features, 15 of them engineered:

- **Demographic:** `SeniorCitizen`, `Partner`, `Dependents`.
- **Usage:** `TenureMonths`, `InternetService`, `MultipleLines`, the 6 internet add-ons, plus the
  engineered `NumServices`, `HasStreaming`, `IsFiber`, `NoInternet`, `IsNewCustomer`, `TenureGroup`.
- **Billing:** `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, plus the engineered
  `AvgChargePerMonth`, `ChargeDelta`, `IsMonthToMonth`, `AutoPay`, `ElectronicCheck`.
- **Service feedback:** none in this dataset. `Churn Reason` is feedback but is leakage, so the proxy
  `HasProtectionBundle` (has online security or tech support) is used.
- **K-Means segment:** `CustomerSegment`.
- **Interactions:** `MonthToMonth_x_New`, `Fiber_x_MonthToMonth`.

## Results

Cross-validation is 5-fold on the training set (mean ± std); test is the held-out 1,409 customers.

| Model | Threshold | CV F1 | CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.50 | 0.646 ± 0.021 | 0.861 ± 0.011 | 0.752 | 0.522 | 0.781 | 0.626 | 0.853 |
| Decision Tree | 0.50 | 0.629 ± 0.005 | 0.844 ± 0.004 | 0.735 | 0.500 | 0.821 | 0.621 | 0.842 |
| Random Forest | 0.50 | 0.651 ± 0.010 | 0.859 ± 0.009 | 0.769 | 0.545 | 0.786 | 0.644 | 0.853 |
| Gradient Boosting | 0.50 | 0.648 ± 0.022 | 0.863 ± 0.011 | 0.759 | 0.531 | 0.786 | 0.634 | 0.852 |
| **Logistic Regression @ threshold 0.51** | 0.51 | 0.647 ± 0.024 | 0.861 ± 0.011 | 0.756 | 0.527 | 0.778 | 0.629 | 0.853 |

**Final model: Logistic Regression with a decision threshold of 0.51.**

Main findings:

- Logistic Regression, Random Forest and Gradient Boosting are within noise of each other (the gaps
  are smaller than one cross-validation standard deviation). Logistic Regression is chosen for its interpretability.
- The model already reaches about 80% recall at the default threshold, so the tuned threshold
  (highest threshold that keeps recall at 80% or more on the training folds) is only slightly higher, at 0.51.
- The strongest churn drivers are contract type, tenure, fiber-optic internet, dependents and
  having no internet service (phone-only customers rarely churn).
- The engineered features, including the K-Means segment, do not change the scores beyond noise.
  They are kept in the final model because they cost nothing and express the EDA findings directly.
- The K-Means segments describe customers well: churn ranges from 5% (long-standing, low-bill
  customers) to 45% (newer fiber customers on monthly contracts).
- ROC-AUC of about 0.85 is in the normal range for this dataset and far below 0.95, so there is no
  sign of leakage.

## Using the saved model

`churn_features.py` must be in the same folder as `churn_model.pkl`.

```python
import joblib
import pandas as pd
from churn_features import clean_data

saved = joblib.load("churn_model.pkl")
customer = clean_data(pd.DataFrame([{...}]))        # one dict with the input columns
probability = saved["pipeline"].predict_proba(customer)[0, 1]
print("Churn" if probability >= saved["threshold"] else "Stay", round(probability, 3))
```

Keys can be the CamelCase names used in the notebook (`"TenureMonths"`) or the original Excel names
(`"Tenure Months"`). The notebook's `predict_churn(customer_dict)` function (Section 7.4) wraps these steps.

## Limitations

- No genuine service-feedback column: `Churn Reason` is leakage, so a proxy is used.
- Moderate precision (about 0.53): roughly half of the flagged customers would not have churned.
- Engineered features did not improve accuracy, and overlapping features share credit, so the
  drivers are reliable by theme, not coefficient by coefficient.
- IBM's sample dataset of a fictional company: one region (California), no dates.
- The reference report mentioned in the assignment was not available, so model choices are justified without it.
