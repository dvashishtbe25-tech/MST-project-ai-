"""Train the telecom churn model without Jupyter.

The notebook (churn_prediction.ipynb) explains every decision; this script only does the
training, with exactly the same features, pipeline, models, grids and random_state=42.

Usage examples (from a terminal in this folder):
    python train.py                 full run: 5-fold CV, all four models
    python train.py --quick         fast test run: small grids, 3-fold CV
    python train.py --model lr      train only Logistic Regression
    python train.py --smote         use SMOTE instead of class weights

Outputs go to the outputs/ folder: churn_model.pkl, results.csv and training_log.txt.
"""
import argparse
import importlib.util
import sys
import time
from datetime import datetime
from pathlib import Path

RANDOM_STATE = 42
# The folder the data and outputs live in. Normally this is the folder of train.py; when the script
# runs as train.exe (built with PyInstaller, sys.frozen is set) it is the folder of the .exe file.
FROZEN = getattr(sys, "frozen", False)
HERE = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"

# Packages the script needs: (name used by "import", name used by "pip install")
REQUIRED_PACKAGES = [("pandas", "pandas"), ("numpy", "numpy"), ("sklearn", "scikit-learn"),
                     ("imblearn", "imbalanced-learn"), ("openpyxl", "openpyxl"), ("joblib", "joblib")]

# The 31 final features, in the same order as the notebook (the order matters for identical results)
FINAL_FEATURES = [
    "SeniorCitizen", "Partner", "Dependents",                                   # demographic
    "TenureMonths", "InternetService", "OnlineSecurity", "OnlineBackup",        # usage
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges",          # billing
    "NumServices", "HasStreaming", "IsFiber", "NoInternet", "IsNewCustomer",    # engineered: usage
    "TenureGroup",
    "AvgChargePerMonth", "ChargeDelta", "IsMonthToMonth", "AutoPay",            # engineered: billing
    "ElectronicCheck",
    "HasProtectionBundle",                                                      # feedback proxy
    "MonthToMonth_x_New", "Fiber_x_MonthToMonth",                               # interactions
    "CustomerSegment",                                                          # K-Means segment
]
N_SEGMENTS = 4             # number of K-Means segments (chosen in the notebook, Section 3.3)
TARGET_RECALL = 0.80       # business rule for the decision threshold: catch at least 80% of churners

MODEL_NAMES = {"lr": "Logistic Regression", "dt": "Decision Tree", "rf": "Random Forest", "gb": "Gradient Boosting"}

# ---------------------------------------------------------------------------
# Logging: everything printed is also kept, so it can be saved to training_log.txt
# ---------------------------------------------------------------------------
LOG_LINES = []


def log(message=""):
    print(message, flush=True)
    LOG_LINES.append(str(message))


class Step:
    """Prints '[n/5] message...' at the start of a step and the time it took at the end."""

    def __init__(self, number, message):
        self.number, self.message = number, message

    def __enter__(self):
        log(f"\n[{self.number}/5] {self.message}...")
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:            # only report the time if the step finished normally
            log(f"      done in {time.perf_counter() - self.start:.1f} s")


def check_packages():
    # Stop with a friendly message if any required package is missing.
    missing = [pip_name for import_name, pip_name in REQUIRED_PACKAGES
               if importlib.util.find_spec(import_name) is None]
    if missing:
        print("These Python packages are missing: " + ", ".join(missing))
        print("Install them with:   python -m pip install -r requirements.txt")
        print("(or double-click train.bat, which installs everything automatically)")
        sys.exit(1)


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
def load_data(path):
    """Read the Excel (or CSV) file, clean it and return X (inputs) and y (1 = churn, 0 = stay)."""
    import pandas as pd
    from churn_features import clean_data

    path = Path(path)
    if not path.is_absolute():
        path = HERE / path
    if not path.exists():
        print(f"Data file not found. Looked for it here:\n    {path}")
        print("Put the file in that folder, or pass its location with --data PATH")
        sys.exit(1)

    raw = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)

    # clean_data() (from churn_features.py, shared with the notebook):
    #  - renames 'Tenure Months' -> 'TenureMonths' etc.
    #  - drops the ID, constant, location and leakage columns
    #    (Churn Score, Churn Reason, Churn Label, CLTV)
    #  - turns TotalCharges into numbers (the 11 blanks are tenure-0 customers -> 0)
    #  - normalises "No internet service" / "No phone service" to "No"
    #  - renames the target ChurnValue -> Churn
    data = clean_data(raw)
    if "Churn" not in data.columns:
        print("The data has no target column ('Churn Value'). Is this the right file?")
        sys.exit(1)

    missing = int(data.isna().sum().sum())
    if missing:
        log(f"      warning: {missing} missing values left after cleaning")

    X = data.drop(columns="Churn")
    y = data["Churn"].astype(int)
    log(f"      file: {path.name}  |  {len(X)} customers, {X.shape[1]} input columns, churn rate {y.mean():.1%}")
    return X, y


# ---------------------------------------------------------------------------
# 2. Pipeline
# ---------------------------------------------------------------------------
def build_pipeline(model, features=FINAL_FEATURES, use_smote=False):
    """Feature engineering -> K-Means segment -> encoding/scaling -> (SMOTE) -> model.

    There is no separate imputation step: after load_data() the data has no missing values
    (the only gaps, 11 blank TotalCharges, are filled during cleaning).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
    from churn_features import ADDON_SERVICES, SegmentAdder, add_features

    ordinal_categories = {"Contract": ["Month-to-month", "One year", "Two year"],
                          "TenureGroup": ["0-12", "13-24", "25-48", "49+"]}
    nominal_columns = ["Gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "InternetService",
                       "PaperlessBilling", "PaymentMethod", "CustomerSegment"] + ADDON_SERVICES

    nominal = [f for f in features if f in nominal_columns]
    ordinal = [f for f in features if f in ordinal_categories]
    numeric = [f for f in features if f not in nominal and f not in ordinal]
    preprocess = ColumnTransformer(
        transformers=[
            ("nominal", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), nominal),
            ("ordinal", OrdinalEncoder(categories=[ordinal_categories[f] for f in ordinal]), ordinal),
            ("numeric", StandardScaler(), numeric),
        ],
        remainder="drop",                # columns not in `features` (e.g. Gender) are left out
        verbose_feature_names_out=False,
    )

    steps = [("features", FunctionTransformer(add_features))]
    if "CustomerSegment" in features:
        steps.append(("segment", SegmentAdder(n_clusters=N_SEGMENTS, random_state=RANDOM_STATE)))
    steps.append(("preprocess", preprocess))
    if use_smote:
        # imblearn's Pipeline applies SMOTE during fit only, so validation/test data are never resampled
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        return ImbPipeline(steps + [("sampler", SMOTE(random_state=RANDOM_STATE)), ("model", model)])
    return Pipeline(steps + [("model", model)])


def model_setups(quick, use_smote):
    """The four models and their parameter grids (same grids as the notebook; smaller with --quick)."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    weight = None if use_smote else "balanced"      # SMOTE replaces class weights
    if quick:
        grids = {"lr": {"model__C": [0.1, 1]},
                 "dt": {"model__max_depth": [4, 5], "model__min_samples_leaf": [25]},
                 "rf": {"model__max_depth": [8], "model__min_samples_leaf": [5]},
                 "gb": {"model__n_estimators": [100], "model__learning_rate": [0.1], "model__max_depth": [2]}}
    else:
        grids = {"lr": {"model__C": [0.01, 0.03, 0.1, 0.3, 1, 3]},
                 "dt": {"model__max_depth": [3, 4, 5, 6], "model__min_samples_leaf": [10, 25, 50]},
                 "rf": {"model__max_depth": [4, 6, 8, None], "model__min_samples_leaf": [5, 10, 20]},
                 "gb": {"model__n_estimators": [100, 200], "model__learning_rate": [0.03, 0.1],
                        "model__max_depth": [2, 3]}}
    models = {
        "lr": LogisticRegression(class_weight=weight, max_iter=1000),
        "dt": DecisionTreeClassifier(class_weight=weight, random_state=RANDOM_STATE),
        "rf": RandomForestClassifier(n_estimators=100 if quick else 300, class_weight=weight,
                                     random_state=RANDOM_STATE, n_jobs=-1),
        "gb": GradientBoostingClassifier(subsample=0.8, random_state=RANDOM_STATE),
    }
    return {key: (models[key], grids[key]) for key in models}


# ---------------------------------------------------------------------------
# 3. Training, model selection and threshold tuning
# ---------------------------------------------------------------------------
def train_and_select(X, y, model_keys, quick, use_smote):
    """Split, tune each model with GridSearchCV, pick the best one and tune its threshold."""
    import numpy as np
    from sklearn.metrics import f1_score, precision_recall_curve
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
    from sklearn.utils.class_weight import compute_sample_weight

    with Step(2, "Splitting the data (stratified 80/20)"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
        n_folds = 3 if quick else 5
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        log(f"      train {len(X_train)} / test {len(X_test)} customers, {n_folds}-fold cross-validation"
            f"{' (quick mode)' if quick else ''}, imbalance handled with "
            f"{'SMOTE' if use_smote else 'class_weight=balanced'}")

    def fit_params(key):
        # Gradient Boosting has no class_weight option, so it gets balanced sample weights instead
        if key == "gb" and not use_smote:
            return {"model__sample_weight": compute_sample_weight("balanced", y_train)}
        return {}

    setups = model_setups(quick, use_smote)
    results = {}
    for i, key in enumerate(model_keys, start=1):
        model, grid = setups[key]
        name = MODEL_NAMES[key]
        with Step(3, f"Training {name} ({i}/{len(model_keys)})"):
            search = GridSearchCV(build_pipeline(model, use_smote=use_smote), grid, cv=cv,
                                  scoring=["f1", "roc_auc", "recall"], refit="f1")
            search.fit(X_train, y_train, **fit_params(key))
            best = search.best_index_
            results[key] = {
                "name": name,
                "pipeline": search.best_estimator_,
                "cv_f1_mean": search.cv_results_["mean_test_f1"][best],
                "cv_f1_std": search.cv_results_["std_test_f1"][best],
                "cv_auc_mean": search.cv_results_["mean_test_roc_auc"][best],
                "cv_auc_std": search.cv_results_["std_test_roc_auc"][best],
                "params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
            }
            r = results[key]
            log(f"      CV ROC-AUC {r['cv_auc_mean']:.3f} ± {r['cv_auc_std']:.3f} | "
                f"CV F1 {r['cv_f1_mean']:.3f} ± {r['cv_f1_std']:.3f} | best parameters {r['params']}")

    with Step(4, "Selecting the best model and tuning its decision threshold"):
        # Rule: highest CV ROC-AUC wins. Models whose ROC-AUC is within one standard deviation of
        # the leader count as a tie ("within noise"); among those, the highest CV F1 wins.
        leader = max(results, key=lambda k: results[k]["cv_auc_mean"])
        noise = results[leader]["cv_auc_std"]
        tied = [k for k in results if results[leader]["cv_auc_mean"] - results[k]["cv_auc_mean"] <= noise]
        best_key = max(tied, key=lambda k: results[k]["cv_f1_mean"])
        log(f"      highest CV ROC-AUC: {results[leader]['name']} ({results[leader]['cv_auc_mean']:.3f}, "
            f"std {noise:.3f})")
        log(f"      within noise of it: {', '.join(results[k]['name'] for k in tied)}")
        log(f"      -> selected {results[best_key]['name']} (highest CV F1 among those: "
            f"{results[best_key]['cv_f1_mean']:.3f})")

        # Threshold: out-of-fold probabilities on the TRAINING set only (the test set stays untouched)
        best_pipeline = results[best_key]["pipeline"]
        oof = cross_val_predict(best_pipeline, X_train, y_train, cv=cv, method="predict_proba",
                                params=fit_params(best_key))[:, 1]
        _, recall, thresholds = precision_recall_curve(y_train, oof)
        recall = recall[:-1]                                   # align with `thresholds`
        # Highest threshold that still catches at least 80% of churners, rounded down to 2 decimals
        threshold = float(np.floor(thresholds[recall >= TARGET_RECALL].max() * 100) / 100)
        fold_f1 = [f1_score(y_train.iloc[val], (oof[val] >= threshold).astype(int))
                   for _, val in cv.split(X_train, y_train)]
        log(f"      decision threshold: {threshold:.2f} (highest threshold with out-of-fold recall >= "
            f"{TARGET_RECALL:.0%})")

    return {"results": results, "best_key": best_key, "threshold": threshold,
            "tuned_cv_f1": (float(np.mean(fold_f1)), float(np.std(fold_f1))),
            "X_test": X_test, "y_test": y_test}


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------
def test_scores(pipeline, X, y, threshold):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    probability = pipeline.predict_proba(X)[:, 1]
    prediction = (probability >= threshold).astype(int)
    return {"Test Accuracy": accuracy_score(y, prediction), "Test Precision": precision_score(y, prediction),
            "Test Recall": recall_score(y, prediction), "Test F1": f1_score(y, prediction),
            "Test ROC-AUC": roc_auc_score(y, probability)}


def evaluate(run):
    """Comparison table (all models at 0.50 + the selected model at its tuned threshold) and confusion matrix."""
    import pandas as pd
    from sklearn.metrics import confusion_matrix

    results, best_key, threshold = run["results"], run["best_key"], run["threshold"]
    X_test, y_test = run["X_test"], run["y_test"]

    rows = {}
    for r in results.values():
        rows[r["name"]] = {"Threshold": 0.50,
                           "CV F1 (mean ± std)": f"{r['cv_f1_mean']:.3f} ± {r['cv_f1_std']:.3f}",
                           "CV ROC-AUC (mean ± std)": f"{r['cv_auc_mean']:.3f} ± {r['cv_auc_std']:.3f}",
                           **test_scores(r["pipeline"], X_test, y_test, 0.50)}
    best = results[best_key]
    f1_mean, f1_std = run["tuned_cv_f1"]
    rows[f"{best['name']} @ threshold {threshold:.2f}"] = {
        "Threshold": threshold,
        "CV F1 (mean ± std)": f"{f1_mean:.3f} ± {f1_std:.3f}",
        "CV ROC-AUC (mean ± std)": f"{best['cv_auc_mean']:.3f} ± {best['cv_auc_std']:.3f}",
        **test_scores(best["pipeline"], X_test, y_test, threshold)}

    table = pd.DataFrame(rows).T.rename_axis("Model")
    for col in [c for c in table.columns if c.startswith("Test")] + ["Threshold"]:
        table[col] = table[col].astype(float).round(3)

    log("\nComparison table (CV = training folds, Test = held-out 20%):")
    log(table.to_string())

    if table.filter(like="ROC-AUC").columns.size and table["Test ROC-AUC"].max() > 0.95:
        log("\nWARNING: a test ROC-AUC above 0.95 is suspicious for this data - check for leakage!")

    prediction = (best["pipeline"].predict_proba(X_test)[:, 1] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
    log(f"\nConfusion matrix - {best['name']} @ threshold {threshold:.2f} (test set):")
    log("                    predicted Stay   predicted Churn")
    log(f"    actual Stay     {tn:>14}   {fp:>15}")
    log(f"    actual Churn    {fn:>14}   {tp:>15}")
    log(f"    -> {tp} of {tp + fn} churners caught, {fp} false alarms among {tp + fp} flagged customers")
    return table


# ---------------------------------------------------------------------------
# 5. Saving
# ---------------------------------------------------------------------------
def save(run, table):
    import joblib

    OUTPUT_DIR.mkdir(exist_ok=True)
    best = run["results"][run["best_key"]]
    model_path = OUTPUT_DIR / "churn_model.pkl"
    joblib.dump({"pipeline": best["pipeline"], "threshold": run["threshold"], "features": FINAL_FEATURES,
                 "model_name": best["name"]}, model_path)
    table.to_csv(OUTPUT_DIR / "results.csv", encoding="utf-8-sig")     # utf-8-sig: opens cleanly in Excel
    log(f"      saved {model_path.relative_to(HERE)}")
    log(f"      saved {(OUTPUT_DIR / 'results.csv').relative_to(HERE)}")
    log(f"      saved {(OUTPUT_DIR / 'training_log.txt').relative_to(HERE)} (written at the very end)")


def write_log():
    # Save everything printed so far, with a timestamp and the command that was used
    header = [f"Churn model training log - {datetime.now():%Y-%m-%d %H:%M:%S}",
              f"Command: python train.py {' '.join(sys.argv[1:])}".rstrip(), ""]
    (OUTPUT_DIR / "training_log.txt").write_text("\n".join(header + LOG_LINES) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Prediction helper (used after training; not called by main)
# ---------------------------------------------------------------------------
def predict_churn(customer_dict, model_path=OUTPUT_DIR / "churn_model.pkl"):
    """Predict churn for one customer given as a dict of the input columns (CamelCase or Excel names)."""
    import joblib
    import pandas as pd
    from churn_features import clean_data

    saved = joblib.load(model_path)
    customer = clean_data(pd.DataFrame([customer_dict]))
    probability = float(saved["pipeline"].predict_proba(customer)[0, 1])
    return {"churn_probability": round(probability, 3),
            "prediction": "Churn" if probability >= saved["threshold"] else "Stay",
            "threshold_used": saved["threshold"], "model": saved.get("model_name", "")}


# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train the telecom churn model (see churn_prediction.ipynb for the explanation).")
    parser.add_argument("--data", default="Telco_customer_churn.xlsx",
                        help="path to the data file, .xlsx or .csv (default: Telco_customer_churn.xlsx)")
    parser.add_argument("--quick", action="store_true",
                        help="fast test run: smaller grids and 3-fold cross-validation")
    parser.add_argument("--model", choices=["lr", "dt", "rf", "gb", "all"], default="all",
                        help="which model(s) to train (default: all)")
    smote = parser.add_mutually_exclusive_group()
    smote.add_argument("--smote", dest="smote", action="store_true",
                       help="balance the classes with SMOTE instead of class weights")
    smote.add_argument("--no-smote", dest="smote", action="store_false",
                       help="balance the classes with class_weight='balanced' (default)")
    parser.set_defaults(smote=False)
    return parser.parse_args()


def main():
    args = parse_args()
    check_packages()
    sys.path.insert(0, str(HERE))         # so churn_features.py is found even when run from elsewhere
    started = time.perf_counter()

    log("=" * 70)
    log(f"Telecom churn model training - {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"options: data={args.data}  quick={args.quick}  model={args.model}  smote={args.smote}")
    log("=" * 70)

    with Step(1, "Loading data"):
        X, y = load_data(args.data)

    model_keys = list(MODEL_NAMES) if args.model == "all" else [args.model]
    run = train_and_select(X, y, model_keys, args.quick, args.smote)

    with Step(5, "Evaluating on the test set and saving the outputs"):
        table = evaluate(run)
        save(run, table)

    log(f"\nTotal time: {time.perf_counter() - started:.1f} s")
    write_log()


def pause_if_double_clicked():
    # train.exe opens its own console window when double-clicked; keep it open so the results can be read
    if FROZEN and sys.stdin is not None and sys.stdin.isatty():
        try:
            input("\nPress Enter to close this window...")
        except (EOFError, KeyboardInterrupt):          # no keyboard attached, or Ctrl+C: just close
            pass


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()      # needed for parallel workers inside train.exe; harmless otherwise
    try:
        main()
    finally:
        pause_if_double_clicked()
