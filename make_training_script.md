# Create a standalone training script and a double-click launcher

Goal: I want to train the churn model without opening Jupyter, either by double-clicking a file
or by running one command. The notebook stays the place for the explanation; the script is for
training.

## 1. `train.py`: the training script
Move the training logic from churn_prediction.ipynb into a clean Python script. It must give the
same results as the notebook (same features, pipeline, models, grids, random_state=42).

Structure:
- `load_data(path)`: reads `Telco_customer_churn.xlsx` (or a CSV if one is given), renames the
  columns, fixes TotalCharges, drops the ID, constant, location and leakage columns
  (Churn Score, Churn Reason, Churn Label, CLTV), and returns X, y.
- Import the feature-engineering transformers from `churn_features.py`; don't duplicate them.
- `build_pipeline(model)`: imputation, feature engineering, K-Means segment, encoding, scaling, model.
- `train_and_select()`: stratified 80/20 split, 5-fold GridSearchCV for Logistic Regression,
  Decision Tree, Random Forest and Gradient Boosting, then picks the best model by CV ROC-AUC
  (use F1 to break ties within noise), then tunes the decision threshold on training folds only.
- `evaluate()`: prints a clean comparison table (CV mean ± std, then test Accuracy, Precision,
  Recall, F1 and ROC-AUC) plus the confusion matrix of the final model.
- `save()`: writes `churn_model.pkl` (pipeline + threshold + feature list), `results.csv`
  (the comparison table) and `training_log.txt` (everything printed, with a timestamp).
  Put the outputs in an `outputs/` folder.

Command-line options (argparse):
- `--data PATH` (default: Telco_customer_churn.xlsx)
- `--quick`: smaller grids and 3-fold CV, for a fast test run (under a minute)
- `--model {lr,dt,rf,gb,all}` (default: all)
- `--no-smote` / `--smote`

Behaviour:
- Print clear progress messages ("[1/5] Loading data...", "[3/5] Training Random Forest...")
  and the time taken by each step.
- Check that the required packages are installed at start-up and print a friendly message
  naming any that are missing.
- If the data file is missing, print exactly where it was looked for, then exit.
- Use `if __name__ == "__main__": main()`.
- Add comments a 2nd-year student can follow.

## 2. `requirements.txt`
List pandas, numpy, scikit-learn, imbalanced-learn, openpyxl, joblib, matplotlib and seaborn,
pinned to the versions currently installed.

## 3. `train.bat`: double-click launcher for Windows
- Changes to the folder the .bat file is in (`cd /d "%~dp0"`).
- Checks that Python is installed, and prints a message if it isn't.
- Creates a virtual environment `.venv` the first time, then installs requirements.txt into it.
  Later runs reuse it.
- Runs `python train.py %*`, so options like `--quick` can be passed.
- Ends with `pause` so the window stays open to show the results.

Also add `train_quick.bat`, which runs `train.bat --quick`.

## 4. Optional: a real .exe
Ask me before doing this; don't do it automatically. If I say yes, use PyInstaller
(`pyinstaller --onefile train.py`, including churn_features.py) to make `dist/train.exe`, and
tell me its size. Expect it to be large (~200–400 MB) because of scikit-learn.

## 5. Test it
- Run `python train.py --quick` and then `python train.py`, and confirm both finish without errors.
- Confirm that the test ROC-AUC from the full run matches the notebook's.
- Load `outputs/churn_model.pkl` in a fresh Python process and run `predict_churn()` on one
  example customer.
- Add a "How to train" section to README.md that explains double-clicking train.bat and running
  `python train.py --quick` from a terminal.

Finish by telling me the files created, how long the full and quick runs took, and the final
results table.
