"""Cleaning and feature-engineering steps for the churn model (IBM Telco dataset).

Kept in a .py file so that the saved model (churn_model.pkl) can be loaded
outside the notebook: joblib needs to import these classes when loading.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Columns removed before modelling (the notebook explains each one)
DROP_COLUMNS = ["CustomerID", "Count", "Country", "State", "City", "ZipCode", "LatLong", "Latitude",
                "Longitude", "ChurnLabel", "ChurnScore", "ChurnReason", "CLTV"]
ADDON_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                  "StreamingTV", "StreamingMovies", "MultipleLines"]
SEGMENT_FEATURES = ["TenureMonths", "MonthlyCharges", "NumServices"]


def to_camel_case(column):
    """'Tenure Months' -> 'TenureMonths'. Names that are already CamelCase stay the same."""
    return "".join(word[0].upper() + word[1:] for word in str(column).split())


def clean_data(raw):
    """Rule-based cleaning. Nothing is learned from the data, so it is safe to run before the split."""
    data = raw.rename(columns=to_camel_case)
    data = data.drop(columns=DROP_COLUMNS, errors="ignore")

    # TotalCharges is stored as text with blanks for brand-new customers.
    # A blank is rebuilt as tenure x monthly charge, which is 0 for a customer with tenure 0.
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"] = data["TotalCharges"].fillna(data["MonthlyCharges"] * data["TenureMonths"])

    # "No internet service" / "No phone service" mean the same as "No" for an add-on
    for col in ADDON_SERVICES:
        data[col] = data[col].replace({"No internet service": "No", "No phone service": "No"})

    # Target: ChurnValue is already 1 = churn, 0 = stay
    if "ChurnValue" in data.columns:
        data = data.rename(columns={"ChurnValue": "Churn"})
    return data


def add_features(X):
    """Creates the engineered features from the cleaned columns."""
    X = X.copy()
    tenure = X["TenureMonths"].clip(lower=1)          # avoids dividing by zero for new customers

    # Billing
    X["AvgChargePerMonth"] = np.where(X["TenureMonths"] > 0, X["TotalCharges"] / tenure, X["MonthlyCharges"])
    X["ChargeDelta"] = X["MonthlyCharges"] - X["AvgChargePerMonth"]
    X["IsMonthToMonth"] = (X["Contract"] == "Month-to-month").astype(int)
    X["AutoPay"] = X["PaymentMethod"].str.contains("automatic").astype(int)
    X["ElectronicCheck"] = (X["PaymentMethod"] == "Electronic check").astype(int)

    # Usage
    X["NumServices"] = (X[ADDON_SERVICES] == "Yes").sum(axis=1)
    X["HasStreaming"] = ((X["StreamingTV"] == "Yes") | (X["StreamingMovies"] == "Yes")).astype(int)
    X["IsFiber"] = (X["InternetService"] == "Fiber optic").astype(int)
    X["NoInternet"] = (X["InternetService"] == "No").astype(int)
    X["IsNewCustomer"] = (X["TenureMonths"] <= 6).astype(int)
    X["TenureGroup"] = pd.cut(X["TenureMonths"], bins=[-1, 12, 24, 48, np.inf],
                              labels=["0-12", "13-24", "25-48", "49+"]).astype(str)

    # Service-quality proxy (the dataset has no feedback column)
    X["HasProtectionBundle"] = ((X["OnlineSecurity"] == "Yes") | (X["TechSupport"] == "Yes")).astype(int)

    # Interactions
    X["MonthToMonth_x_New"] = X["IsMonthToMonth"] * X["IsNewCustomer"]
    X["Fiber_x_MonthToMonth"] = X["IsFiber"] * X["IsMonthToMonth"]
    return X


class SegmentAdder(BaseEstimator, TransformerMixin):
    """Adds a K-Means customer segment (S0, S1, ...) built from usage and billing features.

    The scaler and the K-Means centres are learned in fit(), i.e. from the training data only.
    """

    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X[SEGMENT_FEATURES])
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        self.kmeans_.fit(self.scaler_.transform(X[SEGMENT_FEATURES]))
        return self

    def transform(self, X):
        X = X.copy()
        labels = self.kmeans_.predict(self.scaler_.transform(X[SEGMENT_FEATURES]))
        X["CustomerSegment"] = ["S" + str(label) for label in labels]
        return X
