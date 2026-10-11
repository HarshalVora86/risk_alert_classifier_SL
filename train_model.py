"""
Train and export the complete preprocessing + classification pipeline.

Run:
    python train_model.py

Output:
    model/risk_alert_pipeline.pkl
    model/metrics.json
"""
import json
import pickle
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "dataset/Risk_Alert_Classifier_Dataset_4600.csv"
TARGET = "risk_status"

CATEGORICAL = ["gender", "region", "employment_type"]
NUMERIC = [
    "age",
    "annual_income_inr",
    "credit_score",
    "credit_utilization_ratio",
    "missed_payments_12m",
    "avg_late_payment_days",
    "monthly_transaction_count",
    "monthly_spend_inr",
    "cash_advance_count_6m",
    "complaints_last_6m",
    "failed_login_attempts_3m",
    "account_tenure_months",
    "debt_balance_inr",
    "last_txn_days_ago",
]
FEATURES = CATEGORICAL + NUMERIC


def main():
    df = pd.read_csv(DATA_PATH)

    # fixed reference date -> same result on every run
    df["last_transaction_date"] = pd.to_datetime(df["last_transaction_date"])
    ref_date = df["last_transaction_date"].max() + pd.Timedelta(days=1)
    df["last_txn_days_ago"] = (ref_date - df["last_transaction_date"]).dt.days

    X = df[FEATURES]
    y = df[TARGET]

    # stratify keeps the same high-risk share in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC),
            ("categorical", categorical_pipeline, CATEGORICAL),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=50)),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred)),
        "F1": float(f1_score(y_test, y_pred)),
        "AUC_ROC": float(roc_auc_score(y_test, y_prob)),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
    }

    Path("model").mkdir(exist_ok=True)

    with open("model/risk_alert_pipeline.pkl", "wb") as file:
        pickle.dump(pipeline, file)

    with open("model/metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nModel trained successfully.")
    print("Pipeline: model/risk_alert_pipeline.pkl")
    print("\nEvaluation:")
    for key, value in metrics.items():
        if key != "features":
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
