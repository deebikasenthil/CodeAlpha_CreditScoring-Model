"""
CodeAlpha Machine Learning Internship - Task 1
Credit Scoring Model (beginner-friendly educational project)

Dataset: UCI "Default of Credit Card Clients" (ID 350)
This project compares Logistic Regression and Random Forest.
It is for learning only; it must NOT be used to make real lending decisions.
"""

import joblib
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_dataset():
    """Download the UCI dataset and return features (X) and target (y)."""
    print("Downloading the UCI credit-card default dataset...")
    dataset = fetch_ucirepo(id=350)

    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0].copy()

    # Remove the row ID column if it exists; it does not describe credit risk.
    id_columns = [col for col in X.columns if col.strip().lower() in ("id", "unnamed: 0")]
    X = X.drop(columns=id_columns, errors="ignore")

    # Convert the target to integer labels: 1 = default, 0 = no default.
    y = pd.to_numeric(y, errors="coerce")
    valid_rows = y.notna()
    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].astype(int)

    return X, y


def add_features(X):
    """Create two simple features from existing financial columns."""
    X = X.copy()

    # Credit utilisation = latest bill amount divided by credit limit.
    if {"BILL_AMT1", "LIMIT_BAL"}.issubset(X.columns):
        limit = X["LIMIT_BAL"].replace(0, pd.NA)
        X["credit_utilization"] = X["BILL_AMT1"] / limit

    # Payment-to-bill ratio gives a simple comparison of payment and bill.
    if {"PAY_AMT1", "BILL_AMT1"}.issubset(X.columns):
        bill = X["BILL_AMT1"].abs().replace(0, pd.NA)
        X["payment_to_bill_ratio"] = X["PAY_AMT1"] / bill

    return X


def build_preprocessor(X):
    """Prepare numeric and categorical columns in different ways."""
    categorical_columns = [
        col for col in ["SEX", "EDUCATION", "MARRIAGE"] if col in X.columns
    ]
    numeric_columns = [col for col in X.columns if col not in categorical_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("fill_missing", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("fill_missing", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    return preprocessor


def evaluate_model(name, model, X_test, y_test):
    """Print the main classification metrics."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"\n{'=' * 55}")
    print(name)
    print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_test, predictions, zero_division=0):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, probabilities):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    return f1_score(y_test, predictions, zero_division=0)


def main():
    X, y = load_dataset()
    X = add_features(X)

    print("\nDataset shape:", X.shape)
    print("Target counts (0 = no default, 1 = default):")
    print(y.value_counts().sort_index())

    # Keep some data aside for an honest final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(X)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    best_f1 = -1
    best_model = None
    best_name = ""

    for name, classifier in models.items():
        # A Pipeline applies preprocessing first, then trains the classifier.
        full_model = Pipeline(
            steps=[
                ("preprocessing", build_preprocessor(X_train)),
                ("classifier", classifier),
            ]
        )
        full_model.fit(X_train, y_train)
        score = evaluate_model(name, full_model, X_test, y_test)

        if score > best_f1:
            best_f1 = score
            best_model = full_model
            best_name = name

    joblib.dump(best_model, "credit_scoring_model.joblib")
    print(f"\nBest model by F1-score: {best_name}")
    print("Saved as: credit_scoring_model.joblib")

    # Example prediction using one row from the test set.
    example = X_test.iloc[[0]]
    predicted_class = best_model.predict(example)[0]
    default_probability = best_model.predict_proba(example)[0, 1]

    print("\nExample prediction (one held-out record):")
    print("Predicted class:", int(predicted_class))
    print(f"Estimated model probability of default: {default_probability:.2%}")
    print("This is a model output for coursework, not a real credit decision.")


if __name__ == "__main__":
    main()
