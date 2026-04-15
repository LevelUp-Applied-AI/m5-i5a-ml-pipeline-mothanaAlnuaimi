"""
Module 5 Week A — Integration: ML Evaluation Pipeline

Build a structured evaluation pipeline that compares 5 model
configurations using cross-validation with ColumnTransformer + Pipeline.
"""

import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_and_prepare(filepath="data/telecom_churn.csv"):
    df = pd.read_csv(filepath)

    X = df.drop("churned", axis=1)
    y = df["churned"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
        ]
    )
    return preprocessor


def define_models(preprocessor):
    models = {
        "LogReg (default)": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                C=1.0,
                random_state=42,
                max_iter=1000,
                class_weight="balanced"
            ))
        ]),

        "LogReg (L1, C=0.1)": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                C=0.1,
                penalty="l1",
                solver="saga",
                random_state=42,
                max_iter=1000,
                class_weight="balanced"
            ))
        ]),

        "RidgeClassifier": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RidgeClassifier(
                alpha=1.0,
                class_weight="balanced",
                random_state=42
            ))
        ]),

        "Most-frequent Dummy": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DummyClassifier(strategy="most_frequent"))
        ]),

        "Stratified Dummy": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DummyClassifier(strategy="stratified", random_state=42))
        ])
    }

    return models


def evaluate_models(models, X_train, y_train):
    results = []

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, pipeline in models.items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "f1"]
        )

        results.append({
            "Model": model_name,
            "Mean Accuracy": scores["test_accuracy"].mean(),
            "Std Accuracy": scores["test_accuracy"].std(),
            "Mean Precision": scores["test_precision"].mean(),
            "Mean Recall": scores["test_recall"].mean(),
            "Mean F1": scores["test_f1"].mean()
        })

    return pd.DataFrame(results)


def final_evaluation(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0)
    }

    return metrics


def recommend_model(results_df):
    print("\n=== Model Comparison Table (CV results) ===")
    print(results_df.to_string(index=False))
    print("\n=== Recommendation ===")
    print("Write your recommendation in the PR description.")


if __name__ == "__main__":
    file_path = "data/telecom_churn.csv"

    X_train, X_test, y_train, y_test, numeric_features, categorical_features = load_and_prepare(file_path)

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    models = define_models(preprocessor)

    results_df = evaluate_models(models, X_train, y_train)

    print("\nCross-Validation Results:")
    print(results_df.to_string(index=False))

    real_models_df = results_df[~results_df["Model"].str.contains("Dummy")]
    best_model_name = real_models_df.sort_values(by="Mean F1", ascending=False).iloc[0]["Model"]

    print(f"\nBest real model based on Mean F1: {best_model_name}")

    best_pipeline = models[best_model_name]
    test_metrics = final_evaluation(best_pipeline, X_train, X_test, y_train, y_test)

    print("\nFinal Test Set Evaluation:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")