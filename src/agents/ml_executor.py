"""
Machine Learning Executor.

Prepares datasets, preprocesses features,
trains machine learning models from the
Model Registry, and selects the best model.
"""

from typing import Any

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from src.agents.semantic_analyzer import (
    analyze_semantics,
)

from src.tools.ml_models import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
)


# --------------------------------------------------
# Dataset Preparation
# --------------------------------------------------

def _prepare_dataset(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Automatically determine the target column.
    """

    print("\n========== ML EXECUTOR ==========")
    print("Columns received:")
    print(dataframe.columns.tolist())

    semantic = analyze_semantics(
        dataframe=dataframe,
        use_llm=False,
    )

    print("\nDetected Targets:")
    print(semantic["target_candidates"])

    targets = semantic["target_candidates"]

    if not targets:
        raise ValueError(
            "No target column detected."
        )

    target = targets[0]

    feature_columns = [
        column
        for column in dataframe.columns
        if column != target
    ]

    X = dataframe[feature_columns]
    y = dataframe[target]

    return X, y, target


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

def _build_preprocessing_pipeline(
    X: pd.DataFrame,
) -> ColumnTransformer:
    """
    Automatically build preprocessing pipeline.
    """

    numeric_columns = (
        X.select_dtypes(
            include=["number"]
        ).columns.tolist()
    )

    categorical_columns = (
        X.select_dtypes(
            exclude=["number"]
        ).columns.tolist()
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ]
    )


# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------

def _split_dataset(
    X,
    y,
):
    """
    Split the dataset.
    """

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )


# --------------------------------------------------
# Model Training
# --------------------------------------------------

def _train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    model_names: list[str],
    model_registry: dict[str, Any],
) -> dict[str, Any]:
    """
    Train all requested models.
    """

    results = {}

    for model_name in model_names:

     if model_name not in model_registry:
        continue

    model = model_registry[model_name]

    try:

        print(f"\nTraining model: {model_name}")

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test,
        )

    except Exception as error:

        print(f"\n❌ Model '{model_name}' failed")

        print(type(error).__name__)

        print(error)

        raise

    results[model_name] = {
        "model": model,
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
    }
    return results


# --------------------------------------------------
# Best Model Selection
# --------------------------------------------------

def _select_best_model(
    results: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """
    Select best model using accuracy.
    """

    best_model = max(
        results,
        key=lambda name: results[name]["accuracy"],
    )

    return (
        best_model,
        results[best_model],
    )


# --------------------------------------------------
# Executor
# --------------------------------------------------

def execute_ml_plan(
    dataframe: pd.DataFrame,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute the machine learning plan.
    """

    X, y, target = _prepare_dataset(
        dataframe
    )

    preprocessor = _build_preprocessing_pipeline(
        X
    )

    X_processed = preprocessor.fit_transform(
        X
    )
# -----------------------------------------
# Convert sparse matrix to dense
# -----------------------------------------

    if hasattr(X_processed, "toarray"):
      X_processed = X_processed.toarray()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = _split_dataset(
        X_processed,
        y,
    )

    problem_type = plan["problem_type"]

    if problem_type == "classification":

        registry = CLASSIFICATION_MODELS

    elif problem_type == "regression":

        registry = REGRESSION_MODELS

    else:

        raise ValueError(
            f"Unsupported problem type: {problem_type}"
        )

    results = _train_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model_names=plan["recommended_models"],
        model_registry=registry,
    )

    if not results:
        raise ValueError(
            "No models were successfully trained."
        )

    best_model_name, best_model = (
        _select_best_model(
            results
        )
    )

    return {
        "problem_type": problem_type,
        "target": target,
        "feature_columns": list(X.columns),
        "processed_features": X_processed.shape[1],
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "best_model": best_model_name,
        "accuracy": round(
            best_model["accuracy"],
            4,
        ),
        "precision": round(
            best_model["precision"],
            4,
        ),
        "recall": round(
            best_model["recall"],
            4,
        ),
        "f1_score": round(
            best_model["f1"],
            4,
        ),
    }


__all__ = [
    "execute_ml_plan",
]