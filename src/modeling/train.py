from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.pipeline import Pipeline

from src.modeling.evaluation import (
    calculate_regression_metrics,
    create_prediction_frame,
)
from src.modeling.models import ModelName, get_model
from src.modeling.preprocessing import build_preprocessor


@dataclass
class TrainingResult:
    model_name: str
    feature_set: str
    target: str
    pipeline: Any
    validation_metrics: dict[str, float]
    validation_predictions: pd.DataFrame
    training_time_seconds: float
    best_iteration: int | None


def train_standard_model(
    model_name: ModelName,
    feature_set: str,
    target: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    val_listing_ids: pd.Series,
    random_seed: int = 42,
) -> TrainingResult:
    """
    Train Dummy, Ridge or Random Forest with a
    standard sklearn Pipeline.
    """

    if model_name == "xgboost":
        raise ValueError(
            "Use train_xgboost_model() for XGBoost."
        )

    scale_numeric = model_name == "ridge"

    preprocessor = build_preprocessor(
        X_train,
        scale_numeric=scale_numeric,
    )

    model = get_model(
        model_name,
        random_seed=random_seed,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    start_time = perf_counter()

    pipeline.fit(
        X_train,
        y_train,
    )

    training_time = perf_counter() - start_time

    val_predictions = pipeline.predict(X_val)

    metrics = calculate_regression_metrics(
        y_val,
        val_predictions,
        target=target,
    )

    prediction_frame = create_prediction_frame(
        listing_ids=val_listing_ids,
        y_true=y_val,
        y_pred=val_predictions,
        target=target,
    )

    return TrainingResult(
        model_name=model_name,
        feature_set=feature_set,
        target=target,
        pipeline=pipeline,
        validation_metrics=metrics,
        validation_predictions=prediction_frame,
        training_time_seconds=training_time,
        best_iteration=None,
    )


def train_xgboost_model(
    feature_set: str,
    target: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    val_listing_ids: pd.Series,
    random_seed: int = 42,
) -> TrainingResult:
    """
    Train XGBoost using validation-based early stopping.

    The preprocessor is fitted only on training data.
    Validation data is transformed using the fitted
    preprocessor.
    """

    preprocessor = build_preprocessor(
        X_train,
        scale_numeric=False,
    )

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_val_processed = preprocessor.transform(
        X_val
    )

    model = get_model(
        "xgboost",
        random_seed=random_seed,
    )

    start_time = perf_counter()

    model.fit(
        X_train_processed,
        y_train,
        eval_set=[
            (
                X_val_processed,
                y_val,
            )
        ],
        verbose=False,
    )

    training_time = perf_counter() - start_time

    val_predictions = model.predict(
        X_val_processed
    )

    metrics = calculate_regression_metrics(
        y_val,
        val_predictions,
        target=target,
    )

    prediction_frame = create_prediction_frame(
        listing_ids=val_listing_ids,
        y_true=y_val,
        y_pred=val_predictions,
        target=target,
    )

    fitted_model = {
        "preprocessor": preprocessor,
        "model": model,
    }

    best_iteration = getattr(
        model,
        "best_iteration",
        None,
    )

    return TrainingResult(
        model_name="xgboost",
        feature_set=feature_set,
        target=target,
        pipeline=fitted_model,
        validation_metrics=metrics,
        validation_predictions=prediction_frame,
        training_time_seconds=training_time,
        best_iteration=best_iteration,
    )


def train_model(
    model_name: ModelName,
    feature_set: str,
    target: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    val_listing_ids: pd.Series,
    random_seed: int = 42,
) -> TrainingResult:
    """
    Common public training function.
    """

    if model_name == "xgboost":
        return train_xgboost_model(
            feature_set=feature_set,
            target=target,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            val_listing_ids=val_listing_ids,
            random_seed=random_seed,
        )

    return train_standard_model(
        model_name=model_name,
        feature_set=feature_set,
        target=target,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        val_listing_ids=val_listing_ids,
        random_seed=random_seed,
    )