from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def calculate_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    target: str,
) -> dict[str, float]:
    """
    Calculate regression metrics.

    If target is log_price, also convert predictions
    back to euros using expm1.
    """

    y_true_array = np.asarray(y_true).reshape(-1)
    y_pred_array = np.asarray(y_pred).reshape(-1)

    if len(y_true_array) != len(y_pred_array):
        raise ValueError(
            "y_true and y_pred lengths differ."
        )

    metrics = {
        "mae": mean_absolute_error(
            y_true_array,
            y_pred_array,
        ),
        "rmse": np.sqrt(
            mean_squared_error(
                y_true_array,
                y_pred_array,
            )
        ),
        "r2": r2_score(
            y_true_array,
            y_pred_array,
        ),
    }

    if target == "log_price":
        true_price = np.expm1(y_true_array)
        predicted_price = np.expm1(y_pred_array)

        # Negative euro predictions are not meaningful
        predicted_price = np.clip(
            predicted_price,
            a_min=0,
            a_max=None,
        )

        metrics.update(
            {
                "mae_eur": mean_absolute_error(
                    true_price,
                    predicted_price,
                ),
                "rmse_eur": np.sqrt(
                    mean_squared_error(
                        true_price,
                        predicted_price,
                    )
                ),
            }
        )

    return {
        key: float(value)
        for key, value in metrics.items()
    }


def create_prediction_frame(
    listing_ids: pd.Series,
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    target: str,
) -> pd.DataFrame:
    """
    Create a table for later residual/error analysis.
    """

    prediction_df = pd.DataFrame(
        {
            "listing_id": np.asarray(listing_ids),
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
        }
    )

    prediction_df["residual"] = (
        prediction_df["y_true"]
        - prediction_df["y_pred"]
    )

    prediction_df["absolute_error"] = (
        prediction_df["residual"].abs()
    )

    if target == "log_price":
        prediction_df["true_price_eur"] = np.expm1(
            prediction_df["y_true"]
        )

        prediction_df["predicted_price_eur"] = np.clip(
            np.expm1(prediction_df["y_pred"]),
            a_min=0,
            a_max=None,
        )

        prediction_df["absolute_error_eur"] = (
            prediction_df["true_price_eur"]
            - prediction_df["predicted_price_eur"]
        ).abs()

    return prediction_df