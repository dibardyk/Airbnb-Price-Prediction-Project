from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from src.modeling.model_data import (
    create_xy,
    load_model_ready_data,
)
from src.modeling.train import train_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "output"
    / "results"
)

PREDICTIONS_DIR = (
    PROJECT_ROOT
    / "output"
    / "predictions"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "output"
    / "models"
)


TARGET = "log_price"
RANDOM_SEED = 42


EXPERIMENTS = [
    # Basic benchmark
    ("dummy", "tabular"),

    # Compare algorithms on the same tabular data
    ("ridge", "tabular"),
    ("random_forest", "tabular"),
    ("xgboost", "tabular"),

    # Test the effect of TF-IDF text
    ("ridge", "tabular_text"),
    ("random_forest", "tabular_text"),
    ("xgboost", "tabular_text"),

    # Test the effect of review sentiment
    ("xgboost", "tabular_sentiment"),

    # Full multimodal dataset
    ("xgboost", "all"),
]


def ensure_output_directories() -> None:
    for directory in [
        RESULTS_DIR,
        PREDICTIONS_DIR,
        MODELS_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def run_experiments() -> pd.DataFrame:
    ensure_output_directories()

    train_df, val_df, _ = load_model_ready_data()

    results: list[dict] = []

    for model_name, feature_set in EXPERIMENTS:
        print("\n" + "=" * 70)
        print(
            f"Model: {model_name} | "
            f"Features: {feature_set}"
        )
        print("=" * 70)

        X_train, y_train = create_xy(
            train_df,
            target=TARGET,
            feature_set=feature_set,
        )

        X_val, y_val = create_xy(
            val_df,
            target=TARGET,
            feature_set=feature_set,
        )

        print("X_train shape:", X_train.shape)
        print("X_val shape:", X_val.shape)

        result = train_model(
            model_name=model_name,
            feature_set=feature_set,
            target=TARGET,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            val_listing_ids=val_df["listing_id"],
            random_seed=RANDOM_SEED,
        )

        result_row = {
            "model": model_name,
            "feature_set": feature_set,
            "target": TARGET,
            "training_time_seconds":
                result.training_time_seconds,
            "best_iteration":
                result.best_iteration,
            **result.validation_metrics,
        }

        results.append(result_row)

        experiment_name = (
            f"{model_name}_{feature_set}"
        )

        predictions_path = (
            PREDICTIONS_DIR
            / f"{experiment_name}_val_predictions.csv"
        )

        result.validation_predictions.to_csv(
            predictions_path,
            index=False,
        )

        model_path = (
            MODELS_DIR
            / f"{experiment_name}.joblib"
        )

        joblib.dump(
            result.pipeline,
            model_path,
        )

        print("Metrics:")
        for metric_name, value in (
            result.validation_metrics.items()
        ):
            print(
                f"  {metric_name}: {value:.4f}"
            )

        if result.best_iteration is not None:
            print(
                "Best XGBoost iteration:",
                result.best_iteration,
            )

        print(
            f"Training time: "
            f"{result.training_time_seconds:.2f} s"
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="rmse",
        ascending=True,
    ).reset_index(drop=True)

    results_path = (
        RESULTS_DIR
        / "baseline_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("FINAL VALIDATION RESULTS")
    print("=" * 70)
    print(results_df.to_string(index=False))

    print(
        f"\nResults saved to: {results_path}"
    )

    return results_df


if __name__ == "__main__":
    run_experiments()