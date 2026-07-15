from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from time import perf_counter
import random
import sys

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.evaluation import calculate_regression_metrics
from src.modeling.model_data import create_xy, load_model_ready_data
from src.modeling.preprocessing import build_preprocessor
from xgboost import XGBRegressor


TARGET = "log_price"
FEATURE_SET = "tabular_spatial"
RANDOM_SEED = 42

OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_DIR = OUTPUT_DIR / "results"
MODELS_DIR = OUTPUT_DIR / "models"
TUNING_DIR = OUTPUT_DIR / "tuning"


@dataclass(frozen=True)
class TuneConfig:
    learning_rate: float
    max_depth: int
    min_child_weight: int
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    n_estimators: int


SEARCH_SPACE = {
    "learning_rate": [0.01, 0.03, 0.05],
    "max_depth": [4, 5, 6, 8],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "reg_alpha": [0.0, 0.1, 0.5],
    "reg_lambda": [0.8, 1.0, 1.5],
    "n_estimators": [1500, 2500, 3500],
}


def ensure_output_directories() -> None:
    for directory in [RESULTS_DIR, MODELS_DIR, TUNING_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def iter_configs() -> list[TuneConfig]:
    keys = list(SEARCH_SPACE.keys())
    values = [SEARCH_SPACE[key] for key in keys]
    all_configs = [
        TuneConfig(**dict(zip(keys, combo, strict=True)))
        for combo in product(*values)
    ]

    sample_size = min(36, len(all_configs))
    rng = random.Random(RANDOM_SEED)
    return rng.sample(all_configs, k=sample_size)


def fit_and_score(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: TuneConfig,
) -> tuple[dict[str, float], int | None, float, object, XGBRegressor]:
    preprocessor = build_preprocessor(
        X_train,
        scale_numeric=False,
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    model = XGBRegressor(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=50,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    start_time = perf_counter()
    model.fit(
        X_train_processed,
        y_train,
        eval_set=[(X_val_processed, y_val)],
        verbose=False,
    )
    training_time_seconds = perf_counter() - start_time

    predictions = model.predict(X_val_processed)
    metrics = calculate_regression_metrics(
        y_val,
        predictions,
        target=TARGET,
    )
    best_iteration = getattr(model, "best_iteration", None)

    return metrics, best_iteration, training_time_seconds, preprocessor, model


def tune_tabular_spatial_xgboost() -> pd.DataFrame:
    ensure_output_directories()

    train_df, val_df, _ = load_model_ready_data()
    X_train, y_train = create_xy(
        train_df,
        target=TARGET,
        feature_set=FEATURE_SET,
    )
    X_val, y_val = create_xy(
        val_df,
        target=TARGET,
        feature_set=FEATURE_SET,
    )

    print(f"Training shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")

    rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None
    best_model: dict[str, object] | None = None

    for index, config in enumerate(iter_configs(), start=1):
        print("\n" + "=" * 72)
        print(f"Trial {index}: {config}")
        print("=" * 72)

        metrics, best_iteration, training_time_seconds, preprocessor, model = fit_and_score(
            X_train,
            y_train,
            X_val,
            y_val,
            config,
        )

        row = {
            **asdict(config),
            "feature_set": FEATURE_SET,
            "target": TARGET,
            "training_time_seconds": training_time_seconds,
            "best_iteration": best_iteration,
            **metrics,
        }
        rows.append(row)

        print(
            "rmse="
            f"{metrics['rmse']:.6f}, "
            f"mae={metrics['mae']:.6f}, "
            f"r2={metrics['r2']:.6f}, "
            f"best_iteration={best_iteration}, "
            f"time={training_time_seconds:.2f}s"
        )

        if best_row is None or metrics["rmse"] < best_row["rmse"]:
            best_row = row
            best_model = {
                "preprocessor": preprocessor,
                "model": model,
            }

    results_df = pd.DataFrame(rows).sort_values(by="rmse", ascending=True).reset_index(drop=True)

    results_path = TUNING_DIR / "tabular_spatial_xgboost_tuning_results.csv"
    results_df.to_csv(results_path, index=False)

    if best_row is not None and best_model is not None:
        best_model_path = MODELS_DIR / "xgboost_tabular_spatial_tuned.joblib"
        joblib.dump(best_model, best_model_path)

        print("\nBest configuration:")
        print(pd.Series(best_row).to_string())
        print(f"\nSaved best model to: {best_model_path}")

    print(f"\nSaved tuning table to: {results_path}")
    print("\nTop 10 trials:")
    print(results_df.head(10).to_string(index=False))

    return results_df


if __name__ == "__main__":
    tune_tabular_spatial_xgboost()