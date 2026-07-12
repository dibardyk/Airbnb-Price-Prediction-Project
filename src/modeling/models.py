from typing import Literal

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor


ModelName = Literal[
    "dummy",
    "ridge",
    "random_forest",
    "xgboost",
]


def get_model(
    model_name: ModelName,
    random_seed: int = 42,
):
    """
    Return an untrained regression model.
    """

    if model_name == "dummy":
        return DummyRegressor(
            strategy="median"
        )

    if model_name == "ridge":
        return Ridge(
            alpha=1.0
        )

    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=random_seed,
            n_jobs=-1,
        )

    if model_name == "xgboost":
        return XGBRegressor(
            n_estimators=3000,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            eval_metric="rmse",
            early_stopping_rounds=50,
            random_state=random_seed,
            n_jobs=-1,
        )

    raise ValueError(
        f"Unknown model name: {model_name!r}"
    )