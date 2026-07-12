from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


@dataclass(frozen=True)
class FeatureGroups:
    numeric: list[str]
    categorical: list[str]


def identify_feature_groups(
    X: pd.DataFrame,
) -> FeatureGroups:
    """
    Identify numeric and categorical columns.
    """

    categorical_columns = (
        X.select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )

    numeric_columns = (
        X.select_dtypes(
            include=["number", "bool"]
        )
        .columns
        .tolist()
    )

    recognised_columns = set(
        numeric_columns + categorical_columns
    )

    unrecognised_columns = [
        column
        for column in X.columns
        if column not in recognised_columns
    ]

    if unrecognised_columns:
        raise TypeError(
            "Unsupported feature dtypes for columns: "
            f"{unrecognised_columns}"
        )

    return FeatureGroups(
        numeric=numeric_columns,
        categorical=categorical_columns,
    )


def build_preprocessor(
    X: pd.DataFrame,
    scale_numeric: bool,
) -> ColumnTransformer:
    """
    Build a reusable preprocessing transformer.

    Numeric:
        median imputation
        optional standardisation

    Categorical:
        most-frequent imputation
        one-hot encoding
    """

    feature_groups = identify_feature_groups(X)

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(strategy="median"),
        )
    ]

    if scale_numeric:
        numeric_steps.append(
            (
                "scaler",
                StandardScaler(with_mean=False),
            )
        )

    numeric_pipeline = Pipeline(
        steps=numeric_steps
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                feature_groups.numeric,
            ),
            (
                "categorical",
                categorical_pipeline,
                feature_groups.categorical,
            ),
        ],
        remainder="drop",
    )

    return preprocessor