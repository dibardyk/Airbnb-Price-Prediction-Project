from pathlib import Path
from typing import Literal

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_READY_DIR = PROJECT_ROOT / "data" / "model_ready"


FeatureSet = Literal[
    "tabular",
    "tabular_text",
    "tabular_sentiment",
    "all",
]


RAW_TEXT_COLUMNS = [
    "name",
    "description",
    "neighborhood_overview",
    "host_about",
]

TARGET_COLUMNS = [
    "price",
    "log_price",
]

SENTIMENT_COLUMNS = [
    "review_sentiment_mean",
    "review_sentiment_median",
    "review_sentiment_std",
    "review_count",
    "negative_review_share",
    "positive_review_share",
    "has_reviews",
]


def load_model_ready_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load the merged train, validation and test datasets.
    """

    paths = {
        "train": MODEL_READY_DIR / "train_full.csv",
        "val": MODEL_READY_DIR / "val_full.csv",
        "test": MODEL_READY_DIR / "test_full.csv",
    }

    missing_files = [
        str(path)
        for path in paths.values()
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "The following model-ready files are missing:\n"
            + "\n".join(missing_files)
            + "\nRun merge_features.py first."
        )

    train = pd.read_csv(paths["train"], low_memory=False)
    val = pd.read_csv(paths["val"], low_memory=False)
    test = pd.read_csv(paths["test"], low_memory=False)

    if train.columns.tolist() != val.columns.tolist():
        raise ValueError(
            "Train and validation columns do not match."
        )

    if train.columns.tolist() != test.columns.tolist():
        raise ValueError(
            "Train and test columns do not match."
        )

    return train, val, test


def create_xy(
    df: pd.DataFrame,
    target: str = "log_price",
    feature_set: FeatureSet = "all",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create feature matrix X and target vector y.

    feature_set:
        tabular
            Original structured listing features only.

        tabular_text
            Tabular features plus TF-IDF features.

        tabular_sentiment
            Tabular features plus review sentiment features.

        all
            Tabular, TF-IDF and review sentiment features.
    """

    if target not in TARGET_COLUMNS:
        raise ValueError(
            f"target must be one of {TARGET_COLUMNS}, "
            f"but received {target!r}."
        )

    if target not in df.columns:
        raise KeyError(
            f"Target column {target!r} is missing."
        )

    columns_to_exclude = [
        "listing_id",
        *TARGET_COLUMNS,
        *RAW_TEXT_COLUMNS,
    ]

    existing_exclude_columns = [
        column
        for column in columns_to_exclude
        if column in df.columns
    ]

    X = df.drop(columns=existing_exclude_columns).copy()
    y = df[target].copy()

    tfidf_columns = [
        column
        for column in X.columns
        if column.startswith("tfidf_")
    ]

    sentiment_columns = [
        column
        for column in SENTIMENT_COLUMNS
        if column in X.columns
    ]

    if feature_set == "tabular":
        X = X.drop(
            columns=tfidf_columns + sentiment_columns
        )

    elif feature_set == "tabular_text":
        X = X.drop(columns=sentiment_columns)

    elif feature_set == "tabular_sentiment":
        X = X.drop(columns=tfidf_columns)

    elif feature_set == "all":
        pass

    else:
        raise ValueError(
            "feature_set must be one of: "
            "'tabular', 'tabular_text', "
            "'tabular_sentiment', 'all'."
        )

    if len(X) != len(y):
        raise ValueError(
            "X and y have different numbers of rows."
        )

    if y.isna().any():
        raise ValueError(
            f"Target {target!r} contains missing values."
        )

    return X, y