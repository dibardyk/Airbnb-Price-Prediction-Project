from pathlib import Path

import pandas as pd

from data_loader import load_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_READY_DIR = PROJECT_ROOT / "data" / "model_ready"


SENTIMENT_COLUMNS = [
    "review_sentiment_mean",
    "review_sentiment_median",
    "review_sentiment_std",
    "review_count",
    "negative_review_share",
    "positive_review_share",
]


def load_feature_file(filename):
    path = PROCESSED_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}"
        )

    return pd.read_csv(path)


def merge_split_features(listings, split_name):
    """
    Merge tabular listings, TF-IDF features,
    and review sentiment features for one split.
    """

    tfidf = load_feature_file(
        f"text_features_{split_name}.csv"
    )

    sentiment = load_feature_file(
        f"review_sentiment_features_{split_name}.csv"
    )

    spatial = load_feature_file(
        f"spatial_features_{split_name}.csv"
    )

    # One listing should appear only once in each feature table
    if listings["listing_id"].duplicated().any():
        raise ValueError(
            f"Duplicate listing_id values in {split_name} listings."
        )

    if tfidf["listing_id"].duplicated().any():
        raise ValueError(
            f"Duplicate listing_id values in {split_name} TF-IDF features."
        )

    if sentiment["listing_id"].duplicated().any():
        raise ValueError(
            f"Duplicate listing_id values in {split_name} sentiment features."
        )

    if spatial["listing_id"].duplicated().any():
        raise ValueError(
            f"Duplicate listing_id values in {split_name} spatial features."
        )

    original_rows = len(listings)

    # Every listing should have TF-IDF features
    merged = listings.merge(
        tfidf,
        on="listing_id",
        how="left",
        validate="one_to_one"
    )

    # Some listings may have no reviews, so left join is required
    merged = merged.merge(
        sentiment,
        on="listing_id",
        how="left",
        validate="one_to_one"
    )

    # Every listing has coordinates, so every listing should have spatial features 
    merged = merged.merge(
        spatial,
        on="listing_id",
        how="left",
        validate="one_to_one"
    )

    if len(merged) != original_rows:
        raise ValueError(
            f"Row count changed after merging {split_name}: "
            f"{original_rows} -> {len(merged)}"
        )

    # Listings without reviews receive neutral/default values
    existing_sentiment_columns = [
        col for col in SENTIMENT_COLUMNS
        if col in merged.columns
    ]

    merged[existing_sentiment_columns] = (
        merged[existing_sentiment_columns].fillna(0)
    )

    merged = merged.copy()

    if "review_count" in merged.columns:
        merged["has_reviews"] = (
            merged["review_count"] > 0
        ).astype(int)

    return merged


def create_model_ready_data():
    """
    Create merged train, validation, and test datasets.
    """

    data = load_data()

    MODEL_READY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {}

    for split_name in ["train", "val", "test"]:
        listings = (
            data[split_name]["listings"]
            .copy()
        )

        merged = merge_split_features(
            listings,
            split_name
        )

        output_path = (
            MODEL_READY_DIR
            / f"{split_name}_full.csv"
        )

        merged.to_csv(
            output_path,
            index=False
        )

        output[split_name] = merged

        print(
            f"{split_name}: "
            f"{listings.shape} -> {merged.shape}"
        )

        print(
            f"Saved to: {output_path}"
        )

    print("\nModel-ready datasets created successfully.")

    return output


if __name__ == "__main__":
    create_model_ready_data()