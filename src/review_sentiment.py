import pandas as pd
import numpy as np
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from data_loader import load_data

def compute_sentiment_features():
    data = load_data()

    analyzer = SentimentIntensityAnalyzer()

    output_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        reviews = data[split]["reviews"].copy()

        reviews["comments"] = reviews["comments"].fillna("").astype(str)

        reviews["sentiment_compound"] = reviews["comments"].apply(
            lambda text: analyzer.polarity_scores(text)["compound"]
        )

        review_features = reviews.groupby("listing_id").agg(
            review_sentiment_mean=("sentiment_compound", "mean"),
            review_sentiment_median=("sentiment_compound", "median"),
            review_sentiment_std=("sentiment_compound", "std"),
            review_count=("sentiment_compound", "count"),
            negative_review_share=("sentiment_compound", lambda x: (x < -0.05).mean()),
            positive_review_share=("sentiment_compound", lambda x: (x > 0.05).mean()),
        ).reset_index()

        review_features["review_sentiment_std"] = review_features["review_sentiment_std"].fillna(0)

        output_path = output_dir / f"review_sentiment_features_{split}.csv"
        review_features.to_csv(output_path, index=False)

        print(f"{split}: saved {review_features.shape} to {output_path}")


if __name__ == "__main__":
    compute_sentiment_features()