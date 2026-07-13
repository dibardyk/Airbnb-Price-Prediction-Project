import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from data_loader import load_data


TEXT_COLUMNS = ["name", "description", "neighborhood_overview"]

CUSTOM_STOPWORDS = [
    "berlin", "apartment", "flat", "place", "stay", "room"
]

def combine_text_columns(df, text_columns=TEXT_COLUMNS):
    text = ""
    for col in text_columns:
        if col in df.columns:
            text = text + " " + df[col].fillna("").astype(str)
    return text.str.strip()


def create_text_features(max_features=500):
    data = load_data()

    train = data["train"]["listings"].copy()
    val = data["val"]["listings"].copy()
    test = data["test"]["listings"].copy()

    train_text = combine_text_columns(train)
    val_text = combine_text_columns(val)
    test_text = combine_text_columns(test)

    stop_words = list(ENGLISH_STOP_WORDS.union(CUSTOM_STOPWORDS))

    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words=stop_words,
        lowercase=True,
        min_df=3,
        max_df=0.7
    )

    X_train = vectorizer.fit_transform(train_text)
    X_val = vectorizer.transform(val_text)
    X_test = vectorizer.transform(test_text)

    feature_names = [
        f"tfidf_{word}" for word in vectorizer.get_feature_names_out()
    ]

    train_features = pd.DataFrame(
        X_train.toarray(),
        columns=feature_names
    )
    val_features = pd.DataFrame(
        X_val.toarray(),
        columns=feature_names
    )
    test_features = pd.DataFrame(
        X_test.toarray(),
        columns=feature_names
    )

    train_features.insert(0, "listing_id", train["listing_id"].values)
    val_features.insert(0, "listing_id", val["listing_id"].values)
    test_features.insert(0, "listing_id", test["listing_id"].values)

    output_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_features.to_csv(output_dir / "text_features_train.csv", index=False)
    val_features.to_csv(output_dir / "text_features_val.csv", index=False)
    test_features.to_csv(output_dir / "text_features_test.csv", index=False)

if __name__ == "__main__":
    create_text_features()
    print("Features are created.")