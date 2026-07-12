import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

def load_data(data_dir=None, random_seed=42, price_upper_quantile=0.99):
    """
    Train 80%, Validation 20%, Test 20%
    
    Usage for example:
    from src.data_loader import load_data
    data = load_data()
    df_train = data["train"]["listings"]
    df_val = data["val"]["listings"]
    df_test = data["test"]["listings"]

    For review data:
    text_train = data["train"]["reviews"]
    text_val = data["val"]["reviews"]
    text_test = data["test"]["reviews"]
    """
    base_path = Path(data_dir) if data_dir else (SCRIPT_DIR / "../data/clean")
    base_path = base_path.resolve()
    listings_path = base_path / "listings.csv"
    reviews_path = base_path / "reviews.csv"
    
    if not listings_path.exists() or not reviews_path.exists():
        raise FileNotFoundError(f"Could not find CSVs at {base_path}.")

    listings = pd.read_csv(listings_path, low_memory=False)
    reviews = pd.read_csv(reviews_path, low_memory=False)
    
    train_val_list, test_list = train_test_split(listings, test_size=0.20, random_state=random_seed)
    train_list, val_list = train_test_split(train_val_list, test_size=0.20, random_state=random_seed)
    
    # Outlier removal based only on training set: 99% quantile
    price_upper_bound = train_list["price"].quantile(price_upper_quantile)
    
    def filter_outliers(df):
        mask_price = df["price"] <= price_upper_bound
        return df[mask_price].copy()

    # Apply outlier filtering ONLY to the training set (can be changed, but so it resembles "real" data)
    train_list = filter_outliers(train_list)
    val_list = val_list.copy()
    test_list = test_list.copy()
    
    # For clean imputation replace the 0 (not a real rating) with nan
    def replace_zeros_with_nan(df):
        df = df.copy()
        if "rating_overall" in df.columns:
            df["rating_overall"] = df["rating_overall"].replace(0, np.nan)
        if "rating_location" in df.columns:
            df["rating_location"] = df["rating_location"].replace(0, np.nan)
        return df

    train_list = replace_zeros_with_nan(train_list)
    val_list = replace_zeros_with_nan(val_list)
    test_list = replace_zeros_with_nan(test_list)
    
    # Missing value imputation
    train_rating_mean = (
        train_list["rating_overall"].mean()
        if "rating_overall" in train_list.columns
        else np.nan
    )
    train_loc_mean = (
        train_list["rating_location"].mean()
        if "rating_location" in train_list.columns
        else np.nan
    )
    train_resp_med = (
        train_list["host_response_rate"].median()
        if "host_response_rate" in train_list.columns
        else np.nan
    )
    train_accept_med = (
        train_list["host_acceptance_rate"].median()
        if "host_acceptance_rate" in train_list.columns
        else np.nan
    )
    train_since_med = (
        train_list["host_since_days"].median()
        if "host_since_days" in train_list.columns
        else np.nan
    )

    def impute_missing_values(df):
        df = df.copy()
        if "rating_overall" in df.columns:
            df["rating_overall"] = df["rating_overall"].fillna(train_rating_mean)
            
        if "rating_location" in df.columns:
            df["rating_location"] = df["rating_location"].fillna(train_loc_mean)
            
        if "host_response_rate" in df.columns:
            df["host_response_rate"] = df["host_response_rate"].fillna(train_resp_med)
            
        if "host_acceptance_rate" in df.columns:
            df["host_acceptance_rate"] = df["host_acceptance_rate"].fillna(train_accept_med)
            
        if "host_since_days" in df.columns:
            df["host_since_days"] = df["host_since_days"].fillna(train_since_med)
        return df

    # Apply to all splits
    train_list = impute_missing_values(train_list)
    val_list = impute_missing_values(val_list)
    test_list = impute_missing_values(test_list)
    
    train_rev = reviews[reviews["listing_id"].isin(train_list["listing_id"])]
    val_rev = reviews[reviews["listing_id"].isin(val_list["listing_id"])]
    test_rev = reviews[reviews["listing_id"].isin(test_list["listing_id"])]
    
    return {
        "train": {"listings": train_list, "reviews": train_rev},
        "val": {"listings": val_list, "reviews": val_rev},
        "test": {"listings": test_list, "reviews": test_rev}
    }


if __name__ == "__main__":
    print("Testing the data loader...\n")
    try:
        data = load_data()
        
        print("Successful!\n")
        for split in ["train", "val", "test"]:
            n_listings = len(data[split]["listings"])
            n_reviews = len(data[split]["reviews"])
            
            print(f"{split} set:")
            print(f"  - Listings: {n_listings} rows")
            print(f"  - Reviews:  {n_reviews} rows\n")
        
    except Exception as e:
        print(f"Something went wrong:\n{e}")
