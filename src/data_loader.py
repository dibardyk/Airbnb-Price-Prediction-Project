import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_data(data_dir="../data/clean", random_seed=42):
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
    base_path = Path(__file__).parent / data_dir
    listings_path = base_path / "listings.csv"
    reviews_path = base_path / "reviews.csv"
    
    if not listings_path.exists() or not reviews_path.exists():
        raise FileNotFoundError(f"Could not find CSVs at {base_path}.")

    listings = pd.read_csv(listings_path, low_memory=False)
    reviews = pd.read_csv(reviews_path, low_memory=False)
    
    train_val_list, test_list = train_test_split(listings, test_size=0.20, random_state=random_seed)
    train_list, val_list = train_test_split(train_val_list, test_size=0.20, random_state=random_seed)
    
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
