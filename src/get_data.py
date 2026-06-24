from download import download_raw_data
from cleaning import clean_data

def main():
    print("--- Starting Data Pipeline ---")
    
    print("\n[1/2] Downloading data...")
    download_raw_data()

    print("\n[2/2] Cleaning data...")
    clean_data()

    print("\n--- Pipeline Finished Successfully! ---")

if __name__ == "__main__":
    main()
