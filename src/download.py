import gzip
import shutil
import requests
from pathlib import Path

# No calendar.csv because all the pricing data is missing, so it's basically useless
# neighbourhoods.csv is redundant because its encoded in listings.csv
URLS = [
    "https://data.insideairbnb.com/germany/be/berlin/2025-09-23/data/listings.csv.gz",
    "https://data.insideairbnb.com/germany/be/berlin/2025-09-23/data/reviews.csv.gz",
    "https://data.insideairbnb.com/germany/be/berlin/2025-09-23/visualisations/neighbourhoods.geojson"
]


def download_raw_data(url_list=URLS, output_dir="data/raw"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for url in url_list:
        filename = url.split("/")[-1]
        file_path = output_path / filename
        
        print(f"Downloading: {filename}...")
        try:
            response = requests.get(url)
            response.raise_for_status() 
            
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Download complete: {filename}")
            
            if file_path.suffix == ".gz":
                new_filename = file_path.with_suffix("") 
                
                print(f"Unpacking: {filename} -> {new_filename.name}...")
                with gzip.open(file_path, "rb") as f_in:
                    with open(new_filename, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out) # type: ignore
                
                file_path.unlink()
                print(f"Unpacked: {new_filename.name}")
                
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            
    print("\nAll files successfully downloaded.")
