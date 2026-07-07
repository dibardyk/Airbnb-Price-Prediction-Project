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
SCRIPT_DIR = Path(__file__).parent


def download_raw_data(url_list=URLS, output_dir=None):
    output_path = Path(output_dir) if output_dir else SCRIPT_DIR / "../data/raw"
    output_path = output_path.resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_success = True
    
    for url in url_list:
        filename = Path(url).name
        is_gz = filename.endswith(".gz")
        
        filename = filename[:-3] if is_gz else filename
        filepath = output_path / filename
        
        if filepath.exists():
            print(f"Skipping {filename}, already downloaded.")
            continue
        
        print(f"Processing: {filename}...")
        
        try:
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                
                if is_gz:
                    with gzip.GzipFile(fileobj=response.raw, mode="rb") as f_in:
                        with open(filepath, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                else:
                    with open(filepath, "wb") as f_out:
                        for chunk in response.iter_content(chunk_size=8192):
                            f_out.write(chunk)
                            
            print(f"Successfully processed: {filename}")
                
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            all_success = False
    
    if all_success:
        print("\nAll files successfully downloaded.")
    else:
        print("\nFinished, but some downloads failed.")
