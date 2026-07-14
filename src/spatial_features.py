"""
Features produced:
    dist_to_center     - distance (km) to the Berlin TV tower
    cluster            - k-means location cluster id (fit on train), string
    dist_to_cluster    - distance (m) to the nearest cluster centroid
    dist_to_station    - distance (m) to the nearest U-Bahn/S-Bahn station
    knn_price_smooth   - mean log_price of the k nearest train listings (fit on train)
"""

import sys
import time
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))
from data_loader import load_data
PROCESSED_DIR = (SCRIPT_DIR / "../data/processed").resolve()


CENTER_LAT, CENTER_LON = 52.520833, 13.409444  # Berlin TV tower
N_CLUSTERS = 10
KNN_K = 10


OVERPASS_QUERY = """
[out:json][timeout:60];
(
  node["railway"="station"]["station"~"subway|light_rail"](52.33,13.05,52.68,13.77);
);
out body;
"""
OVERPASS_MIRRORS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]


def haversine(lat, lon, lat0=CENTER_LAT, lon0=CENTER_LON):
    R = 6371
    lat, lon, lat0, lon0 = map(radians, [lat, lon, lat0, lon0])
    dlat, dlon = lat - lat0, lon - lon0
    a = sin(dlat / 2) ** 2 + cos(lat0) * cos(lat) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def to_metric_xy(df):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326").to_crs("EPSG:25833")
    return np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values]) # type: ignore


def fetch_overpass(query, mirrors=OVERPASS_MIRRORS, retries=2):
    headers = {"User-Agent": "AirbnbBerlinProject/1.0 (university course project)"}
    for url in mirrors:
        for _ in range(retries):
            resp = requests.post(url, data={"data": query}, headers=headers, timeout=90)
            if resp.status_code == 200:
                try:
                    return resp.json()
                except requests.exceptions.JSONDecodeError:
                    print(f"{url}: got 200 but not valid JSON, retrying...")
            else:
                print(f"{url}: status {resp.status_code}, retrying...")
            time.sleep(3)
    raise RuntimeError("All Overpass attempts failed - try again in a few minutes.")


def get_station_xy():
    data = fetch_overpass(OVERPASS_QUERY)
    stations = pd.DataFrame([{"lat": el["lat"], "lon": el["lon"], "name": el.get("tags", {}).get("name", "")} for el in data["elements"]])
    print(f"Fetched {len(stations)} stations")
    return to_metric_xy(stations.rename(columns={"lat": "latitude", "lon": "longitude"}))


def create_spatial_features():
    data = load_data()
    splits = {name: data[name]["listings"].copy() for name in ["train", "val", "test"]}

    # dist_to_center: fixed point, safe to compute on every split
    for df in splits.values():
        df["dist_to_center"] = df.apply(lambda r: haversine(r["latitude"], r["longitude"]), axis=1)

    # metric coordinates
    xy = {name: to_metric_xy(df) for name, df in splits.items()}

    # k-means clusters: fit on train only
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    kmeans.fit(xy["train"])
    for name, df in splits.items():
        df["cluster"] = kmeans.predict(xy[name]).astype(str)
        df["dist_to_cluster"] = kmeans.transform(xy[name]).min(axis=1)

    # dist_to_station: fixed reference points, safe on every split
    station_xy = get_station_xy()
    nn_station = NearestNeighbors(n_neighbors=1).fit(station_xy)
    for name, df in splits.items():
        df["dist_to_station"] = nn_station.kneighbors(xy[name])[0].ravel()

    # knn_price_smooth: fit on train only
    nn_price = NearestNeighbors().fit(xy["train"])
    train_log_prices = splits["train"]["log_price"].values

    train_idx = nn_price.kneighbors(xy["train"], n_neighbors=KNN_K + 1, return_distance=False)
    splits["train"]["knn_price_smooth"] = [train_log_prices[[j for j in row if j != i][:KNN_K]].mean() for i, row in enumerate(train_idx)]
    for name in ["val", "test"]:
        idx = nn_price.kneighbors(xy[name], n_neighbors=KNN_K, return_distance=False)
        splits[name]["knn_price_smooth"] = train_log_prices[idx].mean(axis=1)

    # assemble output
    feature_cols = ["listing_id", "dist_to_center", "cluster", "dist_to_cluster",
                     "dist_to_station", "knn_price_smooth"]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        out = df[feature_cols]
        out_path = PROCESSED_DIR / f"spatial_features_{name}.csv"
        out.to_csv(out_path, index=False)
        print(f"{name}: saved {out.shape} to {out_path}")


if __name__ == "__main__":
    create_spatial_features()
    print("Spatial features created.")
