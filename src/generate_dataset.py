"""
generate_dataset.py — Downloads/generates the standard insurance.csv dataset
Run once to populate data/insurance.csv cleanly.
"""
import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "insurance.csv")

URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"

def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading insurance dataset to {CSV_PATH} ...")
    urllib.request.urlretrieve(URL, CSV_PATH)
    print("Done!")

if __name__ == "__main__":
    download_dataset()
