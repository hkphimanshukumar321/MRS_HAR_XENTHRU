#!/usr/bin/env python3
"""
fetch_data.py: Download HAR dataset from Kaggle and store in data/kaggle/.

Usage:
    python fetch_data.py --dataset <Kaggle-dataset-identifier>
Example:
    python fetch_data.py --dataset user123/uwb-har-dataset

This script requires Kaggle API credentials. Ensure that you have placed your
Kaggle API token (kaggle.json) in ~/.kaggle/ or set environment variables 
KAGGLE_USERNAME and KAGGLE_KEY. Modify the `dataset` argument as needed for different datasets.
"""

import os
import argparse

def download_kaggle_dataset(dataset: str, download_path: str = "data/kaggle"):
    """Download and extract a Kaggle dataset to the specified path."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError("Kaggle API not found. Please install it with `pip install kaggle`.")
    api = KaggleApi()
    api.authenticate()
    os.makedirs(download_path, exist_ok=True)
    print(f"Downloading Kaggle dataset {dataset} to {download_path}...")
    api.dataset_download_files(dataset, path=download_path, unzip=True)
    print("Download complete. Files are saved to:", download_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset from Kaggle into data/kaggle.")
    parser.add_argument("--dataset", "-d", required=True,
                        help="Kaggle dataset identifier in the form <owner>/<dataset-name>")
    args = parser.parse_args()
    download_kaggle_dataset(args.dataset)


#use bash file to run the above code 
#Purpose: Downloads the HAR dataset from Kaggle and stores it under data/kaggle. This script uses the Kaggle API to fetch the dataset. Ensure you have a Kaggle API token set up (placed in ~/.kaggle/kaggle.json or environment variables KAGGLE_USERNAME and KAGGLE_KEY). Usage:
#bash
#Copy
#python fetch_data.py --dataset <username/dataset-name>
#Replace <username/dataset-name> with the Kaggle dataset identifier (for example, user123/uwb-har-dataset). The dataset files will be downloaded and unpacked into data/kaggle/. You can modify the script to download competitions or specific files as needed.
#Comments: This script uses the Kaggle API to download and unzip the dataset. You can modify download_kaggle_dataset to download specific files or use api.competition_download_files for competition data. After running this, check data/kaggle/ for the new dataset files (e.g., CSVs, NPZs, etc.). In future, if the dataset updates or moves, update the dataset argument accordingly.
