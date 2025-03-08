""" HAR Data Fetching and Preprocessing Scripts
Below are two Python scripts that help manage a UWB radar Human Activity Recognition (HAR) dataset. The first script fetches data from Kaggle, and the second preprocesses it (background subtraction, normalization, feature extraction) and saves it in multiple formats. Both scripts include detailed comments and usage instructions.
Fetching Data from Kaggle (fetch_data.py)
This script uses the Kaggle API to download a specified dataset and organize the files. Key steps include:
Authentication – Ensure your Kaggle API credentials are set (the kaggle.json token file should be placed in ~/.kaggle/ or environment variables configured)​
STACKOVERFLOW.COM
. The script uses Kaggle's Python API to authenticate with these credentials.
Downloading the Dataset – It calls the Kaggle API to download all files for a given dataset​
STACKOVERFLOW.COM
. You can specify the dataset by its <owner>/<dataset-name> slug.
Extraction – After download, the script extracts the dataset archive.
Organization – Extracted files are moved into subdirectories by type: CSV files to a raw_csvs/ folder, image files to an images/ folder, and other files (e.g., metadata like JSON, TXT) to a metadata/ folder.
Error Handling – The script gracefully handles common errors (missing Kaggle API key, invalid dataset name, network issues) by printing informative messages instead of crashing.
Usage: Run python fetch_data.py -d <owner/dataset-name> -o <output_path> (output path is optional; default is data/kaggle/<dataset-name>). Make sure you have the Kaggle API package installed (pip install kaggle) and your API token configured before running.
"""

#code starts from here 

#!/usr/bin/env python3
"""
fetch_data.py - Download a Kaggle dataset and organize it into subdirectories.

Usage:
    python fetch_data.py --dataset <owner/dataset-name> [--outdir <output_path>]

Example:
    python fetch_data.py --dataset username/uwb-har-dataset --outdir data/kaggle

This script uses Kaggle's API to download the specified dataset (requires a Kaggle API token).
It then extracts the downloaded archive and sorts files into subfolders (raw_csvs/, images/, metadata/).
Errors (e.g., missing credentials or dataset not found) are handled gracefully with clear messages.
"""
import os
import sys
import argparse
import zipfile
import shutil
import uuid

# Attempt to import Kaggle API client
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("ERROR: Kaggle API library not found. Please install it with 'pip install kaggle' and ensure kaggle.json is set up.")
    sys.exit(1)

def download_dataset(dataset, out_dir):
    """Authenticate and download the Kaggle dataset archive to the specified directory."""
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    # Authenticate with Kaggle API using kaggle.json (must be in ~/.kaggle or environment)&#8203;:contentReference[oaicite:2]{index=2}
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print("ERROR: Kaggle authentication failed. Check that your API token is properly placed and configured. Details:", e)
        sys.exit(1)
    # Download the dataset files (will download as a zip file)&#8203;:contentReference[oaicite:3]{index=3}
    try:
        print(f"Downloading Kaggle dataset '{dataset}'...")
        api.dataset_download_files(dataset, path=out_dir, quiet=False, unzip=False)
    except Exception as e:
        print(f"ERROR: Failed to download dataset '{dataset}'. Please check the name and your permissions. Details:", e)
        sys.exit(1)
    # Identify any downloaded zip files and extract them
    for filename in os.listdir(out_dir):
        if filename.endswith(".zip"):
            zip_path = os.path.join(out_dir, filename)
            print(f"Extracting {filename}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(out_dir)
            except Exception as e:
                print(f"ERROR: Failed to extract '{filename}'. You may need to unzip it manually. Details:", e)
            else:
                os.remove(zip_path)  # remove the zip after successful extraction

def organize_files(out_dir):
    """Organize files in out_dir into subdirectories: raw_csvs/, images/, metadata/."""
    # Create subdirectories for organization
    raw_csv_dir = os.path.join(out_dir, "raw_csvs")
    images_dir = os.path.join(out_dir, "images")
    meta_dir = os.path.join(out_dir, "metadata")
    os.makedirs(raw_csv_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    # Walk through all files in out_dir (including any subfolders from extraction)
    for root, dirs, files in os.walk(out_dir):
        # Skip the organization directories themselves to avoid infinite loop
        if root.endswith("raw_csvs") or root.endswith("images") or root.endswith("metadata"):
            continue
        for fname in files:
            file_path = os.path.join(root, fname)
            # Determine file type by extension
            ext = os.path.splitext(fname)[1].lower()
            if ext in [".csv", ".xls", ".xlsx"]:             # raw data tables
                dest_dir = raw_csv_dir
            elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:  # image files
                dest_dir = images_dir
            elif ext in [".json", ".txt", ".pdf", ".md", ".yaml", ".xml"]:  # metadata/docs
                dest_dir = meta_dir
            else:
                # Other file types: treat as metadata by default
                dest_dir = meta_dir
            # Move file to the appropriate subdirectory
            dest_path = os.path.join(dest_dir, fname)
            try:
                shutil.move(file_path, dest_path)
            except Exception as e:
                # Handle filename conflicts by renaming
                new_name = f"{uuid.uuid4().hex}_{fname}"
                new_path = os.path.join(dest_dir, new_name)
                shutil.move(file_path, new_path)
                print(f"Warning: Renamed file {fname} to {new_name} to avoid conflict.")
    # Optionally, clean up any now-empty directories left from extraction
    for root, dirs, files in os.walk(out_dir, topdown=False):
        if root in [raw_csv_dir, images_dir, meta_dir]:
            continue
        if not dirs and not files:
            os.rmdir(root)

def main():
    parser = argparse.ArgumentParser(description="Download a Kaggle dataset and organize it into subfolders.")
    parser.add_argument("--dataset", "-d", required=True, help="Kaggle dataset identifier in the form <owner>/<dataset-name>")
    parser.add_argument("--outdir", "-o", default=None, help="Output directory (default: data/kaggle/<dataset-name>)")
    args = parser.parse_args()
    # Determine output directory (defaults to data/kaggle/<dataset>)
    out_dir = args.outdir if args.outdir else os.path.join("data", "kaggle", args.dataset.split('/')[-1])
    # Download and organize dataset
    download_dataset(args.dataset, out_dir)
    organize_files(out_dir)
    print(f"Dataset downloaded and organized in '{out_dir}'.")
    print(f" - CSV files are in {os.path.join(out_dir, 'raw_csvs')}")
    print(f" - Image files are in {os.path.join(out_dir, 'images')}")
    print(f" - Metadata and other files are in {os.path.join(out_dir, 'metadata')}")

if __name__ == "__main__":
    main()


