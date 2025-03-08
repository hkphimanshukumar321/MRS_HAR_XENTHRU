"""  The preprocessing script loads the raw UWB radar data (e.g. CSV files) from the data/kaggle/ directory and applies several transformations:
Loading Data – It scans the input folder for raw CSV files (as produced by the fetch script) and reads them into memory. If multiple CSV files are present (e.g., separate recordings or sensor streams), they will all be processed and combined.
Background Subtraction – For each dataset file, it removes the static background signal. This is done by computing a baseline (e.g., the mean of each sensor feature over time) and subtracting it from the data, effectively removing static clutter from the UWB radar signals​
RESEARCHGATE.NET
. This step helps isolate the dynamic components (human motion) from stationary reflections.
Noise Filtering (Feature Extraction) – After background removal, a simple smoothing filter is applied to the data (using a rolling mean window) to reduce high-frequency noise. Smoothing the signal after background removal is a common technique to clean UWB radar data​
RESEARCHGATE.NET
. Additional feature extraction or reshaping can be incorporated here (e.g., computing frequency-domain features, performing dimensionality reduction, etc.), but the script currently demonstrates a basic noise filter as a placeholder.
Normalization – The features are scaled to a defined range (0 to 1) for consistency​
STATS.STACKEXCHANGE.COM
. This global min-max normalization ensures that all features contribute proportionately and no single sensor dominates due to scale differences.
Saving in Multiple Formats – The processed dataset is saved to data/processed/ in multiple formats:
CSV – A human-readable CSV file (processed_data.csv) containing the processed features (and labels, if present).
NumPy .npz – A compressed NumPy file (processed_data.npz) for fast loading in Python. This file stores the feature array (and labels array separately, if applicable) for quick retrieval.
TensorFlow .tfrecord – A TFRecord file (processed_data.tfrecord) suitable for ingestion by TensorFlow pipelines. Each record contains the features (as a float list) and the label (as an int or byte) for an example, making it easy to stream into a deep learning model.
Modularity and Extensibility – The code is organized into functions (load_data, normalize_data, and saving functions), making it easy to extend. For example, one could add a function for more complex feature extraction or data augmentation and plug it into the workflow.
Usage: Run python data_preprocessing.py --input-dir <raw_data_path> --output-dir <processed_data_path>. By default, it expects raw data in data/kaggle (as created by the fetch script) and will output to data/processed. Ensure required libraries (pandas, numpy, and tensorflow if TFRecord output is needed) are installed. The script will skip TFRecord generation if TensorFlow is not available, instead of throwing an error.
python
Copy
 """ 

# code starts from here 

#!/usr/bin/env python3
"""
data_preprocessing.py - Preprocess UWB HAR data (background subtraction, normalization, feature extraction) and save in multiple formats.

Usage:
    python data_preprocessing.py --input-dir <path_to_raw_data> --output-dir <path_for_processed_output>

Example:
    python data_preprocessing.py --input-dir data/kaggle/uwb-har-dataset --output-dir data/processed

This script loads raw CSV files from the input directory, applies background subtraction to remove static clutter,
normalizes the features to [0,1], performs simple noise filtering (rolling mean) as a feature extraction step,
and then saves the processed dataset to CSV, NPZ, and TFRecord files in the output directory.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# We will import tensorflow if available for TFRecord writing (handled in the code below)
tf = None  # will hold the TensorFlow module if import succeeds

def load_data(input_dir):
    """Load all CSV files from the input directory (recursively) into a combined DataFrame, after per-file preprocessing."""
    csv_files = []
    # Find all CSV files under the input directory
    for root, dirs, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, fname))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the input directory.")
    dataframes = []
    label_col_detected = None
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Warning: Could not read {file_path} ({e}). Skipping this file.")
            continue
        # If a label column (activity class) is present, detect it (common names: 'label', 'activity', 'class')
        if label_col_detected is None:
            for col in df.columns:
                if col.lower() in ("label", "activity", "class", "target"):
                    label_col_detected = col
                    break
        # Separate features and labels (if label column exists)
        if label_col_detected and label_col_detected in df.columns:
            labels = df[label_col_detected].copy()
            features = df.drop(columns=[label_col_detected])
        else:
            labels = None
            features = df
        # **Background Subtraction**: remove static background by subtracting the mean of each feature&#8203;:contentReference[oaicite:7]{index=7}
        # This assumes each column in 'features' is a sensor or measurement where static offset can be removed.
        baseline = features.mean()
        features_bs = features - baseline  # subtract mean to remove static component (clutter)
        # **Noise Filtering**: apply a rolling mean (window=3) to smooth out short-term noise&#8203;:contentReference[oaicite:8]{index=8}
        features_smoothed = features_bs.rolling(window=3, min_periods=1).mean()
        # (The rolling mean is a simple example of a noise filter. More complex filters or feature transforms can be applied here.)
        processed_df = features_smoothed
        # Reattach label column (if it was separated) without modification
        if labels is not None:
            processed_df[label_col_detected] = labels.values
        dataframes.append(processed_df)
    # Combine all processed data into one DataFrame
    if not dataframes:
        raise RuntimeError("No dataframes to combine after processing. (All files may have failed to read.)")
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df, label_col_detected

def normalize_data(df, label_col=None):
    """Normalize feature columns to [0,1] range using min-max scaling&#8203;:contentReference[oaicite:9]{index=9}. Leaves label column (if any) unchanged."""
    if label_col and label_col in df.columns:
        features = df.drop(columns=[label_col])
    else:
        features = df
    # Compute min and max for each feature column
    min_vals = features.min()
    max_vals = features.max()
    # Perform min-max normalization: (value - min) / (max - min)&#8203;:contentReference[oaicite:10]{index=10}
    features_norm = (features - min_vals) / (max_vals - min_vals)
    # Replace any NaN (if max == min for a feature) with 0.0
    features_norm = features_norm.fillna(0.0)
    # If label column exists, add it back to the DataFrame
    if label_col and label_col in df.columns:
        features_norm[label_col] = df[label_col].values
    return features_norm

def save_to_csv(df, output_dir):
    """Save the DataFrame to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "processed_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved processed data to CSV: {csv_path}")

def save_to_npz(df, label_col=None, output_dir="data/processed"):
    """Save the processed data to a NumPy .npz file (features and labels)."""
    npz_path = os.path.join(output_dir, "processed_data.npz")
    # Separate features and labels for saving
    if label_col and label_col in df.columns:
        X = df.drop(columns=[label_col]).to_numpy()
        y = df[label_col].to_numpy()
        # Save features array X and labels array y
        np.savez_compressed(npz_path, X=X, y=y)
    else:
        data_array = df.to_numpy()
        np.savez_compressed(npz_path, data=data_array)
    print(f"Saved processed data to NumPy NPZ: {npz_path}")

def save_to_tfrecord(df, label_col=None, output_dir="data/processed"):
    """Save the processed data to a TensorFlow TFRecord file (if TensorFlow is available)."""
    tfrecord_path = os.path.join(output_dir, "processed_data.tfrecord")
    if tf is None:
        # TensorFlow not available, skip TFRecord
        print("TensorFlow not installed, skipping TFRecord generation.")
        return
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for _, row in df.iterrows():
            # Prepare feature values (as float list) and label (if present) for this row
            if label_col and label_col in df.columns:
                label_value = row[label_col]
                # Create a feature for label: use int64 if numeric, or bytes if string
                if isinstance(label_value, str):
                    label_feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_value.encode()]))
                else:
                    label_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label_value)]))
            # Create feature list from row values (exclude label if present)
            feature_values = row.drop(label_col).values if label_col and label_col in df.columns else row.values
            feature_values = [float(x) for x in feature_values]  # ensure all features are float type
            features_feat = tf.train.Feature(float_list=tf.train.FloatList(value=feature_values))
            # Build the Example protobuf
            feature_dict = {'features': features_feat}
            if label_col and label_col in df.columns:
                feature_dict['label'] = label_feat
            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            writer.write(example.SerializeToString())
    print(f"Saved processed data to TFRecord: {tfrecord_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess raw HAR data and save it in CSV, NPZ, and TFRecord formats.")
    parser.add_argument("--input-dir", "-i", default="data/kaggle", help="Path to the raw data directory (default: data/kaggle)")
    parser.add_argument("--output-dir", "-o", default="data/processed", help="Path to save processed data (default: data/processed)")
    args = parser.parse_args()
    # Load and preprocess the raw data
    try:
        combined_df, label_col = load_data(args.input_dir)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    # Normalize the combined data (scales features to 0-1 range)
    processed_df = normalize_data(combined_df, label_col)
    # Attempt to import TensorFlow for TFRecord writing
    try:
        import tensorflow as _tf
        global tf
        tf = _tf
    except ImportError:
        tf = None
    # Save processed data in requested formats
    save_to_csv(processed_df, args.output_dir)
    save_to_npz(processed_df, label_col, args.output_dir)
    save_to_tfrecord(processed_df, label_col, args.output_dir)
    print("Preprocessing complete. Processed data is available in the output directory.")

if __name__ == "__main__":
    main()




""" otes: These scripts are modular for easy extension. You can add more complex preprocessing steps (e.g., filtering outliers, transforming data to frequency domain, or performing dimensionality reduction) by inserting them into the workflow (for instance, in the load_data loop or as additional functions). The background subtraction and smoothing implemented here illustrate common signal processing techniques for UWB radar HAR data (removing static reflections and reducing noise)​
RESEARCHGATE.NET
​
RESEARCHGATE.NET
, and the normalization step follows the standard min-max formula​
STATS.STACKEXCHANGE.COM
. Always verify the preprocessing results (using the CSV output for a quick inspection) to ensure they make sense for your specific dataset and adjust the pipeline as needed. With the data prepared in CSV, NPZ, and TFRecord formats, you can easily move on to model training or analysis using your preferred tools.




"""
