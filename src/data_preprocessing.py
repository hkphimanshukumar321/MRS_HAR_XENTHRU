#Purpose: Processes the raw UWB radar data into a format suitable for model training. Steps include stacking multi-radar frames, max-pooling to combine data from multiple sensors or time frames, subtracting background static noise, and normalizing the signal. The output is stored in data/processed/ (for example, as NumPy arrays ready for training). What it does:
#Reads raw data files from data/kaggle/ (the exact file format depends on the dataset, e.g., CSV, NPZ, etc.).
#For each data sample (e.g., an activity instance for a particular subject), performs:
#Background Subtraction: Remove static background by subtracting an average or reference frame (helps isolate moving subjects).
#Stacking & Max-Pooling: If multiple radar sensors or antennas are used, stack their data and optionally combine using max-pooling (taking the maximum value across sensors or time frames to emphasize strong reflections).
#Normalization: Scale signals (e.g., min-max normalization or standardization) so that values range consistently (improves training stability).
#Stores processed sequences and labels into a single file (e.g., processed_data.npz) under data/processed/.


#Usage:
#Simply run:
#bash
#Copy
#python data_preprocessing.py
#This will process the entire raw dataset. You can modify the script to change normalization methods or to select a subset of data.


  #!/usr/bin/env python3
"""
data_preprocessing.py: Preprocess UWB radar data for HAR.

Processes raw data by stacking multi-radar frames (if applicable), applying max-pooling to combine sensor data, subtracting background noise, and normalizing the signals. Outputs processed sequences and labels to data/processed/.

Assumes raw data files are in data/kaggle/. Adjust file reading logic as needed for your dataset format.
"""

import os
import numpy as np

def preprocess_sequence(raw_sequence):
    """
    Apply preprocessing steps to a raw radar data sequence.
    - raw_sequence: numpy array of shape (T, H, W[, C]) representing T time frames of radar data.
      H and W are spatial dimensions (e.g., range and angle bins), and C is number of radar sensors or channels (optional).
    Returns: processed sequence as numpy array of shape (T, H, W, 1) after background subtraction and normalization.
    """
    # 1. Background subtraction: compute static background (mean over time or first frame)
    if raw_sequence.ndim == 3:
        # Shape (T, H, W) with a single radar sensor (no channel dim)
        background = np.mean(raw_sequence, axis=0)
    elif raw_sequence.ndim == 4:
        # Shape (T, H, W, C) with multiple channels/sensors
        background = np.mean(raw_sequence, axis=0)  # average background for each HxW across time
    else:
        background = 0  # If data is 2D or unexpected shape, skip background subtraction
    
    # Subtract background from each frame
    seq_bg_subtracted = raw_sequence - background
    
    # If values can go negative after subtraction, set any negative values to 0 (optional)
    seq_bg_subtracted[seq_bg_subtracted < 0] = 0.0
    
    # 2. Stacking & max-pooling for multi-sensor data:
    if seq_bg_subtracted.ndim == 4:
        # If multiple radar channels, take max across channel dimension for each frame
        # (This fuses multiple radar feeds into one, emphasizing strongest signal per location)
        seq_fused = np.max(seq_bg_subtracted, axis=-1)  # result shape: (T, H, W)
    else:
        seq_fused = seq_bg_subtracted  # single sensor case
    
    # (Optional) Temporal pooling: you could also downsample in time by taking max/mean over sliding windows if needed.
    # For example, to halve the frame rate: 
    # seq_fused = np.maximum(seq_fused[0::2], seq_fused[1::2]) 
    # (this is an example of temporal max-pooling)
    
    # 3. Normalization: scale data to 0-1 range per frame or per sequence
    # Compute global min and max for the sequence
    seq_min = seq_fused.min()
    seq_max = seq_fused.max()
    if seq_max > seq_min:
        seq_norm = (seq_fused - seq_min) / (seq_max - seq_min)
    else:
        seq_norm = seq_fused  # if constant sequence (unlikely), skip normalization
    
    # 4. Add channel dimension for model input (e.g., as a single-channel image per frame)
    processed_seq = seq_norm[..., np.newaxis]  # shape becomes (T, H, W, 1)
    return processed_seq

if __name__ == "__main__":
    raw_data_path = "data/kaggle"
    output_path = "data/processed"
    os.makedirs(output_path, exist_ok=True)
    
    X_data = []    # list to accumulate processed sequences
    y_labels = []  # list of activity labels
    subjects = []  # list of subject IDs
    
    # Example: iterate through files in data/kaggle. 
    # (Modify this part based on actual dataset structure)
    for filename in os.listdir(raw_data_path):
        filepath = os.path.join(raw_data_path, filename)
        if not os.path.isfile(filepath):
            continue
        # Assume filename format: subjectID_activity_label.npy (or .csv, etc.)
        # Example: "subject1_walking.npy" or "S1_walk.csv"
        name, ext = os.path.splitext(filename)
        parts = name.split('_')
        if len(parts) < 2:
            # Skip files that don't match expected naming
            continue
        subj_id = parts[0].replace("subject", "").replace("S", "")
        activity_label = parts[1]  # e.g., "walking"
        
        # Load raw data. Handle different file formats:
        if ext == ".npy" or ext == ".npz":
            raw = np.load(filepath)
            # If npz, assume it has an array under key 'arr_0' or similar:
            if isinstance(raw, np.lib.npyio.NpzFile):
                raw = raw[list(raw.files)[0]]
        elif ext == ".csv":
            raw = np.loadtxt(filepath, delimiter=',')
            # If CSV, raw might be 2D (time x features); reshape or adjust as needed.
        else:
            continue  # skip unsupported file types
        
        # Ensure raw data is in expected shape (T, H, W[, C]).
        # If raw data is 2D (T x features) and features correspond to spatial (H*W),
        # reshape it to (T, H, W) appropriately. This depends on sensor specifics.
        if raw.ndim == 2:
            # Example: if each row is a range profile (with W range bins) and no height dimension:
            raw = raw[:, :, np.newaxis]  # add a width dim if needed (this is dataset-specific)
        
        processed = preprocess_sequence(raw)
        X_data.append(processed)
        y_labels.append(activity_label)
        subjects.append(subj_id)
    
    # Convert lists to arrays for saving
    X_data = np.array(X_data, dtype=np.float32)
    y_labels = np.array(y_labels)
    subjects = np.array(subjects)
    
    # Save processed data and corresponding labels/subject IDs
    out_file = os.path.join(output_path, "processed_data.npz")
    np.savez(out_file, X=X_data, y=y_labels, subject=subjects)
    print(f"Processed data saved to {out_file}. Total samples: {len(X_data)}")


#Comments: Adjust the file reading section according to the actual Kaggle dataset structure.
#The code assumes each file corresponds to one activity sample for a subject, deducing subject and activity from filename. If the dataset is structured differently (e.g., one big file), you would load that and iterate through entries instead. 
#The preprocess_sequence function can be tuned: for example, use a running average for background subtraction, different pooling strategies, or per-frame normalization vs. per-sequence. After running this script, you should have a processed_data.npz containing X (the processed sequences), y (activity labels), and subject (subject identifiers). 
#This will be used for model training. Future enhancements: You might incorporate data augmentation (e.g., adding noise, random shifts) here, or apply more complex filtering (like using a high-pass filter to isolate motion). If the radar data has additional dimensions (e.g., Doppler), extend the preprocessing accordingly (perhaps creating multi-channel inputs). 
  The preprocess_sequence function can also be imported in other scripts (like infer.py or real_time_inference.py) to ensure new raw data is processed in the same way as the training data.
