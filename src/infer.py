"""Purpose: Loads a trained HAR model and performs inference on new data. This can be used to test the model on specific examples or small datasets, and it serves as a template for integrating the model into applications. The script supports both frameworks (TensorFlow and PyTorch) for inference. What it does:
Parses arguments for which framework to use, the model file path, and an input data file (or parameters for generating/obtaining input).
Loads the model (for Keras: tf.keras.models.load_model; for PyTorch: initialize HARModelTorch and load state_dict).
Loads or receives input data. This can be:
A processed data sample (same format as training data, e.g., a sequence of frames).
Potentially raw data that needs preprocessing (the script includes a comment on how to handle raw input by using preprocess_sequence from data_preprocessing.py).
Runs the model on the input to get a prediction of the activity.
Outputs the predicted activity label (or class index).
Usage:
bash
Copy
python infer.py --framework tf --model_path experiments/results/model_subject_1.h5 --input_file my_sample.npz
Replace --framework with torch and adjust --model_path if using the PyTorch model (e.g., model_subject_1.pth). The --input_file should point to a file containing the data to infer on (it could be one sample or multiple). For example, you might take a snippet of the processed dataset or a new recording. Ensure the input dimensions match what the model expects (same sequence length, image size, etc., as during training). """

# code starts from here

#!/usr/bin/env python3
"""
infer.py: Load a trained HAR model and perform inference on new data.

Supports both TensorFlow (Keras) and PyTorch models.
Provide the model file path and an input data file (processed or raw).

Usage:
    python infer.py --framework [tf|torch] --model_path <model_file> --input_file <data_file>
Example:
    python infer.py --framework tf --model_path experiments/results/model_subject_1.h5 --input_file example.npz
"""
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Run inference using a trained HAR model.")
parser.add_argument("--framework", "-f", choices=["tf", "torch"], required=True, help="Framework of the model (tf or torch).")
parser.add_argument("--model_path", "-m", required=True, help="Path to the trained model file (.h5 for Keras, .pth for PyTorch).")
parser.add_argument("--input_file", "-i", required=False, help="Path to input data file (NPZ, NPY, or CSV).")
parser.add_argument("--raw_input", action="store_true", help="Flag indicating the input_file is raw data needing preprocessing.")
args = parser.parse_args()

# Load class names to map predictions to labels
label_names = None
try:
    import json
    with open("experiments/results/classes.json", "r") as f:
        label_names = json.load(f)
except:
    pass

# Inference using TensorFlow/Keras model
if args.framework == "tf":
    import tensorflow as tf
    from data_preprocessing import preprocess_sequence  # to handle raw input if needed
    
    # Load the trained Keras model
    model = tf.keras.models.load_model(args.model_path)
    # Prepare input data
    if args.input_file:
        if args.input_file.endswith(".npz") or args.input_file.endswith(".npy"):
            data = np.load(args.input_file, allow_pickle=True)
            # If NPZ, assume data under key 'X' or first array
            if isinstance(data, np.lib.npyio.NpzFile):
                if "X" in data:
                    X_new = data["X"]
                else:
                    X_new = data[list(data.files)[0]]
            else:
                X_new = data  # NPY case
        elif args.input_file.endswith(".csv"):
            raw = np.loadtxt(args.input_file, delimiter=',')
            X_new = raw  # if raw, will handle below if raw_input flag
        else:
            raise ValueError("Unsupported file type for input.")
    else:
        raise ValueError("No input_file provided for inference.")
    # If the input is raw data, preprocess it
    if args.raw_input:
        print("Preprocessing raw input data...")
        if 'raw' in locals():
            # raw data loaded from CSV
            raw_sequence = raw
        else:
            # if input was npz with raw data
            raw_sequence = X_new
        # Ensure shape is (T, H, W[, C]) for preprocess_sequence
        if raw_sequence.ndim == 2:
            raw_sequence = raw_sequence[:, :, np.newaxis]
        X_proc = preprocess_sequence(raw_sequence)
        # Add batch dimension
        X_new = np.expand_dims(X_proc, axis=0)
    # If already processed, ensure batch dimension
    if X_new.ndim == 4:  # single sample with shape (T, H, W, C)
        X_new = np.expand_dims(X_new, axis=0)
    # Run inference
    preds = model.predict(X_new)
    pred_class_idx = int(np.argmax(preds, axis=1)[0])
    if label_names:
        pred_label = label_names[pred_class_idx]
    else:
        pred_label = str(pred_class_idx)
    print(f"Predicted activity: {pred_label}")

# Inference using PyTorch model
elif args.framework == "torch":
    import torch
    from model import HARModelTorch
    from data_preprocessing import preprocess_sequence
    
    # Before loading model, we need to know num_classes. We attempt to get from classes.json or ask user.
    if label_names:
        num_classes = len(label_names)
    else:
        num_classes = 0
    # Load input data
    X_new = None
    if args.input_file:
        if args.input_file.endswith(".npz") or args.input_file.endswith(".npy"):
            data = np.load(args.input_file, allow_pickle=True)
            if isinstance(data, np.lib.npyio.NpzFile):
                if "X" in data:
                    X_new = data["X"]
                else:
                    X_new = data[list(data.files)[0]]
            else:
                X_new = data
        elif args.input_file.endswith(".csv"):
            raw = np.loadtxt(args.input_file, delimiter=',')
            X_new = raw
        else:
            raise ValueError("Unsupported file type for input.")
    else:
        raise ValueError("No input_file provided for inference.")
    # Preprocess if raw
    if args.raw_input:
        print("Preprocessing raw input data for PyTorch model...")
        raw_seq = X_new
        if raw_seq.ndim == 2:
            raw_seq = raw_seq[:, :, np.newaxis]
        X_proc = preprocess_sequence(raw_seq)
        X_new = np.expand_dims(X_proc, axis=0)  # add batch dim
    else:
        if X_new.ndim == 4:
            X_new = np.expand_dims(X_new, axis=0)
    # Convert input to torch tensor and reshape dims for model [batch, T, C, H, W]
    X_tensor = torch.tensor(X_new, dtype=torch.float32).permute(0, 1, 4, 2, 3)
    # Instantiate model and load weights
    model = HARModelTorch(num_classes=num_classes if num_classes>0 else None, input_channels=1)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()
    # Run inference
    with torch.no_grad():
        outputs = model(X_tensor)
        _, pred_idx = torch.max(outputs, dim=1)
        pred_idx = int(pred_idx.item())
    if label_names:
        pred_label = label_names[pred_idx]
    else:
        pred_label = str(pred_idx)
    print(f"Predicted activity: {pred_label}")

""" Comments: This script is meant to be run on a case-by-case basis. Some things to note or modify:
The script expects a file input. You can modify it to accept data from other sources (like directly from a sensor or an array passed from another module).
For raw data input (--raw_input flag): we reuse preprocess_sequence from the preprocessing module to ensure consistency in data preparation. This means you should have data_preprocessing.py accessible in your PYTHONPATH or the same directory for import. If your data is already processed (i.e., in the same format as training data), you do not need --raw_input.
The output is printed to console. In a real scenario, you might want to return this from a function or send it to another system component. You could wrap this logic in an infer() function that can be imported and called.
If the model was trained on GPU (for PyTorch) and saved, loading on CPU is handled by map_location='cpu' to avoid errors when running inference on a machine without GPU.
We handle the mapping from class index to label using classes.json saved during training. If that file isn’t found or you prefer not to use it, the script will output numeric class indices instead of human-readable labels.
Batch vs single inference: We assume input_file contains one sample or a small batch. The code will run on the entire content of input_file at once. If you have multiple samples and want to iterate or do streaming, you might need to adjust the code to loop through input sequences.
This script can be extended for batch evaluation (e.g., computing accuracy on a set of test samples by providing an input file with many samples and comparing predictions to true labels if known). It’s a basic template to demonstrate loading and using the model."""
