#!/usr/bin/env python3
"""
experiment_saver.py

This module provides functions to save the trained model and evaluation metrics in the 'experiments/results/' directory.
It supports both TensorFlow/Keras and PyTorch models and saves metadata for reproducibility.

Usage:
    from experiment_saver import save_model, save_metrics

    # To save a trained model:
    save_model(model, framework="tensorflow", save_name="har_model_v1")

    # To save evaluation metrics:
    metrics = {"accuracy": 0.93, "confusion_matrix": [[50, 5], [3, 42]]}
    save_metrics(metrics, file_name="evaluation_metrics.json")
"""

import os
import json
from datetime import datetime

def create_results_folder(base_dir="experiments/results"):
    """
    Creates a timestamped results folder.
    
    Returns:
        str: Path to the created folder.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_folder = os.path.join(base_dir, timestamp)
    os.makedirs(result_folder, exist_ok=True)
    print(f"[ExperimentSaver] Results folder created at: {result_folder}")
    return result_folder

def save_model(model, framework, save_name="model", results_folder=None):
    """
    Saves the trained model.
    
    Parameters:
        model: Trained model (tf.keras.Model or torch.nn.Module).
        framework (str): 'tensorflow' or 'pytorch'.
        save_name (str): Base filename (without extension).
        results_folder (str, optional): Folder to save the model. If not provided, a new folder is created.
    
    Returns:
        str: File path to the saved model.
    """
    if results_folder is None:
        results_folder = create_results_folder()
    
    if framework.lower() == "tensorflow":
        file_path = os.path.join(results_folder, save_name + ".h5")
        model.save(file_path)
        print(f"[ExperimentSaver] TensorFlow model saved to {file_path}")
    elif framework.lower() == "pytorch":
        file_path = os.path.join(results_folder, save_name + ".pt")
        import torch
        torch.save(model.state_dict(), file_path)
        print(f"[ExperimentSaver] PyTorch model weights saved to {file_path}")
    else:
        raise ValueError("Unsupported framework. Choose 'tensorflow' or 'pytorch'.")
    
    return file_path

def save_metrics(metrics, file_name="metrics.json", results_folder=None):
    """
    Saves evaluation metrics to a JSON file.
    
    Parameters:
        metrics (dict): Evaluation metrics (e.g., accuracy, confusion matrix).
        file_name (str): Filename for metrics.
        results_folder (str, optional): Folder to save metrics. If not provided, a new folder is created.
    
    Returns:
        str: File path to the saved JSON file.
    """
    if results_folder is None:
        results_folder = create_results_folder()
    
    file_path = os.path.join(results_folder, file_name)
    serializable_metrics = {}
    for key, value in metrics.items():
        try:
            if hasattr(value, "tolist"):
                serializable_metrics[key] = value.tolist()
            else:
                serializable_metrics[key] = float(value)
        except Exception:
            serializable_metrics[key] = str(value)
    
    with open(file_path, "w") as f:
        json.dump(serializable_metrics, f, indent=4)
    
    print(f"[ExperimentSaver] Metrics saved to {file_path}")
    return file_path
