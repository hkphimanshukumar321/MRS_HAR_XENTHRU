""" Purpose: Trains the CNN-LSTM model on the processed dataset, employing Transfer Learning (if applicable) and Leave-One-Subject-Out (LOSO) cross-validation. In LOSO, the model is trained multiple times, each time leaving one subject’s data out for testing and training on all other subjects. This evaluates how well the model generalizes to unseen subjects. The script can run using either TensorFlow/Keras or PyTorch, based on a command-line argument, so you can choose your preferred framework. Key Features:
Data Loading: Loads the preprocessed data from data/processed/processed_data.npz.
LOSO Cross-Validation: Automatically identifies unique subjects. For each subject:
Split data into training (all other subjects) and test (the one subject left out).
If needed, further split training into train/validation (not explicitly done here, but can be added).
Train the model from scratch (or using pre-trained weights if provided) on the training set.
Evaluate on the left-out subject’s test set.
Save the model and logs for that fold.
Transfer Learning: If a pre-trained model path is provided (via --pretrained_model), the script will load those weights into the model before training (for example, loading weights from a previous fold or a model trained on a related dataset). By default, no pre-trained model is used (training from scratch).
Logging and Saving: Training progress (loss, accuracy per epoch) is logged to a file in experiments/logs/, and final results (e.g., accuracy for each fold) are saved in experiments/results/. The trained model for each fold is saved (in Keras .h5 format or PyTorch .pth format). A summary of results can be derived from the logs.
Usage:
bash
Copy
python train.py --framework tf --epochs 20 --pretrained_model path/to/model_weights.h5
The --framework can be tf (TensorFlow) or torch (PyTorch). --epochs sets the number of training epochs for each fold. --pretrained_model is optional; if provided, it should point to a model file to load weights from for transfer learning (you can use a model from a previous run or a model trained on similar data). If using PyTorch, provide a .pth file; for Keras, a .h5. You can omit this or provide a path as needed. """

# code starts from here 


#!/usr/bin/env python3
"""
train.py: Train the HAR model with transfer learning and LOSO cross-validation.

Trains the CNN-LSTM model on the processed dataset, using Leave-One-Subject-Out cross-validation:
for each subject, train on all other subjects and test on the left-out subject.

Supports both TensorFlow (Keras) and PyTorch frameworks (select via --framework).
Also supports optional transfer learning by loading a pre-trained model (--pretrained_model).

Logs training progress to experiments/logs/ and saves models & results to experiments/results/.

Usage:
    python train.py --framework [tf|torch] --epochs <N> [--pretrained_model <path>]
Example:
    python train.py --framework torch --epochs 15
"""
import os
import argparse
import numpy as np

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train HAR model with LOSO cross-validation.")
parser.add_argument("--framework", "-f", choices=["tf", "torch"], required=True,
                    help="Framework to use: 'tf' for TensorFlow/Keras, 'torch' for PyTorch.")
parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of training epochs for each fold.")
parser.add_argument("--pretrained_model", "-p", type=str, default=None,
                    help="Path to pre-trained model weights for transfer learning (optional).")
args = parser.parse_args()

# Load processed data
data = np.load("data/processed/processed_data.npz", allow_pickle=True)
X = data["X"]      # shape (N_samples, T, H, W, 1)
y = data["y"]      # shape (N_samples,)
subjects = data["subject"]  # shape (N_samples,)
# Ensure labels are numeric (for training). If y is not numeric, create an encoding.
unique_labels = np.unique(y)
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
y_numeric = np.array([label_to_idx[label] for label in y], dtype=np.int64)
num_classes = len(unique_labels)

# Save the class names for later use (e.g., inference mapping)
os.makedirs("experiments/results", exist_ok=True)
import json
with open("experiments/results/classes.json", "w") as f:
    json.dump(list(unique_labels), f)

# Prepare logging
os.makedirs("experiments/logs", exist_ok=True)

# Training with TensorFlow (Keras)
if args.framework == "tf":
    import tensorflow as tf
    from model import build_tf_model
    
    # If a pre-trained Keras model is provided for transfer learning
    base_model_path = args.pretrained_model
    for subj in np.unique(subjects):
        subj_str = str(subj)
        print(f"\n=== LOSO Fold: Leaving out Subject {subj_str} ===")
        # Split train and test based on subject
        X_train = X[subjects != subj]
        y_train = y_numeric[subjects != subj]
        X_test = X[subjects == subj]
        y_test = y_numeric[subjects == subj]
        
        # Build model
        input_shape = X_train.shape[1:]  # (T, H, W, 1)
        model = build_tf_model(input_shape=input_shape, num_classes=num_classes)
        # If transfer learning, load weights (except final layer if classes differ)
        if base_model_path:
            try:
                model.load_weights(base_model_path, by_name=True, skip_mismatch=True)
                print(f"Loaded weights from {base_model_path} for transfer learning.")
            except Exception as e:
                print(f"Could not load pre-trained weights: {e}. Proceeding without transfer learning.")
        
        # Train the model
        log_path = f"experiments/logs/train_subject_{subj_str}.log"
        with open(log_path, "w") as log_file:
            for epoch in range(1, args.epochs+1):
                # Fit for one epoch (using whole training set as a single batch for simplicity here)
                history = model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
                loss = history.history['loss'][0]
                acc = history.history['accuracy'][0]
                log_msg = f"Epoch {epoch}/{args.epochs} - loss: {loss:.4f} - acc: {acc:.4f}"
                print(log_msg)
                log_file.write(log_msg + "\n")
            # Evaluate on test set after training
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            result_msg = f"Subject {subj_str} test_accuracy: {test_acc:.4f}"
            print(result_msg)
            log_file.write(result_msg + "\n")
        
        # Save the trained model for this fold
        model_save_path = f"experiments/results/model_subject_{subj_str}.h5"
        model.save(model_save_path)
        print(f"Saved trained model to {model_save_path}")
        
    print("\nTraining complete. Check experiments/logs/ for training details and experiments/results/ for models.")

# Training with PyTorch
elif args.framework == "torch":
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn
    import torch.optim as optim
    from model import HARModelTorch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # If transfer learning is used in PyTorch, load the state_dict
    pretrained_state = None
    if args.pretrained_model:
        if os.path.isfile(args.pretrained_model):
            pretrained_state = torch.load(args.pretrained_model, map_location=device)
            print(f"Pre-trained model weights loaded from {args.pretrained_model}")
        else:
            print(f"Pre-trained model path {args.pretrained_model} not found. Proceeding without.")
    
    for subj in np.unique(subjects):
        subj_str = str(subj)
        print(f"\n=== LOSO Fold: Leaving out Subject {subj_str} ===")
        # Split train/test for this subject
        train_idx = np.where(subjects != subj)[0]
        test_idx = np.where(subjects == subj)[0]
        X_train = X[train_idx]
        y_train = y_numeric[train_idx]
        X_test = X[test_idx]
        y_test = y_numeric[test_idx]
        # Convert to torch tensors
        # Note: X is (N, T, H, W, 1), we need (N, T, C, H, W)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 1, 4, 2, 3)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor  = torch.tensor(X_test, dtype=torch.float32).permute(0, 1, 4, 2, 3)
        y_test_tensor  = torch.tensor(y_test, dtype=torch.long)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        model = HARModelTorch(num_classes=num_classes, input_channels=1).to(device)
        # If transfer learning, load weights (excluding final layer if class count mismatched)
        if pretrained_state:
            try:
                model.load_state_dict(pretrained_state, strict=False)
                print("Transferred pre-trained weights to the model.")
            except Exception as e:
                print(f"Transfer learning failed: {e}")
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        log_path = f"experiments/logs/train_subject_{subj_str}.log"
        log_file = open(log_path, "w")
        for epoch in range(1, args.epochs+1):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                # accumulate stats
                epoch_loss += loss.item() * batch_X.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == batch_y).sum().item()
                total += batch_X.size(0)
            epoch_loss /= total
            epoch_acc = correct / total
            log_msg = f"Epoch {epoch}/{args.epochs} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}"
            print(log_msg)
            log_file.write(log_msg + "\n")
        # Evaluation on test set
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor.to(device))
            _, preds = torch.max(outputs, 1)
            test_acc = (preds.cpu() == y_test_tensor).sum().item() / len(y_test_tensor)
        result_msg = f"Subject {subj_str} test_accuracy: {test_acc:.4f}"
        print(result_msg)
        log_file.write(result_msg + "\n")
        log_file.close()
        # Save model
        model_save_path = f"experiments/results/model_subject_{subj_str}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved PyTorch model to {model_save_path}")
    
    print("\nTraining complete. Check experiments/logs/ for training details and experiments/results/ for models.")



""" Comments: This script is quite detailed. Key points to note or modify:
Data assumptions: It expects processed_data.npz with X, y, subject. If your data is split into separate train/test files or if you want to exclude certain activities, adjust the loading and filtering logic.
Label encoding: We convert labels to numeric (y_numeric). If your labels are already integers, this step isn’t needed. If you want to perform binary classification (say, one activity vs others), you’d adjust how y_numeric is created.
Hyperparameters: The number of epochs (--epochs argument), batch size (currently 32 in code), and learning rate (0.001 for Adam) are set to common defaults. These can be tuned. You may also add learning rate scheduling or early stopping for better training control.
Logging: We log per-epoch loss and accuracy to a file per subject (e.g., train_subject_2.log). This helps in reviewing training progress for each fold. You might integrate a proper logging library or TensorBoard for more advanced tracking.
Saving models: Each subject’s model is saved separately. If you only care about the final model (perhaps trained on all data or an average model), you can modify to train one final model. But for LOSO evaluation, we keep each fold model.
Transfer learning: The code attempts to load a pre-trained model if provided. We use model.load_weights(..., skip_mismatch=True) in Keras to avoid dimension mismatches (e.g., if the number of classes differs). In PyTorch, strict=False allows partial weight loading (so pre-trained CNN layers can load, but the LSTM or final layer can initialize fresh if sizes differ). You should ensure the pre-trained weights make sense (e.g., same architecture). For example, you might pre-train the CNN on some large radar dataset or ImageNet (treating radar frames as images) and then fine-tune with this script.
Cross-device training: The PyTorch part will use GPU if available (device=cuda). If using a CPU-only environment, training will be slower, so consider reducing complexity or using smaller data for testing.
After training completes, you will have logs to inspect training curves and model files for each subject. You can aggregate the results (e.g., average accuracy across subjects) by reading the log files or modifying the script to compute it on the fly. """
