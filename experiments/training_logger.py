#!/usr/bin/env python3
"""
training_logger.py

This module provides a TrainingLogger class to log training metrics (loss, accuracy, etc.) during model training.
It creates a timestamped log folder under 'experiments/logs/' and writes metrics to a CSV file.

Usage:
    from training_logger import TrainingLogger
    
    # Initialize the logger at the start of training
    logger = TrainingLogger()
    
    # During training, after each epoch, log the metrics:
    logger.log_epoch(epoch=1, train_loss=0.345, train_accuracy=0.78, val_loss=0.412, val_accuracy=0.75)
"""

import os
import csv
from datetime import datetime

class TrainingLogger:
    def __init__(self, base_log_dir="experiments/logs"):
        """
        Initializes a new training log in a timestamped directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(base_log_dir, timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.log_dir, "training_log.csv")
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        
        print(f"[TrainingLogger] Log file created at: {self.log_file}")
    
    def log_epoch(self, epoch, train_loss, train_accuracy, val_loss=None, val_accuracy=None):
        """
        Logs metrics for a given epoch.
        
        Parameters:
            epoch (int): Epoch number.
            train_loss (float): Training loss.
            train_accuracy (float): Training accuracy.
            val_loss (float, optional): Validation loss.
            val_accuracy (float, optional): Validation accuracy.
        """
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_accuracy,
                             val_loss if val_loss is not None else "",
                             val_accuracy if val_accuracy is not None else ""])
        print(f"[TrainingLogger] Epoch {epoch} logged: Train Loss={train_loss}, Train Acc={train_accuracy}, "
              f"Val Loss={val_loss}, Val Acc={val_accuracy}")
