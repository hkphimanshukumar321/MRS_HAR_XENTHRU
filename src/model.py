"""Purpose: Defines the deep learning model architecture for HAR, utilizing a combination of Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) layers. We provide implementations in both TensorFlow (Keras) and PyTorch to give flexibility. The CNN layers extract spatial features from radar frames (e.g., range-Doppler or range-time intensity maps), and the LSTM layers capture temporal dynamics across frames. Model Architecture:
Convolutional layers: These handle each frame (2D radar data) by applying convolution and pooling, which can learn features like motion energy, shapes of reflections, etc. In the Keras model, we use TimeDistributed to apply Conv2D to each time frame. In PyTorch, we process the time dimension by flattening it into batch or looping through it.
LSTM layer: After CNN feature extraction, an LSTM processes the sequence of frame features to learn temporal patterns of movement corresponding to activities.
Fully Connected (Dense) layer: The final dense layer outputs class probabilities for the activity recognition (using softmax for multi-class classification).
Both frameworks’ models are kept similar in structure for consistency. You can tweak layer sizes or add more layers to improve performance. We design the model to accept input shape (T, H, W, C) for Keras or (N, T, C, H, W) for PyTorch (where N is batch size, T is sequence length, HxW is frame dimension, C is channels).

"""

#code satrts from here 

#!/usr/bin/env python3
"""
model.py: Define CNN-LSTM model for UWB radar HAR in both TensorFlow (Keras) and PyTorch.

Provides:
- build_tf_model(input_shape, num_classes): returns a compiled tf.keras model.
- HARModelTorch(num_classes, input_channels=1): a PyTorch nn.Module for the model.

The CNN layers extract features per frame, and the LSTM captures temporal patterns.
Modify layer sizes, filter counts, kernel sizes, or LSTM units to experiment with performance.
"""

# TensorFlow (Keras) model definition
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

def build_tf_model(input_shape, num_classes):
    """
    Build and compile a CNN-LSTM model using TensorFlow Keras.
    - input_shape: tuple, e.g., (T, H, W, C) without batch dimension.
    - num_classes: int, number of activity classes.
    Returns: compiled tf.keras Model.
    """
    model = Sequential(name="HAR_CNN_LSTM")
    # TimeDistributed Conv2D to process each time frame
    model.add(layers.TimeDistributed(layers.Conv2D(16, (3,3), activation='relu'), input_shape=input_shape))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2,2))))
    model.add(layers.TimeDistributed(layers.Conv2D(32, (3,3), activation='relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2,2))))
    model.add(layers.TimeDistributed(layers.Flatten()))
    # Now shape is (batch, T, features)
    model.add(layers.LSTM(64, return_sequences=False))  # LSTM over time steps -> outputs last hidden state
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    # Compile the model (using categorical crossentropy for multi-class classification)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# PyTorch model definition
import torch
import torch.nn as nn

class HARModelTorch(nn.Module):
    """
    CNN-LSTM model for HAR in PyTorch.
    - input_channels: number of channels in input frames (default 1 for grayscale radar frame).
    - num_classes: number of output classes for activity classification.
    """
    def __init__(self, num_classes, input_channels=1):
        super(HARModelTorch, self).__init__()
        # CNN feature extractor (example architecture parallel to Keras model)
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=0),  # Conv2D
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # After these layers, we'll flatten in forward pass
        )
        # LSTM: input_size will be determined after flattening CNN output
        # Let's assume input frame size is (H, W). After two pools of size 2, each dimension H and W are quartered.
        # If original frame size is HxW, after CNN part we have feature map of size 32 x (H/4) x (W/4).
        # So LSTM input length per time step = 32 * (H/4) * (W/4). We can compute it in forward.
        self.lstm_hidden = 64
        self.lstm_layers = 1
        # Define LSTM (we'll initialize it later when we know feature length, or compute on the fly)
        # To avoid complexity, we'll define LSTM with a max expected feature length and trim if needed:
        self.lstm = None  # will initialize in forward once we know feature dimension
        # Fully connected layers for classification
        self.fc1 = None  # placeholder, will init after LSTM if needed
        self.num_classes = num_classes

    def forward(self, x):
        # x shape: (batch, T, C, H, W)
        batch_size, seq_len, C, H, W = x.shape
        # Combine batch and sequence dimensions to apply CNN on all frames at once
        x = x.view(batch_size * seq_len, C, H, W)  # shape: (batch*T, C, H, W)
        x = self.cnn_layers(x)  # apply CNN layers
        x = x.view(x.size(0), -1)  # flatten, shape: (batch*T, feature_dim)
        feature_dim = x.size(-1)
        # Initialize LSTM if not already (we now know feature_dim)
        if self.lstm is None:
            self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=self.lstm_hidden, num_layers=self.lstm_layers, batch_first=True)
            # Initialize fully connected layer after LSTM
            self.fc1 = nn.Linear(self.lstm_hidden, self.num_classes)
        # Reshape back to (batch, seq_len, feature_dim) for LSTM
        x = x.view(batch_size, seq_len, feature_dim)
        # LSTM processing
        lstm_out, _ = self.lstm(x)      # lstm_out shape: (batch, seq_len, lstm_hidden)
        # Take the output from the last time step (assuming batch_first=True in LSTM)
        last_out = lstm_out[:, -1, :]   # shape: (batch, lstm_hidden)
        # Classification layer
        out = self.fc1(last_out)        # shape: (batch, num_classes)
        return out


""" Comments: The Keras model is compiled with sparse_categorical_crossentropy assuming labels are integers. If your labels are one-hot encoded, use categorical_crossentropy. Adjust input_shape when calling build_tf_model – for example, if your processed data has shape (T, H, W, 1) = (time frames, height, width, channel). The PyTorch model HARModelTorch sets up layers similarly; note we lazily initialize the LSTM and fully connected layer in the forward method once we know the flattened feature dimension (this is a simplified approach to avoid manually computing the CNN output size). In practice, you can calculate the feature dimension by passing a dummy tensor through cnn_layers and then define self.lstm and self.fc1 in __init__. You can modify the architecture easily: for example, add more convolutional layers for more complex feature extraction, use a deeper LSTM or GRU, or include dropout layers (layers.Dropout in Keras, nn.Dropout in PyTorch) to reduce overfitting. If you have a very large dataset or a complex model, consider using a GPU for training (the code will automatically use GPU if available for PyTorch, and for Keras you can set it up with TensorFlow). Transfer learning: If you want to use transfer learning (for instance, using a pre-trained CNN on a different radar or image dataset), you can load those weights into the CNN layers. In Keras, you would load a model or weights and then freeze some layers by setting layer.trainable = False. In PyTorch, you can load state dict for part of the model. Our training script demonstrates how to integrate this (loading a pre-trained model file if provided)."""
