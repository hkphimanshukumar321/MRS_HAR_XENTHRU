# Project Overview

## Introduction
This project is a Human Activity Recognition (HAR) system using Ultra-Wideband (UWB) radar data. Instead of cameras or wearable sensors, we use RF signals from UWB radars to identify human activities (e.g., walking, sitting, falling) in a contactless manner. The goal is to leverage the unique advantages of UWB radar – such as privacy preservation and robust sensing through obstacles – to recognize activities in indoor environments.

## Dataset
The dataset originates from Kaggle (link or name of dataset). It includes multiple subjects performing a variety of activities in a lab setting, recorded via UWB radar sensors (and possibly additional sensors for ground truth). The raw data consists of radar returns over time, possibly represented as range-time intensity maps or similar structures. There are [N] distinct activities performed by [M] subjects.

Key data characteristics:
- **Sampling Rate:** (e.g., 20 frames per second of radar data)
- **Spatial Dimensions:** Each radar frame has a size of (H x W) = (range bins x angle bins) or other representation.
- **Number of Sensors:** (e.g., 3 UWB radar modules placed around the area)
- **Activities:** List of activities (e.g., walking, sitting, falling, picking up object, etc.)
- **Class Balance:** Each activity has roughly X samples per subject, etc.

## Preprocessing
Raw radar data often contains static reflections from furniture or walls. We apply **background subtraction** to remove static clutter, enhancing the moving human's signature. Multi-sensor data is **stacked** and combined (via max-pooling) to create a unified view of the target's motion. The data is then normalized to a consistent scale. The preprocessing outputs sequences of frames for each activity instance, labeled with the activity and the subject ID. These sequences form the input for the model.

Our preprocessing pipeline is designed to be modular – any changes (like using a different filtering technique or adding data augmentation) can be done in `data_preprocessing.py` without affecting the model definition or training procedure.

## Model Architecture
We employ a hybrid **CNN-LSTM** neural network:
- The CNN portion (convolutional layers) acts on each radar frame (image) to extract spatial features (like the distribution of reflection energy, shapes indicating posture, etc.).
- The LSTM portion takes the sequence of frame features and learns temporal patterns (e.g., the motion trajectory). This is crucial since activities are dynamic and cannot be recognized from a single frame alone.
- The model outputs a probability distribution over the activity classes for a given sequence.

By using CNN-LSTM, we leverage both spatial and temporal information. This architecture has proven effective in many sequence recognition tasks (like action recognition in videos, which is analogous to our problem).

We implemented the model in both TensorFlow and PyTorch to give flexibility for training and deployment. The choice might depend on available hardware or preference.

## Training Strategy
We use **Leave-One-Subject-Out (LOSO) cross-validation** to evaluate model performance robustly. In LOSO, we train the model multiple times, each time excluding all data from one subject as the test set, and training on the data from all other subjects. This tests the model's ability to generalize to an unseen person – a critical measure for HAR systems, since in real deployments, the system may encounter users it was not trained on.

We also incorporate **transfer learning**: if there are pre-trained models (either from similar tasks or previous folds), we can initialize the model with those weights to speed up convergence and potentially improve performance. For example, one might pre-train the CNN on a large generic activity dataset or even ImageNet (treating radar data as images), or use a model from a previous study.

Training details:
- Optimizer, loss: Adam optimizer, cross-entropy loss for classification.
- Epochs: (e.g., 10-20 epochs per fold, as specified by the user running `train.py`).
- Batch size: 32 (though in LOSO, dataset size per fold might be small).
- Metrics: Accuracy is the primary metric, but we also log loss. One could compute precision/recall per class if needed.

After training, we expect to have models that generalize across subjects and can be used for real-time inference.

## Real-Time Deployment
A major advantage of this HAR system is that it can run in real time on edge devices. We prepared a `real_time_inference.py` that is optimized for devices like Raspberry Pi. By using hardware accelerators (Google Coral EdgeTPU or Intel NCS2), even the relatively complex CNN-LSTM model can run quickly enough for real-time feedback (with inference times in the tens of milliseconds range, typically).

Important considerations for deployment:
- The model is converted to a compact format (TFLite or OpenVINO IR) and quantized where possible for efficiency.
- The system reads data from the actual UWB radar sensor in a streaming fashion.
- We maintain a rolling window of the last T frames (the sequence length needed) and continuously classify the current window. This introduces a short latency (the length of the window), but allows continuous classification.
- The system can also compute the user's position via trilateration, enabling location-aware activity recognition (e.g., know where an event like a fall occurred).

## Usage Guide
**Offline Training and Evaluation:**
1. **Setup:** Ensure you have the required libraries installed (TensorFlow 2.x, PyTorch 1.x, numpy, etc.) and Kaggle API token for data download.
2. **Data Download:** Run `python fetch_data.py --dataset <dataset-id>` to get the raw data.
3. **Preprocess:** Run `python data_preprocessing.py` to generate the processed dataset.
4. **Train:** Run `python train.py --framework tf --epochs 15` (or `--framework torch`) to train the model. Adjust epochs or add `--pretrained_model` if you have weights.
5. **Evaluate:** During training, LOSO results are printed. Check `experiments/logs/` for detailed logs. You can compute overall accuracy by averaging per-subject accuracy from the logs.
6. **Test Inference:** Use `python infer.py --framework tf --model_path experiments/results/model_subject_X.h5 --input_file sample.npz` to test the model on a sample. Replace parameters as needed.
7. **Explore Data:** Open `notebooks/exploratory_analysis.ipynb` in Jupyter to see data visuals and additional analysis.

**Real-Time Deployment:**
1. Convert the best model to the required format:
   - For Coral: use the TensorFlow Lite converter and EdgeTPU compiler to get `model_edgetpu.tflite`.
   - For NCS2: use OpenVINO to get `model.xml` and `model.bin`.
2. Copy the model to `experiments/results/` on the Raspberry Pi (or run training directly on a powerful machine and just deploy the model file).
3. Connect and configure the UWB radar sensor on the device.
4. Run `python real_time_inference.py --accelerator coral` (or `--accelerator ncs2`) on the device. It should start the inference loop.
5. In parallel, run `python dashboard.py` to start the web dashboard.
6. Open a browser to the device's dashboard (for example, `http://raspberrypi.local:5000`) to monitor the recognized activities in real time.
7. Ensure you have adequate cooling for the device if running for extended periods, as the computation can be intense even with accelerators.

## Results and Conclusion
*(This section would summarize the performance results, e.g., average accuracy, any difficulties in distinguishing certain activities, etc. If the project is for a report or assignment, include interpretation of results here.)*

The HAR system achieved an average accuracy of **XX%** across LOSO cross-validation, demonstrating effective generalization to unseen subjects in most cases. Activities such as Y and Z were recognized with high precision, while activity W was sometimes confused with V (possibly due to similar motion patterns on radar). These results are promising, indicating that UWB radar has potential for accurate HAR. Further improvements could include collecting more data to cover a wider range of subjects and environments, tuning the model architecture, or incorporating additional sensor modalities (like multiple radar frequencies or adding inertial sensors for a hybrid system).

This project showcases an end-to-end pipeline from data acquisition to real-time deployment, highlighting how modern deep learning techniques can be applied to novel sensor data for practical applications like elder care (fall detection) or smart home automation using privacy-preserving sensors.

