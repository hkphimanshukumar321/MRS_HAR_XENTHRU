# MRS_HAR_XENTHRU
This project deals with  implementation of Human Activity  Recognition  using Xenthru  UWB RADAR 

# Human Activity Recognition (HAR) using Ultra-Wideband (UWB) Radar

## Overview

This project develops a robust system for recognizing human activities utilizing Ultra-Wideband (UWB) radar signals combined with advanced deep learning techniques. Leveraging the advantages of UWB radar, such as non-invasive sensing, privacy preservation, and the ability to operate effectively in various environmental conditions, we implement a comprehensive end-to-end pipeline. The system includes data acquisition, preprocessing, position estimation, model training and evaluation, real-time deployment, and visualization.

## Project Goals

- Accurately classify human activities using UWB radar data.
- Implement real-time activity recognition optimized for edge devices.
- Provide visual feedback through a real-time dashboard.
- Evaluate system robustness using Leave-One-Subject-Out (LOSO) cross-validation.

## Repository Structure

```
HAR_UWB_Project/
├── data/                      # Dataset storage
│   ├── kaggle/                # Raw datasets from Kaggle
│   └── processed/             # Preprocessed data ready for modeling
├── experiments/               # Outputs and evaluation metrics
│   ├── logs/                  # Training and inference logs
│   └── results/               # Trained models and results
├── notebooks/                 # Jupyter notebooks for exploratory analysis
├── src/                       # Main source code
│   ├── fetch_data.py          # Script to download dataset from Kaggle
│   ├── data_preprocessing.py  # Preprocessing steps (stacking, background subtraction, normalization)
│   ├── compute_position.py    # Trilateration-based position computation
│   ├── model.py               # CNN-LSTM architecture definition (TensorFlow & PyTorch)
│   ├── train.py               # Training script with LOSO cross-validation and transfer learning
│   ├── infer.py               # Script for loading models and running inference
│   ├── real_time_inference.py # Optimized inference for Raspberry Pi and hardware accelerators
│   └── dashboard.py           # Flask dashboard for visualization of results
├── docs/                      # Detailed documentation
│   ├── project_overview.md    # High-level description and project usage
│   └── design_docs.md         # Detailed technical and architectural design decisions
├── requirements.txt           # Python dependencies
└── .gitignore                 # Specifies files to ignore for Git
```

## Setup and Installation

Clone the repository:
```bash
git clone <your-repository-url>
cd HAR_UWB_Project
```

Create and activate a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Acquisition

Fetch dataset from Kaggle using provided credentials and the dataset identifier:
```bash
python src/fetch_data.py --dataset user123/uwb-har-dataset
```

## Data Preprocessing

Run preprocessing to prepare data for modeling (stacking radar frames, background subtraction, and normalization):
```bash
python src/data_preprocessing.py
```

Processed data will be stored in `data/processed/`.

## Training the Model

Train the CNN-LSTM model using LOSO cross-validation:
```bash
python src/train.py --framework tf --epochs 20
```

Supported frameworks:
- TensorFlow (tf)
- PyTorch (torch)

You may also specify pretrained models for transfer learning:
```bash
python src/train.py --framework torch --epochs 20 --pretrained_model <path_to_weights>
```

## Model Inference

Perform inference on new or test data samples:
```bash
python src/infer.py --framework tf --model_path experiments/results/model_subject_1.h5 --input_file sample.npz
```

## Real-Time Deployment

Deploy real-time inference on Raspberry Pi using hardware accelerators:

- Google Coral Edge TPU:
```bash
python src/real_time_inference.py --accelerator coral
```

- Intel Neural Compute Stick 2:
```bash
python src/real_time_inference.py --accelerator ncs2
```

- CPU Only:
```bash
python src/real_time_inference.py --accelerator cpu
```

## Visualization Dashboard

Launch the Flask-based web dashboard to monitor real-time activity recognition:
```bash
python src/dashboard.py
```

Access the dashboard via a web browser:
```
http://<device_ip>:5000
```

## Exploratory Analysis

Use the provided Jupyter notebook to analyze and visualize the dataset and results:
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

## Contributions

Contributions are welcome. Please open issues or pull requests clearly describing proposed changes or improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
