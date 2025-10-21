# Brain Tumor Classification using a Convolutional Neural Network

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) for multi-class brain tumor classification from MRI scans. The model is trained to distinguish between four different classes: **Glioma**, **Meningioma**, **Pituitary** tumor, and **No Tumor**.

## Overview

This project implements a deep learning approach for the classification of brain tumors from MRI images.

-   **Multi-Class Classification**: A custom CNN is built from scratch and trained to classify MRI images into one of four categories:
    -  Glioma
    -  Meningioma
    -  No Tumor
    -  Pituitary
-   **Architecture**: The model uses a standard CNN architecture with convolutional layers for feature extraction (enhanced with Batch Normalization and ReLU activations) and fully connected layers for the final classification.

## Requirements
See `requirements.txt` for a full list of dependencies. Main requirements:
- PyTorch
- torchvision
- scikit-learn
- OpenCV (opencv-python)
- Matplotlib & Seaborn (for plotting)
- KaggleHub (for data download)

## Installation
Follow these steps to set up the project environment.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/CNN-Tumor-Detection.git
    cd CNN-Tumor-Detection
    ```

2.  **(Recommended) Create and activate a virtual environment:**
    For Conda:
    ```bash
    conda create --name mri_env python=3.10
    conda activate mri_env
    ```

3.  **Install PyTorch:**
    This project was developed using PyTorch with **CUDA 12.1**. Please visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to find the installation command that matches your system's hardware (CUDA or CPU).

    The recommended command for a CUDA 12.1 environment is:
    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    ```

4.  **Install other dependencies:**
    Once PyTorch is installed, install the rest of the required packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To train the model and generate evaluation results, run the main script:
```bash
python train.py
```
## Project Structure
```
CNN-Tumor-Detection/
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies
├── LICENSE            # MIT License
├── train.py           # Training script for CNN model
└── .gitignore         # Git ignore patterns
```

## Future Plans
- Implement Faster R-CNN for object detection to locate tumor regions

## License
This project is licensed under the MIT License - see the LICENSE file for details.
