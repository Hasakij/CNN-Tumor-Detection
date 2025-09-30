# CNN-Tumor-Detection
Using PyTorch to analyze brain MRI images. This repository explores two approaches: a simple CNN model to classify images (tumor/no tumor) and a Faster R-CNN model to detect and locate the tumor's position.

## Overview
This project implements deep learning approaches for brain tumor detection from MRI scans:
- **Classification**: A Convolutional Neural Network (CNN) to classify MRI images as tumor or no tumor
- **Object Detection** (planned): Faster R-CNN for precise tumor localization and bounding box detection

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
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    ```

4.  **Install other dependencies:**
    Once PyTorch is installed, install the rest of the required packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To train the model and generate evaluation results, run the main script:```bash
python train.py
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
- Add data preprocessing and augmentation
- Develop evaluation metrics and visualization tools
- Create inference scripts for model deployment

## License
This project is licensed under the MIT License - see the LICENSE file for details.
