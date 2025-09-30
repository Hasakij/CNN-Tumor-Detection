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
- opencv-python

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Train the CNN classification model:
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
- Add data preprocessing and augmentation
- Develop evaluation metrics and visualization tools
- Create inference scripts for model deployment

## License
This project is licensed under the MIT License - see the LICENSE file for details.
