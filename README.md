# Human Detection with Custom HOG and SVM

A comprehensive computer vision project for robust human detection using custom Histogram of Oriented Gradients (HOG) feature extraction and Support Vector Machine (SVM) classification. Includes a modular PyQt6 GUI for dataset processing, feature extraction, model training, evaluation, and ablation studies.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [GUI Components](#gui-components)
- [Custom HOG Implementation](#custom-hog-implementation)
- [Ablation Studies](#ablation-studies)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview
This project implements a full pipeline for human detection in images using:
- **Custom HOG feature extraction** (with multiple gradient options, gamma correction, and normalization)
- **SVM classifier** (with hyperparameter tuning and ablation studies)
- **PyQt6-based GUI** for interactive data processing, feature extraction, model training, and evaluation
- **Extensive data preprocessing** for both human and nonhuman datasets

The project is designed for research, experimentation, and educational use in computer vision and machine learning.

---

## Features
- **Customizable HOG extraction**: Multiple gradient filters (Sobel, Scharr, Prewitt, etc.), normalization, gamma correction, and visualization
- **Flexible SVM training**: GUI-based model building, hyperparameter tuning, and model export
- **Ablation study support**: Easily test the impact of different HOG and SVM parameters
- **End-to-end GUI**: Import images, extract features, train models, and evaluate results visually
- **Jupyter notebook**: For reproducible experiments and advanced analysis

---

## Directory Structure
```
Project/CITS4402/
├── Analysis/                # Data analysis and processing scripts
│   └── data/               # Raw and processed datasets
├── GUI.py                  # Main entry point for the PyQt6 GUI
├── GUI/                    # (Reserved for additional GUI modules)
├── Other/
│   ├── custom_hog_methods.py   # Custom HOG feature extraction implementation
│   ├── GUIlib/             # GUI logic for HOG, SVM, and evaluation
│   └── data/               # Additional datasets and processed features
├── Test_Examples/          # Example test sets for evaluation
├── svm_model_customHOG.pkl # Example trained SVM model
├── notebook.ipynb          # Main Jupyter notebook for experiments
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── LICENSE                 # License file
```

---

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>/Project/CITS4402
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   - Python 3.11+ is recommended.
   - Jupyter is required for running the notebook.

---

## Usage
### 1. **Run the GUI**
```bash
python GUI.py
```
- Use the GUI to:
  - Extract HOG features from images
  - Build and train SVM models
  - Evaluate models on test sets
  - Visualize HOG features and results

### 2. **Jupyter Notebook**
- Open `notebook.ipynb` for step-by-step experiments, ablation studies, and advanced analysis.

---

## Data Preparation
- **Raw data**: Place human and nonhuman images in `Analysis/data/raw/human` and `Analysis/data/raw/nonhuman` (or `Other/data/raw/human1`, etc.).
- **Processed data**: Scripts and the GUI will generate resized, cropped, and deduplicated datasets in `processed/` folders.
- **Feature files**: HOG features are saved as `.txt` files for SVM training and evaluation.

---

## GUI Components
- **HOG GUI**: Configure and extract HOG features from images, save to disk
- **Build GUI**: Import HOG features, train SVM models, tune hyperparameters, save models
- **Eval GUI**: Import models and test sets, evaluate performance, visualize metrics (accuracy, precision, recall, F1, DET curves)

---

## Custom HOG Implementation
- Located in `Other/custom_hog_methods.py`
- Supports multiple gradient filters: Sobel, Scharr, Prewitt, Roberts, DoG, Central Difference
- Options for gamma correction, Gaussian blur, block normalization, signed/unsigned gradients
- Visualization utility for HOG features
- Designed for extensibility and experimentation

---

## Ablation Studies
- Easily modify HOG and SVM parameters in the GUI or notebook
- Compare the impact of different feature extraction and model settings
- Results and analysis are documented in `notebook.ipynb`

