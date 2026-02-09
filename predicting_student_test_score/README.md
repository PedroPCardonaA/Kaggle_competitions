# Predicting Student Test Score

This repository contains two solutions for the [Kaggle Playground Series S6E1](https://www.kaggle.com/competitions/playground-series-s6e1/overview) competition, which focuses on predicting student test scores based on various features.

## Competition Overview

The goal is to predict exam scores for students using tabular data containing demographic, academic, and behavioral features.

## Solutions

### Solution 1: Random Forest with PCA

**File:** `train_model.ipynb`

This solution uses a traditional machine learning approach combining:
- **Principal Component Analysis (PCA)** for dimensionality reduction
- **Random Forest Regressor** with hyperparameter tuning via GridSearchCV
- **Comprehensive preprocessing pipeline** including:
  - Standard scaling
  - Label encoding for categorical features
  - Cross-validation for robust model evaluation

**Key Features:**
- Exploratory data analysis and feature engineering
- Hyperparameter optimization using GridSearchCV
- Model performance evaluation with multiple metrics (MSE, MAE, R²)
- Generates submission file: `data/submission_rf_pca.csv`

### Solution 2: TabNet (Deep Learning)

**File:** `transformer_tabular.py`

This solution leverages a deep learning approach using TabNet, a transformer-based architecture specifically designed for tabular data:
- **TabNet Regressor** from PyTorch TabNet
- **Attention mechanism** for interpretable feature importance
- **Early stopping** with validation monitoring
- **Optimized hyperparameters:**
  - n_d=64, n_a=64 (dimension parameters)
  - n_steps=5 (number of decision steps)
  - gamma=1.5 (relaxation parameter)
  - Max epochs: 200 with patience=20

**Key Features:**
- Automated categorical feature encoding
- Train-validation split for early stopping
- Batch processing with virtual batch size
- Generates submission file: `data/submission_tabnet.csv`

## Project Structure

```
├── data/
│   ├── train.csv                    # Training dataset
│   ├── test.csv                     # Test dataset
│   ├── sample_submission.csv        # Submission format
│   ├── submission_rf_pca.csv        # Random Forest predictions
│   └── submission_tabnet.csv        # TabNet predictions
├── data_analysis.ipynb              # Exploratory data analysis
├── train_model.ipynb                # Random Forest + PCA solution
├── transformer_tabular.py           # TabNet solution
└── README.md                        # This file
```

## Requirements

### For Random Forest Solution:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

### For TabNet Solution:
- pandas
- numpy
- pytorch-tabnet
- torch

## Usage

### Running the Random Forest Model

Open and execute `train_model.ipynb` in Jupyter Notebook or JupyterLab:
```bash
jupyter notebook train_model.ipynb
```

### Running the TabNet Model

Execute the TabNet script:
```bash
python transformer_tabular.py
```

## Data Analysis

The `data_analysis.ipynb` notebook contains comprehensive exploratory data analysis including:
- Data statistics and distributions
- Missing value analysis
- Correlation analysis
- Target variable distribution
- Feature visualization

## Results

Both models generate predictions in the Kaggle submission format with columns:
- `id`: Test sample identifier
- `exam_score`: Predicted test score

## Author

Pedro

## Competition Link

[Playground Series - Season 6, Episode 1](https://www.kaggle.com/competitions/playground-series-s6e1/overview)
