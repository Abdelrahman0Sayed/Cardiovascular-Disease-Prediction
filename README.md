# Cardiovascular Disease Prediction

## Project Overview

Welcome to the **Cardiovascular Disease Prediction** project! This Jupyter Notebook implements a machine learning pipeline to predict cardiovascular disease using the `cardio_train.csv` dataset. The project includes robust data preprocessing, training multiple machine learning models, hyperparameter tuning, and comprehensive performance evaluation with visualizations such as confusion matrices, precision-recall curves, and ROC curves.

## Table of Contents

- Project Overview
- Dataset
- Installation
- Usage
- Data Preprocessing
- Exploratory Data Analysis
- Models Implemented
- Evaluation Metrics
- Results and Visualizations
- Contributing
- License

## Dataset

The dataset (`cardio_train.csv`) contains health-related features for predicting cardiovascular disease. It includes 8,000 patient records (subset for efficiency) with the following features:

- **Age**: Age in days (converted to years).
- **Gender**: 1 (female), 2 (male).
- **Height**: Height in cm.
- **Weight**: Weight in kg.
- **ap_hi/ap_lo**: Systolic/diastolic blood pressure.
- **Cholesterol, Gluc**: Levels (1: normal, 2: above normal, 3: well above normal).
- **Smoke, Alco, Active**: Binary (0: no, 1: yes).
- **Cardio**: Target (0: no disease, 1: disease).

**Dataset Snapshot**:

## Installation

To run the notebook, install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn graphviz networkx
```

Ensure `cardio_train.csv` is in the project directory. For rendering Graphviz visualizations, install Graphviz:

```bash
# On Ubuntu
sudo apt-get install graphviz

# On macOS
brew install graphviz

# On Windows
# Download and install from https://graphviz.org/download/
```

## Usage

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies (see Installation).

3. Launch Jupyter Notebook:

   ```bash
   jupyter notebook ML-Project.ipynb
   ```

4. Execute cells sequentially to preprocess data, train models, and generate visualizations.

## Data Preprocessing

The preprocessing pipeline ensures data quality and model compatibility:

- **Cleaning**: Remove duplicates and drop `id` column.
- **Feature Transformation**:
  - Convert age (days to years).
  - Bin continuous features (`age`, `height`, `weight`, `ap_hi`, `ap_lo`) into categorical ranges to reduce model complexity.
- **Encoding**: Apply `LabelEncoder` to categorical features.
- **Splitting**: Divide data into 75% training and 25% testing sets.

**Preprocessed Data Example**:

## Exploratory Data Analysis

To understand the dataset, we visualized key statistics:

- **Class Distribution**: Balanced target classes (cardio: 0 vs. 1).
- **Feature Correlations**: Heatmap of correlations between features.
- **Age Distribution**: Histogram of age groups after binning.

## Models Implemented

Three machine learning models are trained with hyperparameter tuning via `GridSearchCV`:

1. **Decision Tree Classifier**:
   - Parameters: `max_depth` (2, 5, 10, 15, 20), `min_samples_split` (2, 5, 10, 20, 50), `max_leaf_nodes` (10, 30, 50, 100, 200).
2. **Random Forest Classifier**:
   - Parameters: `n_estimators` (100, 200), `max_depth` (None, 10, 20), `min_samples_split` (2, 5, 10), `min_samples_leaf` (1, 2, 4), `max_features` (sqrt, log2), `criterion` (gini, entropy).
3. **Support Vector Classifier (SVC)**:
   - Parameters: `kernel` (poly, rbf), `C` (0.001, 0.01, 0.1, 0.5, 1, 10).

## Evaluation Metrics

Models are evaluated using:

- **Accuracy**: Proportion of correct predictions.
- **Confusion Matrix**: Visualizes true positives, false positives, etc.
- **Precision-Recall Curve**: Assesses trade-off for imbalanced classes.
- **ROC Curve and AUC**: Measures model discrimination ability.
- **Classification Report**: Includes precision, recall, and F1-score.

## Results and Visualizations

The notebook generates insightful visualizations:

- **Confusion Matrices**: For each model, showing prediction performance.

- **ROC Curves**: Comparing AUC scores across models.

- **Model Performance Summary**:

| Model | Accuracy | AUC | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- | --- |
| Decision Tree | TBD\* | TBD\* | TBD\* | TBD\* | TBD\* |
| Random Forest | TBD\* | TBD\* | TBD\* | TBD\* | TBD\* |
| SVC | TBD\* | TBD\* | TBD\* | TBD\* | TBD\* |

- \*Note: Run the notebook to compute exact metrics.

**Key Insights**:

- Random Forest typically outperforms due to ensemble learning.
- Decision Tree is interpretable but may overfit without tuning.
- SVC is robust but computationally intensive.

---

**Author**: \[Abdelrahman Sayed Nasr\]\
**Contact**: \[abdelrahmansayed880@gmail.com\]\
**Last Updated**: May 9, 2025

---
