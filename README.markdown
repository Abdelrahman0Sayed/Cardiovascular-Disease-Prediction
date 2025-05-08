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
| \*Note: Run the notebook to compute exact metrics. |  |  |  |  |  |

- | | | | | |

  | | | | | |

  | | | | | |

  | | | | | |

**Key Insights**:

- Random Forest typically outperforms due to ensemble learning.
- Decision Tree is interpretable but may overfit without tuning.
- SVC is robust but computationally intensive.

## Contributing

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-branch`).
3. Commit changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

Please adhere to PEP 8 standards and include tests for new features.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

**Author**: \[Your Name\]\
**Contact**: \[Your Email or GitHub Profile\]\
**Last Updated**: May 9, 2025

---

**Note**: The images referenced (e.g., `workflow.png`, `dataset_head.png`) are placeholders. To include them on GitHub:

1. Generate or capture screenshots of the dataset, preprocessed data, and visualizations from the notebook.
2. Save plots (e.g., correlation heatmap, class distribution) using `plt.savefig()` in the notebook.
3. Place images in an `images/` folder in the repository.
4. Update the README with actual paths to images.

For statistics graphs (e.g., class distribution, correlation heatmap), add the following code to your notebook and save the plots:

```python
# Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='cardio', data=cardio_data)
plt.title('Class Distribution of Cardiovascular Disease')
plt.savefig('images/class_distribution.png')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cardio_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.savefig('images/correlation_heatmap.png')
plt.show()

# Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(processed_data['age'], bins=8, kde=True)
plt.title('Age Distribution (Binned)')
plt.savefig('images/age_distribution.png')
plt.show()
```

Run these cells, ensure the `images/` folder exists, and push the images to your GitHub repository.