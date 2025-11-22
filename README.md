# Heart Disease Prediction Project

Advanced machine learning workflows for heart disease prediction using clinical data.

## Overview

This project provides two Python scripts for heart disease prediction:

1. **code.py**: Simple Logistic Regression model with EDA and visualization
2. **compare_sklearn_models.py**: Advanced model comparison with hyperparameter tuning, cross-validation, and comprehensive metrics

Both scripts output results to an `output/` folder for organized storage.

## Dataset

- **File**: `heart1.csv`
- **Source**: Processed UCI Heart Disease dataset
- **Features**: 16 columns (demographics, lab results, ECG data, etc.)
- **Target**: `num` column (0-4 severity; binary classification as 0=healthy, 1=disease)
- **Records**: 920 samples

## Requirements

Install dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or manually:
- pandas>=1.3.0
- numpy>=1.21.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- scikit-learn>=1.0.0

Optional dependencies (for `compare_sklearn_models.py`):
- xgboost
- lightgbm
(If not available, skipped automatically)

## Script 1: code.py (Simple Analysis)

### Description
- Loads `heart1.csv`
- Cleans data (removes duplicates, selects numeric features)
- Generates 3 exploratory plots
- Trains Logistic Regression with accuracy/confusion matrix/classification report
- Saves all outputs to `output/` folder

### Usage
```bash
python3 code.py
```

### Outputs (in `output/` folder)
- `code_target_distribution.png`: Class distribution (healthy vs disease)
- `code_age_distribution.png`: Age group distribution by disease status
- `code_correlation_heatmap.png`: Feature correlation heatmap
- `code_confusion_matrix.png`: Confusion matrix heatmap
- `code_results.txt`: Model accuracy, classification report, confusion matrix

### Results (Sample)
- Logistic Regression Accuracy: 75.00%
- Balanced F1-scores for both classes

## Script 2: compare_sklearn_models.py (Advanced Comparison)

### Description
- Loads and preprocesses data with leak-safe feature selection
- Compares 6 ML models with RandomizedSearchCV hyperparameter tuning
- 5-fold stratified cross-validation
- Generates ROC/PR curves for each model
- Saves best model as joblib file

### Models Evaluated
1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. SVM
5. K-Nearest Neighbors
6. Decision Tree

### Tuning
- Uses ROC-AUC scoring
- Randomized search with configurable iterations (`N_ITER_SEARCH`)
- StratifiedKFold for robust CV

### Usage
```bash
python3 compare_sklearn_models.py
```
Note: Runtime ~5-10 minutes depending on hardware.

### Outputs (in `output/` folder)
- `compare_results.txt`: Detailed metrics for each model (CV scores, test accuracy, ROC AUC, confusion matrices, classification reports)
- `best_model_tuned.joblib`: Serialized best-performing model pipeline
- `combined_roc.png`: Overlay ROC curves for all models
- `compare_barplot.png`: Test accuracy bar chart comparison
- `pr_curves/` folder: Individual Precision-Recall curves (PNG for each model)
- `roc_curves/` folder: Individual ROC curves (PNG for each model)

### Results (Sample from Recent Run)
| Model | CV ROC-AUC | Test Accuracy | Test ROC-AUC |
|-------|------------|---------------|---------------|
| LogisticRegression | 0.898 | 85.87% | 0.898 |
| RandomForest | 0.912 | 83.15% | 0.897 |
| GradientBoosting | 0.895 | 81.52% | 0.879 |
| SVM | N/A | 83.70% | 0.876 |
| KNN | 0.796 | 75.54% | 0.715 |
| DecisionTree | 0.825 | 77.17% | 0.786 |

- Best model typically RandomForest or GradientBoosting by ROC-AUC
- High accuracies but potential overfitting due to dataset size
- SVM and KNN may underperform due to scaling/feature space

## File Structure

```
.
├── README.md
├── code.py                           # Simple Logistic Regression
├── compare_sklearn_models.py         # Advanced model comparison
├── heart1.csv                        # Dataset
├── code_fixed.py                     # Backup/version
├── output/                           # Generated outputs
│   ├── code_*.png                    # EDA plots
│   ├── code_results.txt              # LR metrics
│   ├── compare_*.png                 # Comparison plots
│   ├── combined_roc.png              # ROC overlay
│   ├── pr_curves/                    # PR curve PNGs
│   ├── roc_curves/                   # ROC curve PNGs
│   ├── compare_results.txt           # Detailed comparison
│   └── *.joblib                      # Saved models
└── .gitignore                        # Version control ignores
```

## Key Features

- **Automated Leak Detection**: Removes features highly correlated with target
- **Pipeline Standardization**: Consistent preprocessing (impute + scale) for all models
- **Robust Evaluation**: CV, confusion matrix, ROC/PR curves, classification reports
- **Organization**: All results saved neatly without console overload
- **Reproducibility**: Fixed random seeds, stratified splits
- **Extensibility**: Easy to add new models/parameters

## Comments on Performance

- **Overfitting Concern**: High accuracies suggest potential overfitting given small dataset
- **Logistic Regression Baseline**: 75% accuracy shows sensible baseline
- **Tree Ensembles**: Often best performers but may overfit without regularization
- **SVM**: Strong classifier when tuned, good for binary classification
- **KNN**: Sensitive to scaling/feature selection, often lower performance
- **Decision Tree**: Interpretable but prone to overfitting

## Future Enhancements

1. **Cross-Validation Strategies**: Increase k-folds, add test sets
2. **Feature Engineering**: Encode categorical variables, create derived features
3. **Class Imbalance Handling**: SMOTE for synthetic minority oversampling
4. **Advanced Models**: Neural networks, stacking ensembles
5. **Deployment**: Web app with Streamlit, model API with FastAPI
6. **Reproducibility**: Docker containerization

## Author & License

Educational project for heart disease prediction analysis.
Generated for Saurav Anand - Dynamix Networks Internship.
No commercial license; for learning purposes only.

## Run Summary

To reproduce full analysis:
```bash
python3 code.py                    # Basic analysis
python3 compare_sklearn_models.py  # Advanced comparison
```
