# Heart Disease Prediction Project

## Overview
This project demonstrates a complete machine learning pipeline for predicting heart disease using clinical patient data. It includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and comparison of different classifiers.

## Dataset
- **Source**: `heart1.csv` (processed UCI Heart Disease dataset)
- **Features**: 16 columns including patient demographics, medical measurements, and diagnostic indicators
- **Target**: `num` (0-4 severity scale, converted to binary: 0 = no disease, >0 = disease)
- **Size**: 920 rows, 16 columns

## Workflow

### 1. Data Loading and Preprocessing (`code.py`)
- Loads heart1.csv dataset
- Performs duplicate removal
- Creates binary target variable (`target`: 0 or 1)
- Handles missing values (imputation with column means for numeric features)
- Feature selection (only numeric columns)

### 2. Exploratory Data Analysis (EDA)
- **Plots Generated**:
  - Target variable distribution (bar plot)
  - Age distribution by heart disease status (histplot)
  - Feature correlation heatmap
- Saved to `output/` folder as PNG images

### 3. Model Training and Evaluation
- **Algorithm**: Logistic Regression
- **Train/Test Split**: 80/20 with random state 42
- **Feature Scaling**: StandardScaler
- **Evaluation Metrics**: Accuracy, Confusion Matrix, Precision, Recall, F1-Score

### 4. Model Comparison (`compare_sklearn_models.py`)
- Compares 4 models: Logistic Regression, Decision Tree, Random Forest, KNN
- Same preprocessing as above
- **Caution**: Models show 100% accuracy suggesting overfitting due to dataset characteristics

## Requirements
- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
- Install with: `pip install pandas numpy matplotlib seaborn scikit-learn`

## Files
- `code.py`: Main script for Logistic Regression model
- `compare_sklearn_models.py`: Script for comparing multiple ML models
- `heart1.csv`: Dataset file
- `output/`: Output directory containing:
  - `code_target_distribution.png`: Target class distribution
  - `code_age_distribution.png`: Age vs heart disease
  - `code_correlation_heatmap.png`: Feature correlations
  - `code_confusion_matrix.png`: Confusion matrix heatmap
  - `code_results.txt`: Logistic Regression metrics
  - `compare_barplot.png`: Model comparison bar chart
  - `compare_results.txt`: Model accuracies
- `README.md`: This documentation file

## Usage

### Run Individual Model Analysis
```bash
python3 code.py
```
Outputs all results to `output/` folder without displaying plots.

### Run Model Comparison
```bash
python3 compare_sklearn_models.py
```
Compares multiple models and saves comparison outputs.

## Results Summary

### Logistic Regression Performance (code.py)
- **Accuracy**: 75.00%
- **Confusion Matrix**:
  | Predicted No | Predicted Yes |
  |--------------|---------------|
  | 58 (True Negative) | 17 (False Positive) |
  | 29 (False Negative) | 80 (True Positive) |

- **Classification Report**:
  - Class 0 (No Disease): Precision=0.67, Recall=0.77, F1=0.72
  - Class 1 (Disease): Precision=0.82, Recall=0.73, F1=0.78

### Model Comparison Results (compare_sklearn_models.py)
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 100.00% |
| Decision Tree | 100.00% |
| Random Forest | 100.00% |
| K-Nearest Neighbors | 92.39% |

⚠️ **Note**: The high accuracies (100% for most models) indicate potential overfitting. The dataset may be small or have characteristics leading to over-optimization on the training set.

## Data Processing Details
- **Duplicates**: None found/removed
- **Missing Values**: Filled with column means (817 missing values in numeric columns)
- **Feature Engineering**: Created binary target from multi-class diagnosis
- **Scaling**: StandardScaler applied after train/test split

## Visualizations
All plots are automatically saved to the `output/` folder:
- Target distribution shows class balance
- Age distribution highlights disease prevalence by age groups
- Correlation heatmap identifies feature relationships (target correlation shown)

## Future Improvements
1. Cross-validation to mitigate overfitting
2. Feature engineering with categorical variables encoding
3. Hyperparameter tuning
4. Larger dataset or train/validation/test splits
5. Additional evaluation metrics and curves (ROC, Precision-Recall)

## Author
Dynamix Networks Intern (Heart Disease Prediction Task)

## License
