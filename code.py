import os
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

extra_models = {}

RANDOM_STATE = 42
CV_SPLITS = 5
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUTPUT_DIR, 'compare_results.txt')
BARPLOT_FILE = os.path.join(OUTPUT_DIR, 'compare_barplot.png')
BEST_MODEL_FILE = os.path.join(OUTPUT_DIR, 'best_model.joblib')

data = pd.read_csv('heart1.csv')
data['target'] = (data['num'] > 0).astype(int)

X = data.select_dtypes(include=[np.number]).drop(columns=['num', 'target'], errors='ignore')
y = data['target']

leakage_threshold = 0.01
suspects = [c for c in X.columns if is_numeric_dtype(X[c]) and np.mean(X[c].values == y.values) > leakage_threshold]
drop_cols = []
if 'id' in X.columns:
    drop_cols.append('id')
drop_cols += suspects
drop_cols = list(dict.fromkeys(drop_cols))
if drop_cols:
    print("Dropping columns:", drop_cols)
X = X.drop(columns=drop_cols, errors='ignore')

print("Missing values (per feature):")
print(X.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=RANDOM_STATE, stratify=y)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

models = {
    'LogisticRegression': LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(probability=True, random_state=RANDOM_STATE),
    'GradientBoosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
}
models.update(extra_models)

results = []
test_accuracies = {}

for name, estimator in models.items():
    print(f"\n=== Evaluating {name} ===")
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('clf', estimator)
    ])
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    try:
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
    except Exception as e:
        print(f"CV failed for {name}: {e}")
        cv_scores = np.array([np.nan]*CV_SPLITS)
    cv_mean = np.nanmean(cv_scores)
    print("CV accuracies:", np.round(cv_scores, 4), " mean:", np.round(cv_mean, 4))

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    print(f"Test accuracy for {name}: {test_acc:.4f}")
    print("Confusion Matrix:\n", cm)
    results.append({
        'model': name,
        'cv_scores': cv_scores.tolist(),
        'cv_mean': float(cv_mean),
        'test_accuracy': float(test_acc),
        'confusion_matrix': cm,
        'classification_report': cr
    })
    test_accuracies[name] = test_acc

with open(RESULTS_FILE, 'w') as f:
    for r in results:
        f.write(f"Model: {r['model']}\n")
        f.write(f"CV scores: {r['cv_scores']}\n")
        f.write(f"CV mean: {r['cv_mean']:.4f}\n")
        f.write(f"Test accuracy: {r['test_accuracy']:.4f}\n")
        f.write("Confusion matrix:\n")
        np.savetxt(f, r['confusion_matrix'], fmt='%d')
        f.write("\nClassification report:\n")
        f.write(r['classification_report'])
        f.write("\n" + ("-"*40) + "\n")

print(f"\nSaved comparison results to: {RESULTS_FILE}")

plt.figure(figsize=(10, 5))
sns.barplot(x=list(test_accuracies.keys()), y=list(test_accuracies.values()))
plt.ylim(0, 1)
plt.ylabel('Test Accuracy')
plt.title('Model Test Accuracies')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(BARPLOT_FILE)
plt.close()
print(f"Saved barplot to: {BARPLOT_FILE}")

best_model_name = max(test_accuracies, key=test_accuracies.get)
best_model_idx = [r['model'] for r in results].index(best_model_name)
best_estimator = list(models.values())[list(models.keys()).index(best_model_name)]
best_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('clf', best_estimator)
])
best_pipe.fit(X_train, y_train)
joblib.dump(best_pipe, BEST_MODEL_FILE)
print(f"Saved best model ({best_model_name}) to: {BEST_MODEL_FILE}")

print("\nSummary of test accuracies:")
for name, acc in test_accuracies.items():
    print(f" - {name}: {acc:.4f}")
