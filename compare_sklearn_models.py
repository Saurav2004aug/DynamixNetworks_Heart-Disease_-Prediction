import os
import sys
import time
import math
import joblib
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from pandas.api.types import is_numeric_dtype

RANDOM_STATE = 42
CV_SPLITS = 5
N_ITER_SEARCH = 20
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_PATH = "heart1.csv"
RESULTS_PATH = os.path.join(OUTPUT_DIR, "compare_results.txt")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model_tuned.joblib")
COMBINED_ROC_PNG = os.path.join(OUTPUT_DIR, "combined_roc.png")
PR_CURVES_DIR = os.path.join(OUTPUT_DIR, "pr_curves")
ROC_CURVES_DIR = os.path.join(OUTPUT_DIR, "roc_curves")
os.makedirs(PR_CURVES_DIR, exist_ok=True)
os.makedirs(ROC_CURVES_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compare_tune_and_roc")

def drop_leaky_columns(df: pd.DataFrame, label_col: str = "num", threshold: float = 0.01) -> pd.DataFrame:
    df = df.copy()
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe")
    df['target'] = (df[label_col] > 0).astype(int)
    X = df.select_dtypes(include=[np.number]).drop(columns=[label_col, 'target'], errors='ignore')
    y = (df[label_col] > 0).astype(int)
    to_drop = []
    for c in X.columns:
        if is_numeric_dtype(X[c]):
            prop = (X[c].values == y.values).mean()
            if prop > threshold:
                to_drop.append(c)
    if 'id' in df.columns:
        to_drop.append('id')
    logger.info("Dropping columns due to potential leakage / id: %s", to_drop)
    return df.drop(columns=to_drop + [label_col], errors='ignore')

def build_pipeline(estimator):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", estimator)
    ])

def main():
    start_time = time.time()
    logger.info("Starting model comparison + tuning run")

    if not os.path.exists(DATA_PATH):
        logger.error("Data file not found: %s", DATA_PATH)
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded data shape: %s", df.shape)

    clean_df = drop_leaky_columns(df, label_col="num", threshold=0.01)
    if 'target' not in clean_df.columns:
        df['target'] = (df['num'] > 0).astype(int)
        clean_df['target'] = df['target']
    X = clean_df.select_dtypes(include=[np.number]).drop(columns=['target'], errors='ignore')
    y = clean_df['target'].astype(int)
    logger.info("Features used (%d): %s", X.shape[1], list(X.columns))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train/test split: %s / %s", X_train.shape, X_test.shape)
    logger.info("Target distribution train: %s", y_train.value_counts(normalize=True).to_dict())

    candidate_models = {
        "LogisticRegression": {
            "est": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            "params": {
                "clf__C": [0.01, 0.1, 1, 10, 100],
                "clf__penalty": ["l2"],
                "clf__solver": ["lbfgs"]
            }
        },
        "RandomForest": {
            "est": RandomForestClassifier(random_state=RANDOM_STATE),
            "params": {
                "clf__n_estimators": [50, 100, 200],
                "clf__max_depth": [None, 5, 10, 20],
                "clf__class_weight": [None, "balanced"]
            }
        },
        "GradientBoosting": {
            "est": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "params": {
                "clf__n_estimators": [50, 100, 200],
                "clf__learning_rate": [0.01, 0.05, 0.1]
            }
        },
        "SVM": {
            "est": SVC(probability=True, random_state=RANDOM_STATE),
            "params": {
                "clf__C": [0.1, 1, 10],
                "clf__kernel": ["rbf", "linear"]
            }
        },
        "KNN": {
            "est": KNeighborsClassifier(),
            "params": {
                "clf__n_neighbors": [3, 5, 7, 9]
            }
        },
        "DecisionTree": {
            "est": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "params": {
                "clf__max_depth": [None, 5, 10, 20],
                "clf__class_weight": [None, "balanced"]
            }
        }
    }

    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    summary = []
    best_overall = None
    best_overall_score = -math.inf

    for name, cfg in candidate_models.items():
        logger.info("Tuning model: %s", name)
        pipe = build_pipeline(cfg["est"])
        param_dist = cfg.get("params", {})
        if not param_dist:
            logger.warning("No param grid provided for %s; skipping search", name)
            pipe.fit(X_train, y_train)
            best_pipe = pipe
            best_score = None
            search = None
        else:
            search = RandomizedSearchCV(
                pipe,
                param_distributions=param_dist,
                n_iter=min(N_ITER_SEARCH, max(1, sum(len(v) for v in param_dist.values()))),
                scoring="roc_auc",
                cv=skf,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=0
            )
            search.fit(X_train, y_train)
            best_pipe = search.best_estimator_
            best_score = search.best_score_
            logger.info("%s best CV ROC-AUC: %.4f", name, best_score)

        best_pipe.fit(X_train, y_train)

        y_proba = best_pipe.predict_proba(X_test)[:, 1] if hasattr(best_pipe, "predict_proba") else None
        y_pred = best_pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")
        avg_precision = average_precision_score(y_test, y_proba) if y_proba is not None else float("nan")

        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_proba)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
            plt.plot([0,1],[0,1],"--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {name}")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(ROC_CURVES_DIR, f"{name}_roc.png"))
            plt.close()
            plt.figure(figsize=(6,5))
            plt.plot(pr_rec, pr_prec, label=f"{name} (AP = {avg_precision:.3f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve - {name}")
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.savefig(os.path.join(PR_CURVES_DIR, f"{name}_pr.png"))
            plt.close()

        cm = confusion_matrix(y_test, y_pred)
        clf_report = classification_report(y_test, y_pred, zero_division=0)

        summary.append({
            "model": name,
            "cv_best_score": best_score,
            "test_accuracy": acc,
            "test_roc_auc": roc_auc,
            "test_avg_precision": avg_precision,
            "confusion_matrix": cm,
            "classification_report": clf_report,
            "estimator": best_pipe
        })

        if (not math.isnan(roc_auc) and roc_auc > best_overall_score) or (math.isnan(roc_auc) and acc > best_overall_score):
            best_overall = best_pipe
            best_overall_score = roc_auc if not math.isnan(roc_auc) else acc
            best_model_name = name

        logger.info("%s: test_acc=%.4f, test_roc_auc=%.4f", name, acc, roc_auc)

    plt.figure(figsize=(8, 6))
    any_curve = False
    for s in summary:
        model_name = s["model"]
        estimator = s["estimator"]
        if hasattr(estimator, "predict_proba"):
            y_proba = estimator.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc_score(y_test, y_proba)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")
            any_curve = True
    if any_curve:
        plt.plot([0,1],[0,1],"--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Combined ROC Curves")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(COMBINED_ROC_PNG)
        plt.close()
        logger.info("Saved combined ROC to %s", COMBINED_ROC_PNG)
    else:
        logger.info("No predict_proba available for any model; skipped combined ROC.")

    if best_overall is not None:
        joblib.dump(best_overall, BEST_MODEL_PATH)
        logger.info("Saved best model (%s) to %s", best_model_name, BEST_MODEL_PATH)

    with open(RESULTS_PATH, "w") as f:
        f.write("Model comparison results\n")
        f.write(f"Data: {DATA_PATH}\n")
        f.write(f"Run time (sec): {time.time() - start_time:.2f}\n\n")
        for s in summary:
            f.write(f"Model: {s['model']}\n")
            f.write(f"CV best ROC-AUC: {s['cv_best_score']}\n")
            f.write(f"Test accuracy: {s['test_accuracy']:.4f}\n")
            f.write(f"Test ROC AUC: {s['test_roc_auc']:.4f}\n")
            f.write(f"Test Average Precision (AP): {s['test_avg_precision']:.4f}\n")
            f.write("Confusion matrix:\n")
            np.savetxt(f, s["confusion_matrix"], fmt="%d")
            f.write("\nClassification report:\n")
            f.write(s["classification_report"])
            f.write("\n" + ("-"*60) + "\n")

    logger.info("Saved results to %s", RESULTS_PATH)
    logger.info("All done. Total time: %.2f sec", time.time() - start_time)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Fatal error in compare_tune_and_roc: %s", e)
        raise
