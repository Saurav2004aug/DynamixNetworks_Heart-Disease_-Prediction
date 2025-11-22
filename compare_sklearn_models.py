import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

print("All libraries and models imported successfully.")

try:
    data = pd.read_csv("heart1.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("ERROR: 'heart1.csv' file not found. Please check.")
    data = pd.DataFrame()

if not data.empty:
    data = data.drop_duplicates()
    data['target'] = (data['num'] > 0).astype(int)
    X = data.select_dtypes(include=[np.number]).drop('target', axis=1)
    y = data['target']
    X = X.fillna(X.mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Data processing (splitting and scaling) completed.")

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "K-Nearest Neighbors (K-NN)": KNeighborsClassifier(n_neighbors=5)
    }

    model_accuracies = []

    print("\n--- Model Training and Evaluation Started ---")

    for model_name, model_instance in models.items():
        print(f"Training {model_name}...")
        model_instance.fit(X_train, y_train)
        y_pred = model_instance.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_accuracies.append(accuracy)
        print(f"{model_name} Accuracy: {accuracy * 100:.2f}%\n")

    print("\n--- All Models Final Accuracy ---")

    plt.figure(figsize=(10, 7))
    model_names = list(models.keys())
    sns.barplot(x=model_accuracies, y=model_names)
    plt.title("Machine Learning Models Accuracy Comparison")
    plt.xlabel("Accuracy (0.0 to 1.0)")
    plt.ylabel("Model")
    for index, value in enumerate(model_accuracies):
        plt.text(value + 0.01, index, f"{value*100:.2f}%")
    plt.savefig('output/compare_barplot.png')

    with open('output/compare_results.txt', 'w') as f:
        f.write("Model Accuracies:\n")
        for model, acc in zip(model_names, model_accuracies):
            f.write(f"{model}: {acc*100:.2f}%\n")
