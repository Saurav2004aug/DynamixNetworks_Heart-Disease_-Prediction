import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("All libraries imported successfully.")

try:
    data = pd.read_csv("heart1.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("ERROR: File not found. Did you name the file 'heart1.csv'?")
    print("Please check the file name and correct it.")
    data = pd.DataFrame()

if not data.empty:
    print("\n--- First 5 rows of Data ---")
    print(data.head())

    print(f"\nDataset contains {data.shape[0]} rows and {data.shape[1]} columns.")

    print("\n--- Column Information (Data Types) ---")
    data.info()

    print("\n--- Statistical Summary of Data ---")
    print(data.describe())

    print("\n--- Checking for Missing Values ---")
    print(data.isnull().sum())

    print("\n--- Checking for Duplicate Rows ---")
    duplicate_count = data.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_count}")

    if duplicate_count > 0:
        data = data.drop_duplicates()
        print(f"Duplicate rows removed. New shape: {data.shape}")

    print("\n--- Starting EDA and Visualizations... ---")

    sns.set_style("darkgrid")

    # Assuming 'num' contains the diagnosis (0 = no disease, 1-4 = disease)
    data['target'] = (data['num'] > 0).astype(int)

    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

    plt.figure(figsize=(6, 5))
    sns.countplot(x='target', data=data)
    plt.title("Distribution of Heart Disease (0 = No, >0 = Yes)")
    plt.xlabel("Target (Heart Disease)")
    plt.ylabel("Count (Number of People)")
    plt.savefig('output/code_target_distribution.png')

    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='age', hue='target', kde=True, palette='magma')
    plt.title("Distribution of Heart Disease by Age")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.legend(title='Heart Disease', labels=['No', 'Yes'])
    plt.savefig('output/code_age_distribution.png')

    plt.figure(figsize=(14, 10))
    numeric_data = data[numeric_columns]
    correlation_matrix = numeric_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={'size': 8})
    plt.title("Correlation Heatmap of Numeric Features")
    plt.savefig('output/code_correlation_heatmap.png')

    print("--- EDA and Visualizations Completed ---")

    columns_to_drop = ['target', 'num', 'id', 'dataset']
    X = data.select_dtypes(include=[np.number]).drop(columns=[col for col in columns_to_drop if col in data.columns], errors='ignore')
    y = data['target']

    print(f"\nShape of Features (X) (before cleaning): {X.shape}")

    missing_before = X.isnull().sum().sum()
    print(f"Missing values in features: {missing_before}")

    if missing_before > 0:
        X = X.fillna(X.mean())
        print("Missing values have been filled with column means.")

        missing_after = X.isnull().sum().sum()
        print(f"Missing values after filling: {missing_after}")

    print(f"Final shape of Features (X): {X.shape}")
    print(f"Shape of Target (y): {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nTraining data (X_train) contains {X_train.shape[0]} samples.")
    print(f"Testing data (X_test) contains {X_test.shape[0]} samples.")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    print("\nData scaled successfully.")

    print("Model training started...")

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    print("Model trained successfully!")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n--- Model Performance ---")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('output/code_confusion_matrix.png')

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    with open('output/code_results.txt', 'w') as f:
        f.write(f"Model Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        np.savetxt(f, cm, fmt='%d')
        f.write(f"\n\nClassification Report:\n{classification_report(y_test, y_pred)}")