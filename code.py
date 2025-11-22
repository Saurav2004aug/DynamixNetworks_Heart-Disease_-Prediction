import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Sabhi libraries successfully import ho gayi hain.")

try:
    data = pd.read_csv("heart1.csv")
    print("Dataset successfully load ho gaya hai.")
except FileNotFoundError:
    print("ERROR: File nahi mili. Kya aapne file ka naam 'heart1.csv' rakha hai?")
    print("Please file ka naam check karein aur use sahi karein.")
    data = pd.DataFrame()

if not data.empty:
    print("\n--- Data ki pehli 5 rows ---")
    print(data.head())

    print(f"\nDataset mein {data.shape[0]} rows aur {data.shape[1]} columns hain.")

    print("\n--- Column Information (Data Types) ---")
    data.info()

    print("\n--- Data ka Statistical Summary ---")
    print(data.describe())

    print("\n--- Missing Values ki Jaanch ---")
    print(data.isnull().sum())

    print("\n--- Duplicate Rows ki Jaanch ---")
    duplicate_count = data.duplicated().sum()
    print(f"Duplicate rows ki sankhya: {duplicate_count}")

    if duplicate_count > 0:
        data = data.drop_duplicates()
        print(f"Duplicate rows hata di gayi hain. Nayi shape: {data.shape}")

    print("\n--- EDA aur Visualizations Shuru... ---")

    sns.set_style("darkgrid")

    data['target'] = (data['num'] > 0).astype(int)

    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

    plt.figure(figsize=(6, 5))
    sns.countplot(x='target', data=data)
    plt.title("Heart Disease ka Distribution (0 = Nahi, >0 = Haan)")
    plt.xlabel("Target (Heart Disease)")
    plt.ylabel("Count (Logon ki sankhya)")
    plt.savefig('output/code_target_distribution.png')

    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='age', hue='target', kde=True, palette='magma')
    plt.title("Age ke Aadhar Par Heart Disease ka Distribution")
    plt.xlabel("Age (Umar)")
    plt.ylabel("Count")
    plt.legend(title='Heart Disease', labels=['Nahi', 'Haan'])
    plt.savefig('output/code_age_distribution.png')

    plt.figure(figsize=(14, 10))
    numeric_data = data[numeric_columns]
    correlation_matrix = numeric_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={'size': 8})
    plt.title("Numeric Features ka Correlation Heatmap")
    plt.savefig('output/code_correlation_heatmap.png')

    print("--- EDA aur Visualizations Poore Hue ---")

    columns_to_drop = ['target', 'num', 'id', 'dataset']
    X = data.select_dtypes(include=[np.number]).drop(columns=[col for col in columns_to_drop if col in data.columns], errors='ignore')
    y = data['target']

    print(f"\nFeatures (X) ka shape (before cleaning): {X.shape}")

    missing_before = X.isnull().sum().sum()
    print(f"Missing values in features: {missing_before}")

    if missing_before > 0:
        X = X.fillna(X.mean())
        print("Missing values ko column means se fill kar diya gaya hai.")

        missing_after = X.isnull().sum().sum()
        print(f"Missing values after filling: {missing_after}")

    print(f"Features (X) ka final shape: {X.shape}")
    print(f"Target (y) ka shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nTraining data (X_train) mein {X_train.shape[0]} samples hain.")
    print(f"Testing data (X_test) mein {X_test.shape[0]} samples hain.")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    print("\nData successfully scale ho gaya hai.")

    print("Model training shuru ho gayi hai...")

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    print("Model successfully train ho gaya hai!")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n--- Model Performance ---")
    print(f"Model ki Accuracy: {accuracy * 100:.2f}%")

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted (Model ne jo kaha)')
    plt.ylabel('Actual (Asli mein jo tha)')
    plt.savefig('output/code_confusion_matrix.png')

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    with open('output/code_results.txt', 'w') as f:
        f.write(f"Model Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        np.savetxt(f, cm, fmt='%d')
        f.write(f"\n\nClassification Report:\n{classification_report(y_test, y_pred)}")

