import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def load_data(csv_path="data/processed/features.csv"):
    df = pd.read_csv(csv_path)
    X = df.drop("label", axis=1).values
    y = df["label"].values
    return X, y

def train_models(csv_path="data/processed/features.csv", models_dir="models"):
    X, y = load_data(csv_path)

    le = LabelEncoder()
    y_enc = le.fit_transform(y) # type: ignore

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    kernels = ["linear", "poly", "rbf", "sigmoid"]
    os.makedirs(models_dir, exist_ok=True)

    metrics = {}
    for kernel in kernels:
        print(f"\n🔹 Training SVM with {kernel} kernel...")
        model = SVC(kernel=kernel, C=10, gamma="scale", degree=3) # type: ignore
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        metrics[kernel] = {"accuracy": acc, "f1_weighted": f1}

        print(f"  Accuracy: {acc:.3f} | F1: {f1:.3f}")
        print("  Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("  Classification report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        # Save individual model (bundle with scaler + encoder)
        bundle = {"model": model, "scaler": scaler, "label_encoder": le}
        joblib.dump(bundle, os.path.join(models_dir, f"svm_{kernel}.pkl"))
        print(f"✅ Saved model: models/svm_{kernel}.pkl")

    metrics_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\n📊 Saved metrics summary to {metrics_path}")

    best_kernel = max(
        metrics,
        key=lambda k: (metrics[k]["accuracy"], metrics[k]["f1_weighted"])
    )
    print("\n🎯 Best kernel (by accuracy → f1 fallback):", best_kernel)
    return metrics

if __name__ == "__main__":
    train_models()