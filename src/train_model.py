import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def load_data(csv_path="data/processed/features.csv"):
    """Load feature CSV and split into X, y."""
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

    os.makedirs(models_dir, exist_ok=True)

    param_grids = {
        "linear": {"C": [0.1, 1, 10], "kernel": ["linear"]},
        "poly": {"C": [0.1, 1, 10], "degree": [2, 3, 4], "kernel": ["poly"], "gamma": ["scale", 0.1, 0.01]},
        "rbf": {"C": [0.1, 1, 10], "gamma": ["scale", 0.1, 0.01], "kernel": ["rbf"]},
        "sigmoid": {"C": [0.1, 1, 10], "gamma": ["scale", 0.1, 0.01], "kernel": ["sigmoid"]}
    }

    metrics = {}

    for kernel, grid in param_grids.items():
        print(f"\n🔹 Tuning and training SVM with {kernel} kernel (probability=True)...")
        svc = SVC(probability=True)
        grid_search = GridSearchCV(
            estimator=svc,
            param_grid=grid,
            scoring="accuracy",
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train_s, y_train)

        print("   ➤ Best parameters:", grid_search.best_params_)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        metrics[kernel] = {
            "accuracy": acc,
            "f1_weighted": f1,
            "best_params": grid_search.best_params_
        }

        print(f"  Accuracy: {acc:.3f} | F1: {f1:.3f}")
        print("  Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("  Classification report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        bundle = {"model": best_model, "scaler": scaler, "label_encoder": le}
        out_path = os.path.join(models_dir, f"svm_{kernel}.pkl")
        joblib.dump(bundle, out_path)
        print(f" Saved model: {out_path}")

    metrics_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\n Saved metrics summary to {metrics_path}")

    best_kernel = max(metrics, key=lambda k: (metrics[k]["accuracy"], metrics[k]["f1_weighted"]))
    print("\n Best kernel (by accuracy → f1 fallback):", best_kernel)
    print(" Best params:", metrics[best_kernel]["best_params"])

    return metrics

if __name__ == "__main__":
    train_models()