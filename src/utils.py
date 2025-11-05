import os
import json
import joblib
import pandas as pd
import streamlit as st

def load_model(kernel: str, models_dir: str = "models"):
    path = os.path.join(models_dir, f"svm_{kernel}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model for kernel '{kernel}' not found at {path}")
    bundle = joblib.load(path)
    return bundle["model"], bundle["scaler"], bundle["label_encoder"]

def load_metrics(models_dir: str = "models") -> pd.DataFrame:
    path = os.path.join(models_dir, "metrics.json")
    if not os.path.exists(path):
        raise FileNotFoundError("metrics.json not found — train models first.")
    with open(path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data).T.reset_index().rename(columns={"index": "kernel"})
    return df

def display_metrics_table(df: pd.DataFrame):
    best_idx = df["accuracy"].idxmax()
    st.subheader("Kernel Performance Summary")
    st.dataframe(df.style.highlight_max(axis=0, color="#d0f0c0"))
    best_kernel = df.loc[best_idx, "kernel"]
    best_acc = df.loc[best_idx, "accuracy"]
    st.success(f"🏆 **Best Kernel:** {best_kernel} ({best_acc:.3f} accuracy)")
