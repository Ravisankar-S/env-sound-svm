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

def display_metrics_table(df):
    df_display = df.copy()
    
    if "best_params" in df_display.columns:
        df_display["best_params"] = df_display["best_params"].apply(
            lambda x: ", ".join([f"{k}={v}" for k, v in x.items()]) if isinstance(x, dict) else str(x)
        )
    
    df_display.index = pd.Index(range(1, len(df_display) + 1), name="S.No")

    numeric_cols = df_display.select_dtypes(include=["float", "int"]).columns.tolist()
    st.subheader("Summary")

    if len(numeric_cols) > 0:
        st.dataframe(
            df_display.style.highlight_max(subset=numeric_cols, axis=0, color="#d0f0c0")
        )
    else:
        st.dataframe(df_display)

    best_idx = df_display["accuracy"].idxmax()
    best_kernel = df_display.loc[best_idx, "kernel"]
    best_acc = df_display.loc[best_idx, "accuracy"]
    best_f1 = df_display.loc[best_idx, "f1_weighted"]
    st.success(f"🏆 **Best Kernel:** {best_kernel} ({best_acc:.3f} accuracy, {best_f1:.3f} F1 score)")
