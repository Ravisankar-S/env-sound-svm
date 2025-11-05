import json
import os
import numpy as np
import librosa
from src.feature_extraction import extract_features
from src.utils import load_model

def get_best_kernel(models_dir="models"):
    metrics_path = os.path.join(models_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        return "rbf"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    best = max(metrics, key=lambda k: (metrics[k]["accuracy"], metrics[k]["f1_weighted"]))
    return best

def predict_sound(file_path: str, kernel: str = None, models_dir: str = "models"): # type: ignore
    if kernel is None:
        kernel = get_best_kernel(models_dir)
        print(f" Auto-selected best kernel: {kernel}")
    
    model, scaler, le = load_model(kernel, models_dir)

    feats = extract_features(file_path).reshape(1, -1)
    feats_s = scaler.transform(feats)

    pred_idx = model.predict(feats_s)[0]
    pred_label = le.inverse_transform([pred_idx])[0]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(feats_s)
        confidence = np.max(scores) / np.sum(np.abs(scores))
    elif hasattr(model, "predict_proba"):
        probs = model.predict_proba(feats_s)
        confidence = np.max(probs)
    else:
        confidence = None

    return pred_label, confidence

# CLI (optional)
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/predict_sound.py <path_to_audio> [kernel]")
    else:
        audio_path = sys.argv[1]
        kernel = sys.argv[2] if len(sys.argv) > 2 else "rbf"
        label, conf = predict_sound(audio_path, kernel)
        print(f"Predicted: {label} (confidence: {conf:.3f})")
