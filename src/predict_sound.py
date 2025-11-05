import json
import os
import numpy as np
from src.feature_extraction import extract_features
from src.utils import load_model

def get_best_kernel(models_dir="models"):
    """Reads metrics.json and returns the best kernel (accuracy → F1 fallback)."""
    metrics_path = os.path.join(models_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        return "rbf"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    best = max(metrics, key=lambda k: (metrics[k]["accuracy"], metrics[k]["f1_weighted"]))
    return best

def predict_sound(file_path: str, kernel: str = None, models_dir: str = "models"): # type: ignore
    """
    Predicts the sound class for a given audio file using a trained SVM model.
    If kernel=None, automatically loads the best kernel from metrics.json.
    Uses calibrated probability estimates for confidence.
    """
    if kernel is None:
        kernel = get_best_kernel(models_dir)
        print(f"📈 Auto-selected best kernel: {kernel}")

    model, scaler, le = load_model(kernel, models_dir)

    feats = extract_features(file_path).reshape(1, -1)
    feats_s = scaler.transform(feats)

    pred_idx = model.predict(feats_s)[0]
    pred_label = le.inverse_transform([pred_idx])[0]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feats_s)[0]
        confidence = float(np.max(probs))
    else:
        scores = model.decision_function(feats_s)
        confidence = float(np.max(scores) / (np.sum(np.abs(scores)) + 1e-8))

    return pred_label, confidence

# CLI for quick testing
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/predict_sound.py <path_to_audio> [kernel]")
    else:
        audio_path = sys.argv[1]
        kernel = sys.argv[2] if len(sys.argv) > 2 else None
        label, conf = predict_sound(audio_path, kernel) # type: ignore
        print(f"Predicted: {label} | Confidence: {conf:.3f}")
