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

def adaptive_kernel_selection(file_path: str, models_dir: str = "models", confidence_threshold: float = 0.1):
    global_best_kernel = get_best_kernel(models_dir)
    
    all_results = []
    max_confidence = 0.0
    max_confidence_kernel = None
    max_confidence_label = None
    
    with open(os.path.join(models_dir, "metrics.json"), "r") as f:
        metrics = json.load(f)
    
    available_kernels = list(metrics.keys())
    
    for kernel in available_kernels:
        try:
            label, conf = predict_sound(file_path, kernel=kernel, models_dir=models_dir)
            all_results.append({
                "kernel": kernel,
                "label": label,
                "confidence": conf
            })
            
            if conf > max_confidence:
                max_confidence = conf
                max_confidence_kernel = kernel
                max_confidence_label = label
                
        except Exception as e:
            all_results.append({
                "kernel": kernel,
                "label": f"Error: {e}",
                "confidence": 0.0
            })
    
    global_best_result = next((r for r in all_results if r["kernel"] == global_best_kernel), None)
    global_best_confidence = global_best_result["confidence"] if global_best_result else 0.0
    global_best_label = global_best_result["label"] if global_best_result else "Unknown"
    
    confidence_diff = max_confidence - global_best_confidence
    
    if max_confidence_kernel and confidence_diff >= confidence_threshold and max_confidence_kernel != global_best_kernel:
        chosen_kernel = max_confidence_kernel
        final_label = max_confidence_label
        final_confidence = max_confidence
        decision_reason = f"Switched to {max_confidence_kernel.upper()} (confidence margin: {confidence_diff:.3f} ≥ {confidence_threshold})"
        switched = True
    else:
        chosen_kernel = global_best_kernel
        final_label = global_best_label
        final_confidence = global_best_confidence
        if confidence_diff < confidence_threshold:
            decision_reason = f"Retained {global_best_kernel.upper()} (confidence margin: {confidence_diff:.3f} < {confidence_threshold})"
        else:
            decision_reason = f"Retained {global_best_kernel.upper()} (already the highest confidence)"
        switched = False
    
    decision_info = {
        "global_best_kernel": global_best_kernel,
        "global_best_confidence": global_best_confidence,
        "max_confidence_kernel": max_confidence_kernel,
        "max_confidence": max_confidence,
        "confidence_diff": confidence_diff,
        "threshold": confidence_threshold,
        "reason": decision_reason,
        "switched": switched
    }
    
    return chosen_kernel, final_label, final_confidence, all_results, decision_info

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
