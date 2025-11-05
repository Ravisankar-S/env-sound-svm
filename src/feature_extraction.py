import os
import numpy as np
import pandas as pd
import librosa

def extract_features(file_path, sr=None, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=sr)
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    # Spectral centroid
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
    # Zero crossing rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    # Concatenate
    features = np.hstack([mfccs_mean, chroma_mean, spec_cent, zcr])
    return features

def process_dataset(root_dir="data/raw", out_csv="data/processed/features.csv"):
    rows = []
    labels = []
    file_count = 0

    for label in sorted(os.listdir(root_dir)):
        label_dir = os.path.join(root_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in sorted(os.listdir(label_dir)):
            fpath = os.path.join(label_dir, fname)
            if not fpath.lower().endswith((".wav", ".ogg", ".mp3")):
                continue
            try:
                feats = extract_features(fpath)
                rows.append(feats)
                labels.append(label)
                file_count += 1
            except Exception as e:
                print(f"Error processing {fpath}: {e}")

    if file_count == 0:
        raise RuntimeError(f"No audio files found in {root_dir}. Put labeled folders with audio files there.")

    # df
    n_mfcc = 13
    cols = [f"mfcc_{i}" for i in range(n_mfcc)] + [f"chroma_{i}" for i in range(12)] + ["spec_centroid", "zcr"]
    df = pd.DataFrame(rows, columns=cols)
    df["label"] = labels
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved features for {file_count} files to {out_csv}")

if __name__ == "__main__":
    process_dataset()
