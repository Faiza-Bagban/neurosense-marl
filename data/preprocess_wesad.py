"""
data/preprocess_wesad.py
Window WESAD chest signals into fixed-length segments and extract
physio features (HRV from ECG, EDA stats, EMG, Temp, Resp) per window.
Output: processed feature matrix + binary stress label, saved as CSV.

WESAD chest sampling rate: 700 Hz.
Label codes: 0=undefined/transient, 1=baseline, 2=stress, 3=amusement,
4=meditation, 5/6/7=unused conditions.
We keep only baseline(1), stress(2), amusement(3) and binarize:
stress=1 (label==2), non-stress=0 (label in {1,3}).
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from data.loaders import load_wesad_subject, WESAD_SUBJECTS

FS = 700  # chest sampling rate (Hz)
WINDOW_SEC = 60
STEP_SEC = 30  # 50% overlap
WINDOW_SIZE = FS * WINDOW_SEC
STEP_SIZE = FS * STEP_SEC

KEEP_LABELS = {1, 2, 3}


def extract_hrv_features(ecg_window: np.ndarray, fs: int = FS) -> dict:
    """Rough HRV features from raw ECG using simple peak detection."""
    ecg_window = ecg_window.flatten()
    peaks, _ = find_peaks(ecg_window, distance=fs * 0.4, height=np.mean(ecg_window) + 0.5 * np.std(ecg_window))
    if len(peaks) < 2:
        return {"hrv_mean_rr": np.nan, "hrv_std_rr": np.nan, "hrv_rmssd": np.nan}
    rr = np.diff(peaks) / fs * 1000  # ms
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 1 else np.nan
    return {"hrv_mean_rr": np.mean(rr), "hrv_std_rr": np.std(rr), "hrv_rmssd": rmssd}


def extract_window_features(chest: dict, start: int, end: int) -> dict:
    eda = chest["EDA"][start:end].flatten()
    emg = chest["EMG"][start:end].flatten()
    temp = chest["Temp"][start:end].flatten()
    resp = chest["Resp"][start:end].flatten()
    ecg = chest["ECG"][start:end].flatten()

    feats = {
        "eda_mean": np.mean(eda), "eda_std": np.std(eda), "eda_slope": (eda[-1] - eda[0]) / len(eda),
        "emg_mean_abs": np.mean(np.abs(emg)), "emg_std": np.std(emg),
        "temp_mean": np.mean(temp), "temp_std": np.std(temp),
        "resp_mean": np.mean(resp), "resp_std": np.std(resp),
    }
    feats.update(extract_hrv_features(ecg))
    return feats


def process_subject(subject_id: str) -> pd.DataFrame:
    data = load_wesad_subject(subject_id)
    chest = data["signal"]["chest"]
    labels = data["label"].flatten()
    n_samples = len(labels)

    rows = []
    for start in range(0, n_samples - WINDOW_SIZE, STEP_SIZE):
        end = start + WINDOW_SIZE
        window_labels = labels[start:end]
        vals, counts = np.unique(window_labels, return_counts=True)
        majority_label = vals[np.argmax(counts)]
        if majority_label not in KEEP_LABELS:
            continue
        feats = extract_window_features(chest, start, end)
        feats["label_raw"] = int(majority_label)
        feats["label"] = 1 if majority_label == 2 else 0  # stress vs non-stress
        feats["subject"] = subject_id
        rows.append(feats)

    return pd.DataFrame(rows)


def process_all_subjects(subjects=None) -> pd.DataFrame:
    subjects = subjects or WESAD_SUBJECTS
    all_dfs = []
    for s in subjects:
        print(f"Processing {s}...")
        df = process_subject(s)
        all_dfs.append(df)
    full = pd.concat(all_dfs, ignore_index=True)
    return full


if __name__ == "__main__":
    df = process_all_subjects()
    print("Total windows:", len(df))
    print("Label distribution:\n", df["label"].value_counts())
    df.to_csv("data/processed/wesad_features.csv", index=False)
    print("Saved to data/processed/wesad_features.csv")