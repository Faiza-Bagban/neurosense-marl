"""
data/loaders.py
Loaders for WESAD (physio) and Reddit depression text dataset.
"""
import pickle
import pandas as pd
from pathlib import Path

# ---- paths ----
RAW_DIR = Path(__file__).resolve().parent / "raw"
WESAD_DIR = RAW_DIR / "WESAD"
REDDIT_CSV = RAW_DIR / "reddit" / "depression_dataset_reddit_cleaned.csv"

WESAD_SUBJECTS = ["S2","S3","S4","S5","S6","S7","S8","S9","S10","S11","S13","S14","S15","S16","S17"]


def load_wesad_subject(subject_id: str) -> dict:
    """
    Load one WESAD subject's pickle file.
    Returns dict with keys: 'signal' (chest/wrist), 'label', 'subject'.
    WESAD label codes: 0=not defined,1=baseline,2=stress,3=amusement,4=meditation
    """
    pkl_path = WESAD_DIR / subject_id / f"{subject_id}.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data  # data['signal']['chest'/'wrist'], data['label'], data['subject']


def load_all_wesad(subjects=None) -> dict:
    """Load all (or subset) of WESAD subjects into a dict keyed by subject id."""
    subjects = subjects or WESAD_SUBJECTS
    return {s: load_wesad_subject(s) for s in subjects}


def load_reddit_text() -> pd.DataFrame:
    """Load Reddit depression text dataset."""
    df = pd.read_csv(REDDIT_CSV)
    df = df.rename(columns={"clean_text": "text", "is_depression": "label"})
    df = df.dropna(subset=["text", "label"])
    return df


if __name__ == "__main__":
    # quick sanity check
    df = load_reddit_text()
    print("Reddit rows:", len(df), "columns:", df.columns.tolist())

    s2 = load_wesad_subject("S2")
    print("WESAD S2 keys:", s2.keys())
    print("Chest signal keys:", s2["signal"]["chest"].keys())
    print("Label array shape:", s2["label"].shape)