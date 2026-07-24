"""
data/preprocess_reddit.py
Clean Reddit depression text and extract TF-IDF features for the
behavior agent. Uses sklearn only (no extra installs needed).
Output: sparse TF-IDF matrix saved as .npz + labels as .csv.
"""
import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from data.loaders import load_reddit_text

MAX_FEATURES = 300  # keep small — RL agent input dim manageable


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # urls
    text = re.sub(r"[^a-z\s]", " ", text)                 # keep letters only
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_reddit():
    df = load_reddit_text()
    df["clean"] = df["text"].apply(clean_text)
    df = df[df["clean"].str.len() > 0]

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["clean"])

    sparse.save_npz("data/processed/reddit_tfidf.npz", tfidf_matrix)
    df[["label"]].to_csv("data/processed/reddit_labels.csv", index=False)

    # save vocabulary too, useful for LLM aggregator rationale later
    vocab = vectorizer.get_feature_names_out()
    pd.DataFrame({"term": vocab}).to_csv("data/processed/reddit_vocab.csv", index=False)

    print("TF-IDF matrix shape:", tfidf_matrix.shape)
    print("Label distribution:\n", df["label"].value_counts())
    print("Saved: reddit_tfidf.npz, reddit_labels.csv, reddit_vocab.csv")


if __name__ == "__main__":
    preprocess_reddit()