"""Prepare train/test splits from generated and external data.

Loads synthetic call transcripts, SMS spam data, and URL phishing data.
Cleans text, creates stratified train/test splits, and saves to CSV.

Usage
-----
    python training/prepare_datasets.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Project root on sys.path so we can import sentinel_edge modules
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ======================================================================
# Text cleaning
# ======================================================================

def clean_text(text: str) -> str:
    """Normalise a single text string for training.

    - Lowercase
    - Strip leading/trailing whitespace
    - Collapse multiple whitespace characters
    - Remove non-printable characters (keep basic punctuation)
    """
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    # Remove non-printable / control chars (keep newline, tab, printable)
    text = re.sub(r"[^\x20-\x7E]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ======================================================================
# Load + clean helpers
# ======================================================================

def load_and_clean(data_dir: str) -> pd.DataFrame:
    """Load all CSVs in *data_dir*, clean text, return unified DataFrame.

    Expects CSVs with at least columns: ``text``, ``label``.
    An optional ``category`` column is preserved if present.
    """
    dfs: list[pd.DataFrame] = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(data_dir, fname)
        df = pd.read_csv(path, dtype={"text": str, "label": int})
        dfs.append(df)
        print(f"  Loaded {path}: {len(df):,} rows")

    if not dfs:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    combined = pd.concat(dfs, ignore_index=True)

    # Clean
    combined["text"] = combined["text"].apply(clean_text)

    # Drop empty/null
    combined = combined[combined["text"].str.len() > 0].copy()
    combined = combined.dropna(subset=["text", "label"])

    # Ensure label is int
    combined["label"] = combined["label"].astype(int)

    print(f"  Combined: {len(combined):,} rows after cleaning")
    return combined


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split preserving label distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``label`` column.
    test_size : float
        Fraction reserved for the test set.
    seed : int
        Random state.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"  Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows")
    print(
        f"  Train label distribution: "
        f"{dict(train_df['label'].value_counts().sort_index())}"
    )
    print(
        f"  Test  label distribution: "
        f"{dict(test_df['label'].value_counts().sort_index())}"
    )
    return train_df, test_df


def save_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
    prefix: str,
) -> None:
    """Save train/test DataFrames as CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, f"{prefix}_train.csv")
    test_path = os.path.join(output_dir, f"{prefix}_test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"  Saved {train_path}")
    print(f"  Saved {test_path}")


# ======================================================================
# Call transcript data
# ======================================================================

def prepare_call_data(
    raw_dir: str,
    output_dir: str,
    test_size: float = 0.2,
    seed: int = 42,
) -> None:
    """Prepare call fraud dataset from synthetic transcripts.

    Looks for CSVs in ``raw_dir`` (e.g. data/raw/synthetic_transcripts/).
    """
    print("\n[Call Fraud Data]")
    df = load_and_clean(raw_dir)
    train_df, test_df = create_train_test_split(df, test_size=test_size, seed=seed)
    save_split(train_df, test_df, output_dir, "call_fraud")


# ======================================================================
# SMS spam data  (Kaggle SMS Spam Collection format)
# ======================================================================

def prepare_sms_data(
    raw_path: str,
    output_dir: str,
    test_size: float = 0.2,
    seed: int = 42,
) -> None:
    """Load SMS spam data, clean, split, save.

    Supports two common formats:
    1. Tab-separated file with columns: label_str, text
       (Kaggle SMS Spam Collection: "ham"/"spam" label)
    2. CSV with columns: text, label (numeric)
    """
    print("\n[SMS Spam Data]")

    if not os.path.exists(raw_path):
        # If the raw file doesn't exist, try the directory approach
        if os.path.isdir(raw_path):
            df = load_and_clean(raw_path)
        else:
            print(f"  WARNING: {raw_path} not found, skipping SMS data preparation.")
            print("  To use SMS data, place the Kaggle SMS Spam Collection file at:")
            print(f"    {raw_path}")
            print("  Expected format: tab-separated with columns 'label' and 'text'")
            print("  Or CSV with 'text' and 'label' columns.")
            return
    else:
        # Try reading as the Kaggle SMS Spam Collection format first
        try:
            df = pd.read_csv(
                raw_path,
                sep="\t",
                header=None,
                names=["label_str", "text"],
                encoding="latin-1",
            )
            # Convert ham/spam to 0/1
            if "label_str" in df.columns:
                label_map = {"ham": 0, "spam": 1}
                df["label"] = df["label_str"].map(label_map)
                df = df.dropna(subset=["label"])
                df["label"] = df["label"].astype(int)
                df["category"] = df["label_str"]
                df = df[["text", "label", "category"]]
                print(f"  Loaded Kaggle SMS format from {raw_path}: {len(df):,} rows")
        except Exception:
            # Fall back to standard CSV
            df = pd.read_csv(raw_path, dtype={"text": str, "label": int})
            print(f"  Loaded CSV from {raw_path}: {len(df):,} rows")

    # Clean
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0].copy()
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    print(f"  After cleaning: {len(df):,} rows")
    print(f"  Label distribution: {dict(df['label'].value_counts().sort_index())}")

    train_df, test_df = create_train_test_split(df, test_size=test_size, seed=seed)
    save_split(train_df, test_df, output_dir, "sms_fraud")


# ======================================================================
# URL phishing data
# ======================================================================

def prepare_url_data(
    raw_path: str,
    output_dir: str,
    test_size: float = 0.2,
    seed: int = 42,
) -> None:
    """Load URL phishing data, clean, split, save.

    Supports:
    1. CSV with columns: url, label (0=legit, 1=phishing)
    2. CSV with columns: url, type  ("benign"/"phishing"/"defacement"/"malware")
    3. Directory of CSVs
    """
    print("\n[URL Phishing Data]")

    if not os.path.exists(raw_path):
        if os.path.isdir(raw_path):
            # Try loading all CSVs from directory
            dfs = []
            for fname in sorted(os.listdir(raw_path)):
                if fname.endswith(".csv"):
                    fpath = os.path.join(raw_path, fname)
                    dfs.append(pd.read_csv(fpath))
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
            else:
                print(f"  WARNING: No CSV files in {raw_path}, skipping URL data.")
                return
        else:
            print(f"  WARNING: {raw_path} not found, skipping URL data preparation.")
            print("  To use URL data, place a phishing URL dataset at:")
            print(f"    {raw_path}")
            print("  Expected format: CSV with 'url' and 'label' columns.")
            return
    else:
        df = pd.read_csv(raw_path, low_memory=False)
        print(f"  Loaded {raw_path}: {len(df):,} rows")

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Handle different label formats
    if "type" in df.columns and "label" not in df.columns:
        # Multi-class: benign=0, everything else=1
        df["label"] = (df["type"].str.lower() != "benign").astype(int)
        df["category"] = df["type"].str.lower()
    elif "label" in df.columns:
        # Already numeric or string
        if df["label"].dtype == object:
            label_map = {
                "good": 0, "benign": 0, "legitimate": 0, "safe": 0,
                "bad": 1, "phishing": 1, "malicious": 1, "malware": 1,
                "defacement": 1, "spam": 1,
            }
            df["label"] = df["label"].str.lower().map(label_map)
        df["label"] = df["label"].astype(int)

    # Need a 'url' or 'text' column
    if "url" in df.columns:
        df = df.rename(columns={"url": "text"})
    elif "text" not in df.columns:
        # Try first string column
        for col in df.columns:
            if df[col].dtype == object and col not in ("label", "category", "type"):
                df = df.rename(columns={col: "text"})
                break

    if "text" not in df.columns or "label" not in df.columns:
        print("  WARNING: Could not identify text and label columns, skipping.")
        return

    # Clean (light cleaning for URLs - preserve structure)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    # Keep only text, label, category
    cols = ["text", "label"]
    if "category" in df.columns:
        cols.append("category")
    df = df[cols]

    print(f"  After cleaning: {len(df):,} rows")
    print(f"  Label distribution: {dict(df['label'].value_counts().sort_index())}")

    train_df, test_df = create_train_test_split(df, test_size=test_size, seed=seed)
    save_split(train_df, test_df, output_dir, "url_fraud")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare train/test splits from generated and external data."
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root data directory (default: data)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for test set (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    raw_dir = os.path.join(args.data_root, "raw")
    processed_dir = os.path.join(args.data_root, "processed")

    print("=" * 60)
    print("Preparing datasets for SentinelEdge training")
    print("=" * 60)

    # 1. Call fraud (synthetic transcripts)
    call_raw = os.path.join(raw_dir, "synthetic_transcripts")
    prepare_call_data(call_raw, processed_dir, args.test_size, args.seed)

    # 2. SMS spam
    sms_raw = os.path.join(raw_dir, "sms_spam", "SMSSpamCollection")
    if not os.path.exists(sms_raw):
        # Try alternative filename
        sms_raw_alt = os.path.join(raw_dir, "sms_spam", "sms_spam.csv")
        if os.path.exists(sms_raw_alt):
            sms_raw = sms_raw_alt
        else:
            # Try the directory itself
            sms_raw = os.path.join(raw_dir, "sms_spam")
    prepare_sms_data(sms_raw, processed_dir, args.test_size, args.seed)

    # 3. URL phishing
    url_raw = os.path.join(raw_dir, "phishing_urls", "urls.csv")
    if not os.path.exists(url_raw):
        url_raw_alt = os.path.join(raw_dir, "phishing_urls", "phishing_urls.csv")
        if os.path.exists(url_raw_alt):
            url_raw = url_raw_alt
        else:
            url_raw = os.path.join(raw_dir, "phishing_urls")
    prepare_url_data(url_raw, processed_dir, args.test_size, args.seed)

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)

    # List output files
    if os.path.isdir(processed_dir):
        print("\nOutput files:")
        for fname in sorted(os.listdir(processed_dir)):
            fpath = os.path.join(processed_dir, fname)
            if os.path.isfile(fpath):
                size_kb = os.path.getsize(fpath) / 1024
                print(f"  {fpath}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
