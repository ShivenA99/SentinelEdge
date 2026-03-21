"""Fit TF-IDF vectorizer on training data and save.

Loads the training CSV, fits a TfidfFeatureExtractor, and serialises it
for use by the classification training scripts and the inference pipeline.

Usage
-----
    python training/fit_tfidf.py [--train-csv data/processed/call_fraud_train.csv]
                                  [--output models/tfidf_call_vectorizer.pkl]
                                  [--max-features 500]
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sentinel_edge.features.tfidf import TfidfFeatureExtractor


def fit_and_save(
    train_csv: str,
    output_path: str,
    max_features: int = 500,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    sublinear_tf: bool = True,
) -> TfidfFeatureExtractor:
    """Fit a TF-IDF vectorizer on the training corpus and save.

    Parameters
    ----------
    train_csv : str
        Path to the training CSV (must contain a ``text`` column).
    output_path : str
        Where to write the fitted vectorizer (joblib format).
    max_features : int
        Maximum vocabulary size.
    ngram_range : tuple[int, int]
        N-gram range for the vectorizer.
    min_df : int
        Minimum document frequency.
    sublinear_tf : bool
        Use sublinear TF scaling.

    Returns
    -------
    TfidfFeatureExtractor
        The fitted extractor.
    """
    print(f"Loading training data from {train_csv} ...")
    df = pd.read_csv(train_csv, dtype={"text": str})
    texts = df["text"].fillna("").tolist()
    print(f"  {len(texts):,} training documents loaded")

    print(f"Fitting TF-IDF vectorizer (max_features={max_features}, "
          f"ngram_range={ngram_range}) ...")
    extractor = TfidfFeatureExtractor(model_path=None)

    # Override default vectorizer with the requested params
    from sklearn.feature_extraction.text import TfidfVectorizer
    extractor.vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )

    extractor.fit(texts)

    print(f"  Vocabulary size: {extractor.n_features}")
    print(f"Saving vectorizer to {output_path} ...")
    extractor.save(output_path)
    print("Done!")

    return extractor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit TF-IDF vectorizer on training data."
    )
    parser.add_argument(
        "--train-csv",
        default="data/processed/call_fraud_train.csv",
        help="Training CSV path (default: data/processed/call_fraud_train.csv)",
    )
    parser.add_argument(
        "--output",
        default="models/tfidf_call_vectorizer.pkl",
        help="Output path for fitted vectorizer (default: models/tfidf_call_vectorizer.pkl)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=500,
        help="Maximum vocabulary size (default: 500)",
    )
    args = parser.parse_args()

    fit_and_save(
        train_csv=args.train_csv,
        output_path=args.output,
        max_features=args.max_features,
    )


if __name__ == "__main__":
    main()
