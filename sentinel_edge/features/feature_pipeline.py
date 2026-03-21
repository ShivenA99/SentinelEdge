"""Unified feature pipeline combining handcrafted and TF-IDF features.

Produces a single 518-dimensional feature vector per text sample:
  - 18 handcrafted linguistic / structural features
  - 500 TF-IDF features (unigrams + bigrams)
"""

from __future__ import annotations

import numpy as np

from sentinel_edge.features.handcrafted import extract_handcrafted_features
from sentinel_edge.features.tfidf import TfidfFeatureExtractor

# Dimensionality constants
HANDCRAFTED_DIM = 18
TFIDF_DIM = 500
TOTAL_DIM = HANDCRAFTED_DIM + TFIDF_DIM  # 518


class FeaturePipeline:
    """Combines handcrafted and TF-IDF features into a single vector.

    Parameters
    ----------
    tfidf_path : str | None
        Path to a pre-fitted TF-IDF vectorizer (joblib).  If *None* the
        TF-IDF component will be initialised un-fitted and will return
        zero vectors until ``fit_tfidf`` is called.
    """

    def __init__(self, tfidf_path: str | None = None) -> None:
        self.tfidf = TfidfFeatureExtractor(tfidf_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> np.ndarray:
        """Produce a 518-dimensional feature vector for *text*.

        Parameters
        ----------
        text : str
            A single document (SMS, transcript sentence, etc.).

        Returns
        -------
        np.ndarray
            1-D float64 array of shape ``(518,)``.
        """
        handcrafted = extract_handcrafted_features(text)
        tfidf_feats = self.tfidf.transform(text)

        handcrafted_arr = np.array(
            list(handcrafted.values()), dtype=np.float64
        )
        return np.concatenate([handcrafted_arr, tfidf_feats])

    def extract_handcrafted_only(self, text: str) -> dict[str, float]:
        """Return only the 18 handcrafted features as a dict.

        Useful for interpretability: the alert engine generates
        human-readable reasons from these named features.
        """
        return extract_handcrafted_features(text)

    def fit_tfidf(self, texts: list[str]) -> None:
        """Fit the TF-IDF component on a training corpus.

        Parameters
        ----------
        texts : list[str]
            Training documents.
        """
        self.tfidf.fit(texts)

    def save_tfidf(self, path: str) -> None:
        """Persist the fitted TF-IDF vectorizer to *path*."""
        self.tfidf.save(path)

    @property
    def is_ready(self) -> bool:
        """True when the TF-IDF component is fitted / loaded."""
        return self.tfidf.is_fitted
