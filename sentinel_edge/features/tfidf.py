"""TF-IDF feature extraction wrapper.

Wraps scikit-learn's TfidfVectorizer with persistence (joblib) and a
convenient single-text transform interface for the inference pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# Default hyper-parameters shared between fit and fallback init
_DEFAULT_MAX_FEATURES = 500
_DEFAULT_NGRAM_RANGE = (1, 2)
_DEFAULT_MIN_DF = 2
_DEFAULT_SUBLINEAR_TF = True


class TfidfFeatureExtractor:
    """Thin wrapper around ``TfidfVectorizer`` for the SentinelEdge pipeline.

    Parameters
    ----------
    model_path : str | None
        Path to a joblib-serialised ``TfidfVectorizer``.  If provided the
        vectorizer is loaded immediately and is ready for ``transform``.
        If *None*, a default (un-fitted) vectorizer is created -- you must
        call ``fit`` before ``transform``.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._fitted = False

        if model_path is not None:
            path = Path(model_path)
            if path.exists():
                self.vectorizer: TfidfVectorizer = joblib.load(path)
                self._fitted = True
                logger.info("Loaded TF-IDF vectorizer from %s", model_path)
            else:
                logger.warning(
                    "TF-IDF model path %s does not exist; creating default vectorizer",
                    model_path,
                )
                self.vectorizer = self._make_default_vectorizer()
        else:
            self.vectorizer = self._make_default_vectorizer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, texts: list[str]) -> None:
        """Fit the vectorizer on a corpus of texts.

        Parameters
        ----------
        texts : list[str]
            Training corpus.  Each element is one document (e.g. a full
            SMS or a single transcript sentence).
        """
        self.vectorizer = self._make_default_vectorizer()
        self.vectorizer.fit(texts)
        self._fitted = True
        logger.info(
            "Fitted TF-IDF vectorizer on %d documents (%d features)",
            len(texts),
            len(self.vectorizer.vocabulary_),
        )

    def transform(self, text: str) -> np.ndarray:
        """Transform a single text into a TF-IDF feature vector.

        Parameters
        ----------
        text : str
            A single document string.

        Returns
        -------
        np.ndarray
            Dense 1-D array of length ``max_features`` (default 500).
            If the vectorizer has not been fitted, returns a zero vector
            so that downstream code does not crash during cold-start.
        """
        if not self._fitted:
            logger.warning(
                "TF-IDF vectorizer is not fitted; returning zero vector"
            )
            return np.zeros(_DEFAULT_MAX_FEATURES, dtype=np.float64)

        sparse_vec = self.vectorizer.transform([text])
        return np.asarray(sparse_vec.toarray()).flatten()

    def save(self, path: str) -> None:
        """Persist the fitted vectorizer to disk via joblib.

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``models/tfidf.joblib``).
        """
        if not self._fitted:
            raise RuntimeError("Cannot save an un-fitted vectorizer.")
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, dest)
        logger.info("Saved TF-IDF vectorizer to %s", path)

    @classmethod
    def load(cls, path: str) -> "TfidfFeatureExtractor":
        """Class-method shortcut to instantiate from a saved model.

        Parameters
        ----------
        path : str
            Path to a joblib-serialised ``TfidfVectorizer``.

        Returns
        -------
        TfidfFeatureExtractor
            Ready-to-use extractor instance.
        """
        return cls(model_path=path)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether the vectorizer has been fitted or loaded."""
        return self._fitted

    @property
    def n_features(self) -> int:
        """Number of output features (vocabulary size, capped by max_features)."""
        if self._fitted:
            return len(self.vectorizer.vocabulary_)
        return _DEFAULT_MAX_FEATURES

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _make_default_vectorizer() -> TfidfVectorizer:
        """Create a new un-fitted vectorizer with the project defaults."""
        return TfidfVectorizer(
            max_features=_DEFAULT_MAX_FEATURES,
            ngram_range=_DEFAULT_NGRAM_RANGE,
            min_df=_DEFAULT_MIN_DF,
            sublinear_tf=_DEFAULT_SUBLINEAR_TF,
        )
