"""Unified feature pipeline combining handcrafted and text features.

Supports two modes:
  - ``"tfidf"`` (default, backward compatible):
    18 handcrafted + 500 TF-IDF = 518 dimensions
  - ``"embedding"``:
    18 handcrafted + 384 MiniLM sentence embeddings = 402 dimensions
"""

from __future__ import annotations

import numpy as np

from sentinel_edge.features.handcrafted import extract_handcrafted_features
from sentinel_edge.features.tfidf import TfidfFeatureExtractor

# Dimensionality constants
HANDCRAFTED_DIM = 18
TFIDF_DIM = 500
EMBEDDING_DIM = 384
TFIDF_TOTAL_DIM = HANDCRAFTED_DIM + TFIDF_DIM      # 518
EMBEDDING_TOTAL_DIM = HANDCRAFTED_DIM + EMBEDDING_DIM  # 402

# Backward-compatible aliases
TOTAL_DIM = TFIDF_TOTAL_DIM


class FeaturePipeline:
    """Combines handcrafted and text features into a single vector.

    Parameters
    ----------
    tfidf_path : str | None
        Path to a pre-fitted TF-IDF vectorizer (joblib).  If *None* the
        TF-IDF component will be initialised un-fitted and will return
        zero vectors until ``fit_tfidf`` is called.
    mode : str
        Feature extraction mode:
        - ``"tfidf"`` -- handcrafted (18) + TF-IDF (500) = 518 dims
        - ``"embedding"`` -- handcrafted (18) + MiniLM (384) = 402 dims
    embedding_model : str
        Sentence-transformer model name (only used when ``mode="embedding"``).
    """

    def __init__(
        self,
        tfidf_path: str | None = None,
        mode: str = "tfidf",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        if mode not in ("tfidf", "embedding"):
            raise ValueError(
                f"Unknown mode '{mode}'. Expected 'tfidf' or 'embedding'."
            )
        self.mode = mode
        self._embedding_model_name = embedding_model

        # TF-IDF component (always initialised for backward compat)
        self.tfidf = TfidfFeatureExtractor(tfidf_path)

        # Embedding component (lazy-loaded only when needed)
        self._embedder = None

    # ------------------------------------------------------------------
    # Lazy embedding loader
    # ------------------------------------------------------------------

    def _get_embedder(self):
        """Return the EmbeddingExtractor, creating it on first use."""
        if self._embedder is None:
            from sentinel_edge.features.embedding import EmbeddingExtractor

            self._embedder = EmbeddingExtractor(self._embedding_model_name)
        return self._embedder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> np.ndarray:
        """Produce a feature vector for *text*.

        The dimensionality depends on the configured mode:
          - ``"tfidf"``:     518 dims (18 handcrafted + 500 TF-IDF)
          - ``"embedding"``: 402 dims (18 handcrafted + 384 embeddings)

        Parameters
        ----------
        text : str
            A single document (SMS, transcript sentence, etc.).

        Returns
        -------
        np.ndarray
            1-D float array.
        """
        handcrafted = extract_handcrafted_features(text)
        handcrafted_arr = np.array(
            list(handcrafted.values()), dtype=np.float64
        )

        if self.mode == "embedding":
            embedder = self._get_embedder()
            emb = embedder.encode(text).astype(np.float64)
            return np.concatenate([handcrafted_arr, emb])
        else:
            tfidf_feats = self.tfidf.transform(text)
            return np.concatenate([handcrafted_arr, tfidf_feats])

    def extract_batch(self, texts: list[str]) -> np.ndarray:
        """Extract features for a batch of texts.

        Parameters
        ----------
        texts : list[str]
            List of documents.

        Returns
        -------
        np.ndarray
            2-D float32 array of shape ``(len(texts), feature_dim)``.
        """
        # Handcrafted features for all texts
        hc_list = []
        for t in texts:
            feats = extract_handcrafted_features(t)
            hc_list.append(list(feats.values()))
        hc_matrix = np.array(hc_list, dtype=np.float32)

        if self.mode == "embedding":
            embedder = self._get_embedder()
            emb_matrix = embedder.encode_batch(texts)
            return np.hstack([hc_matrix, emb_matrix])
        else:
            tfidf_matrix = np.zeros(
                (len(texts), self.tfidf.n_features), dtype=np.float32
            )
            for i, t in enumerate(texts):
                tfidf_matrix[i] = self.tfidf.transform(t).astype(np.float32)
            return np.hstack([hc_matrix, tfidf_matrix])

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
    def feature_dim(self) -> int:
        """Total feature vector dimensionality for the current mode."""
        if self.mode == "embedding":
            return EMBEDDING_TOTAL_DIM
        return TFIDF_TOTAL_DIM

    @property
    def is_ready(self) -> bool:
        """True when the pipeline is ready for feature extraction.

        - In ``"tfidf"`` mode: requires a fitted TF-IDF vectorizer.
        - In ``"embedding"`` mode: always ready (embedder lazy-loads).
        """
        if self.mode == "embedding":
            return True
        return self.tfidf.is_fitted
