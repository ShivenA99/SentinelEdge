"""Sentence embedding feature extractor using all-MiniLM-L6-v2.

Produces 384-dimensional dense embeddings that capture semantic meaning,
replacing sparse 500-dim TF-IDF vectors. Falls back to a deterministic
hash-based embedding when sentence-transformers is unavailable.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384


class EmbeddingExtractor:
    """Dense sentence embedding extractor.

    Uses the ``all-MiniLM-L6-v2`` sentence-transformer model for
    high-quality 384-dim embeddings.  If the model cannot be loaded
    (missing dependency or download failure) a deterministic hash-based
    fallback is used so the pipeline never crashes.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for the sentence-transformer.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._is_fallback = False

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Lazy-load the sentence-transformers model."""
        if self._model is not None or self._is_fallback:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            logger.info(
                "Loaded sentence-transformer model '%s'", self.model_name
            )
        except Exception as exc:
            logger.warning(
                "Could not load sentence-transformer '%s': %s  "
                "-- using deterministic hash fallback (non-semantic).",
                self.model_name,
                exc,
            )
            self._is_fallback = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """Encode a single sentence to a 384-dim float32 vector.

        Parameters
        ----------
        text : str
            Input sentence.

        Returns
        -------
        np.ndarray
            1-D array of shape ``(384,)`` with dtype ``float32``.
        """
        self._load()
        if self._is_fallback:
            return self._fallback_encode(text)
        return self._model.encode(text, convert_to_numpy=True).astype(
            np.float32
        )

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of sentences.

        Parameters
        ----------
        texts : list[str]
            List of input sentences.

        Returns
        -------
        np.ndarray
            2-D array of shape ``(len(texts), 384)`` with dtype ``float32``.
        """
        self._load()
        if self._is_fallback:
            return np.array(
                [self._fallback_encode(t) for t in texts], dtype=np.float32
            )
        return self._model.encode(
            texts, convert_to_numpy=True, show_progress_bar=len(texts) > 100
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Fallback embedding
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_encode(text: str) -> np.ndarray:
        """Deterministic hash-based embedding (non-semantic).

        Produces a normalised 384-dim float32 vector from the SHA-512
        hash of the lowercased input.  This is NOT a semantic embedding
        but allows the full pipeline to run without the heavy
        ``sentence-transformers`` dependency.

        Parameters
        ----------
        text : str
            Input sentence.

        Returns
        -------
        np.ndarray
            Unit-norm 1-D array of shape ``(384,)``.
        """
        seed = text.lower().encode("utf-8")
        h = hashlib.sha512(seed).digest()

        # Extend the hash to fill 384 float32 values (384 * 4 = 1536 bytes)
        extended = b""
        for i in range(24):  # 24 * 64 bytes = 1536 bytes
            extended += hashlib.sha512(h + i.to_bytes(4, "big")).digest()

        arr = np.frombuffer(extended[: EMBEDDING_DIM * 4], dtype=np.float32).copy()

        # Normalise to unit vector
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr /= norm
        return arr

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the output embeddings."""
        return EMBEDDING_DIM

    @property
    def using_fallback(self) -> bool:
        """Whether the extractor is using the hash-based fallback."""
        self._load()
        return self._is_fallback
