"""Exponential Moving Average (EMA) score accumulator.

Smooths per-sentence fraud scores over a phone call so that a single
high-scoring sentence doesn't cause a spurious alert while sustained
suspicious language steadily raises the risk level.
"""

from __future__ import annotations


class ScoreAccumulator:
    """Maintains an EMA of per-sentence fraud scores.

    Parameters
    ----------
    alpha : float
        Smoothing factor in ``(0, 1]``.  Higher values weight recent
        sentences more heavily.  ``alpha = 1.0`` disables smoothing.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.ema: float = 0.0
        self.count: int = 0
        self.history: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, new_score: float) -> float:
        """Incorporate a new sentence score and return the updated EMA.

        Parameters
        ----------
        new_score : float
            Raw fraud probability for the latest sentence (0.0 -- 1.0).

        Returns
        -------
        float
            Updated EMA value.
        """
        self.history.append(new_score)
        self.count += 1

        if self.count == 1:
            # Bootstrap: initialise EMA to the first observation
            self.ema = new_score
        else:
            self.ema = self.alpha * new_score + (1.0 - self.alpha) * self.ema

        return self.ema

    def reset(self) -> None:
        """Clear accumulated state for a new call."""
        self.ema = 0.0
        self.count = 0
        self.history.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_score(self) -> float:
        """The latest EMA value (0.0 if no data has been accumulated)."""
        return self.ema

    @property
    def peak_score(self) -> float:
        """Highest individual sentence score seen so far."""
        return max(self.history) if self.history else 0.0

    @property
    def mean_score(self) -> float:
        """Simple arithmetic mean of all sentence scores."""
        return sum(self.history) / len(self.history) if self.history else 0.0
