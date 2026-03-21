"""Circular (ring) buffer for raw PCM audio samples.

Provides O(1) writes and efficient reads of the most recent *N* seconds
of audio, designed for real-time phone call capture on memory-constrained
edge devices.  A 30-second buffer at 16 kHz / 16-bit mono consumes
only ~960 KB.
"""

from __future__ import annotations

import numpy as np


class RingBuffer:
    """Fixed-size circular buffer backed by a NumPy int16 array.

    Parameters
    ----------
    duration_seconds : float
        Maximum buffer duration in seconds.
    sample_rate : int
        Audio sample rate in Hz (default 16 000).
    """

    def __init__(
        self,
        duration_seconds: float = 30.0,
        sample_rate: int = 16_000,
    ) -> None:
        self.sample_rate = sample_rate
        self.capacity = int(duration_seconds * sample_rate)
        self.buffer = np.zeros(self.capacity, dtype=np.int16)
        self.write_pos: int = 0
        self.total_written: int = 0  # cumulative sample count

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, samples: np.ndarray) -> None:
        """Append PCM samples to the buffer, wrapping as needed.

        Parameters
        ----------
        samples : np.ndarray
            1-D array of int16 PCM samples.  May be longer than the
            buffer capacity -- only the last ``capacity`` samples are
            retained in that case.
        """
        samples = np.asarray(samples, dtype=np.int16).ravel()
        n = len(samples)

        if n == 0:
            return

        # If the incoming chunk is larger than the buffer, keep only the
        # tail that fits.
        if n >= self.capacity:
            samples = samples[-self.capacity:]
            n = self.capacity
            self.buffer[:] = samples
            self.write_pos = 0
            self.total_written += n
            return

        # Number of samples that fit before wrapping
        first_chunk = min(n, self.capacity - self.write_pos)
        self.buffer[self.write_pos : self.write_pos + first_chunk] = samples[:first_chunk]

        # Wrap-around portion (if any)
        remaining = n - first_chunk
        if remaining > 0:
            self.buffer[:remaining] = samples[first_chunk:]

        self.write_pos = (self.write_pos + n) % self.capacity
        self.total_written += n

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_last(self, duration_seconds: float) -> np.ndarray:
        """Read the most recent *duration_seconds* of audio.

        Parameters
        ----------
        duration_seconds : float
            How many seconds of audio to retrieve.

        Returns
        -------
        np.ndarray
            1-D int16 array.  May be shorter than requested if the
            buffer has not yet been fully written.
        """
        n_requested = int(duration_seconds * self.sample_rate)
        n_available = min(n_requested, self.capacity, self.total_written)

        if n_available == 0:
            return np.array([], dtype=np.int16)

        start = (self.write_pos - n_available) % self.capacity

        if start < self.write_pos:
            return self.buffer[start : self.write_pos].copy()
        else:
            # Data wraps around the end of the buffer
            return np.concatenate([
                self.buffer[start:],
                self.buffer[: self.write_pos],
            ])

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Zero-fill the buffer and reset write position."""
        self.buffer[:] = 0
        self.write_pos = 0
        self.total_written = 0

    @property
    def duration_available(self) -> float:
        """Seconds of valid audio currently in the buffer."""
        return min(self.total_written, self.capacity) / self.sample_rate

    @property
    def is_full(self) -> bool:
        """Whether the buffer has been completely written at least once."""
        return self.total_written >= self.capacity
