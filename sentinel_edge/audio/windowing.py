"""Sliding window extraction and mel-spectrogram computation.

Implements a lightweight mel-spectrogram using only NumPy and SciPy
(no librosa dependency), suitable for edge deployment where every
megabyte of installed packages matters.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq  # type: ignore[import-untyped]
from scipy.signal import get_window  # type: ignore[import-untyped]

from sentinel_edge.audio.ring_buffer import RingBuffer


def _hz_to_mel(hz: float) -> float:
    """Convert frequency in Hz to the mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    """Convert mel-scale value back to Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Build a mel-scaled triangular filterbank matrix.

    Parameters
    ----------
    n_mels : int
        Number of mel bands.
    n_fft : int
        FFT size (determines frequency resolution).
    sample_rate : int
        Audio sample rate in Hz.
    fmin, fmax : float
        Minimum / maximum frequency for the filterbank.

    Returns
    -------
    np.ndarray
        Shape ``(n_mels, n_fft // 2 + 1)`` filterbank matrix.
    """
    if fmax is None:
        fmax = sample_rate / 2.0

    # Mel-spaced centre frequencies
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])

    # Map Hz points to FFT bin indices
    n_freqs = n_fft // 2 + 1
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for i in range(n_mels):
        left = bin_points[i]
        centre = bin_points[i + 1]
        right = bin_points[i + 2]

        # Rising slope
        for j in range(left, centre):
            if centre != left:
                filterbank[i, j] = (j - left) / (centre - left)

        # Falling slope
        for j in range(centre, right):
            if right != centre:
                filterbank[i, j] = (right - j) / (right - centre)

    return filterbank


class AudioWindower:
    """Extracts fixed-length audio windows and converts them to mel spectrograms.

    Parameters
    ----------
    window_size : float
        Duration of each analysis window in seconds.
    hop_size : float
        Step between consecutive windows in seconds (controls overlap).
    sample_rate : int
        Audio sample rate in Hz.
    n_mels : int
        Number of mel frequency bands.
    n_fft : int
        FFT window size in samples.  Defaults to 512 (~32 ms at 16 kHz).
    hop_length : int
        FFT hop length in samples.  Defaults to 160 (~10 ms at 16 kHz).
    """

    def __init__(
        self,
        window_size: float = 5.0,
        hop_size: float = 1.0,
        sample_rate: int = 16_000,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
    ) -> None:
        self.window_size = window_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Pre-compute the mel filterbank and STFT window
        self._filterbank = _mel_filterbank(
            n_mels=n_mels,
            n_fft=n_fft,
            sample_rate=sample_rate,
        )
        self._fft_window = get_window("hann", n_fft, fftbins=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_window(self, ring_buffer: RingBuffer) -> np.ndarray | None:
        """Grab the latest window from the ring buffer and compute its mel spectrogram.

        Parameters
        ----------
        ring_buffer : RingBuffer
            Active audio ring buffer.

        Returns
        -------
        np.ndarray | None
            Log-mel spectrogram of shape ``(n_mels, T)`` or *None* if the
            ring buffer does not yet contain enough data.
        """
        if ring_buffer.duration_available < self.window_size:
            return None

        samples = ring_buffer.read_last(self.window_size)
        audio_float = samples.astype(np.float32) / 32768.0
        return self.compute_mel_spectrogram(audio_float)

    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute a log-mel spectrogram from a float32 audio waveform.

        Parameters
        ----------
        audio : np.ndarray
            1-D float32 array normalised to ``[-1, 1]``.

        Returns
        -------
        np.ndarray
            Log-mel spectrogram of shape ``(n_mels, T)`` where ``T`` is
            the number of STFT frames.
        """
        audio = np.asarray(audio, dtype=np.float64)

        # --- Short-Time Fourier Transform (STFT) ---
        n_samples = len(audio)
        n_frames = 1 + (n_samples - self.n_fft) // self.hop_length
        if n_frames <= 0:
            # Audio too short for even one FFT frame
            return np.zeros((self.n_mels, 1), dtype=np.float64)

        # Build the spectrogram frame-by-frame
        stft_matrix = np.zeros(
            (self.n_fft // 2 + 1, n_frames), dtype=np.float64
        )
        for t in range(n_frames):
            start = t * self.hop_length
            frame = audio[start : start + self.n_fft] * self._fft_window
            spectrum = rfft(frame)
            stft_matrix[:, t] = np.abs(spectrum) ** 2  # power spectrum

        # --- Mel filterbank application ---
        mel_spec = self._filterbank @ stft_matrix  # (n_mels, T)

        # --- Log compression ---
        # Add a small floor to avoid log(0)
        log_mel = np.log(np.maximum(mel_spec, 1e-10))

        return log_mel
