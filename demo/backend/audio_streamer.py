"""Stream audio files through the detection pipeline.

Reads WAV files and yields fixed-duration chunks of audio samples,
simulating real-time microphone input for the fraud-detection pipeline.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np


class AudioStreamer:
    """Reads a WAV file and streams chunks to simulate real-time audio.

    Parameters
    ----------
    sample_rate : int
        Expected sample rate of the input WAV files (default 16 kHz).
    """

    def __init__(self, sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stream_file(
        self, wav_path: str, chunk_duration: float = 1.0
    ):
        """Generator that yields audio chunks from a WAV file.

        Each chunk is *chunk_duration* seconds of float32 audio normalised
        to the range ``[-1, 1]``.  The final chunk may be shorter than the
        requested duration.

        Parameters
        ----------
        wav_path : str
            Path to a 16-bit PCM WAV file.
        chunk_duration : float
            Duration of each chunk in seconds.

        Yields
        ------
        np.ndarray
            1-D float32 array of audio samples.

        Raises
        ------
        FileNotFoundError
            If *wav_path* does not exist.
        ValueError
            If the WAV file has an unexpected sample rate.
        """
        path = Path(wav_path)
        if not path.exists():
            raise FileNotFoundError(f"WAV file not found: {wav_path}")

        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()

            if framerate != self.sample_rate:
                raise ValueError(
                    f"Expected sample rate {self.sample_rate}, "
                    f"got {framerate} in {wav_path}"
                )

            chunk_frames = int(chunk_duration * framerate)

            frames_read = 0
            while frames_read < n_frames:
                to_read = min(chunk_frames, n_frames - frames_read)
                raw_bytes = wf.readframes(to_read)
                frames_read += to_read

                # Convert raw bytes to numpy float32
                audio = self._bytes_to_float32(
                    raw_bytes, n_channels=n_channels, sampwidth=sampwidth
                )
                yield audio

    def generate_silence(self, duration: float) -> np.ndarray:
        """Generate silent audio.

        Parameters
        ----------
        duration : float
            Duration in seconds.

        Returns
        -------
        np.ndarray
            1-D float32 zero array.
        """
        return np.zeros(int(duration * self.sample_rate), dtype=np.float32)

    def generate_tone(
        self,
        frequency: float = 440.0,
        duration: float = 1.0,
        amplitude: float = 0.5,
    ) -> np.ndarray:
        """Generate a simple sine-wave tone (useful for testing).

        Parameters
        ----------
        frequency : float
            Tone frequency in Hz.
        duration : float
            Duration in seconds.
        amplitude : float
            Peak amplitude (0.0 -- 1.0).

        Returns
        -------
        np.ndarray
            1-D float32 sine wave.
        """
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples, dtype=np.float32)
        return np.float32(amplitude) * np.sin(
            np.float32(2.0 * np.pi * frequency) * t
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _bytes_to_float32(
        raw_bytes: bytes,
        n_channels: int = 1,
        sampwidth: int = 2,
    ) -> np.ndarray:
        """Convert raw WAV bytes to a mono float32 array in ``[-1, 1]``.

        Parameters
        ----------
        raw_bytes : bytes
            Raw PCM bytes from ``wave.readframes()``.
        n_channels : int
            Number of audio channels (1 = mono, 2 = stereo).
        sampwidth : int
            Sample width in bytes (1, 2, or 4).

        Returns
        -------
        np.ndarray
            1-D float32 array normalised to ``[-1.0, 1.0]``.
        """
        if sampwidth == 1:
            dtype = np.uint8
            max_val = 128.0
            offset = 128.0  # unsigned 8-bit
        elif sampwidth == 2:
            dtype = np.int16
            max_val = 32768.0
            offset = 0.0
        elif sampwidth == 4:
            dtype = np.int32
            max_val = 2_147_483_648.0
            offset = 0.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        samples = np.frombuffer(raw_bytes, dtype=dtype).astype(np.float32)

        if offset != 0.0:
            samples = samples - offset

        samples = samples / max_val

        # Mix to mono if stereo
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        return samples
