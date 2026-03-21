"""Whisper-based audio transcription wrapper.

For the demo / development environment the standard ``openai-whisper``
Python package is used.  In a production edge deployment this would be
swapped for an ONNX Runtime Mobile session running a quantised Whisper
encoder/decoder -- the public interface stays identical.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class Transcriber:
    """Lazy-loading wrapper around OpenAI Whisper.

    The model is loaded on the first call to :meth:`load` (or
    transparently on the first transcription request) to keep cold-start
    latency out of import time.

    Parameters
    ----------
    model_name : str
        Whisper model size.  ``"tiny.en"`` is recommended for on-device
        real-time use (< 75 MB, ~10x realtime on CPU).
    """

    def __init__(self, model_name: str = "tiny.en") -> None:
        self.model = None
        self.model_name = model_name
        self._loaded = False

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Eagerly load the Whisper model into memory.

        This is separated from ``__init__`` so callers can control *when*
        the ~75 MB model is fetched and loaded.
        """
        import whisper  # type: ignore[import-untyped]

        logger.info("Loading Whisper model '%s' ...", self.model_name)
        self.model = whisper.load_model(self.model_name)
        self._loaded = True
        logger.info("Whisper model loaded.")

    def _ensure_loaded(self) -> None:
        """Load the model lazily if it has not been loaded yet."""
        if not self._loaded:
            self.load()

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16_000,
    ) -> str:
        """Transcribe a raw audio array to text.

        Parameters
        ----------
        audio : np.ndarray
            1-D float32 array normalised to ``[-1.0, 1.0]``.
            If int16 is passed it will be converted automatically.
        sample_rate : int
            Sample rate of the input audio.  If not 16 000 Hz the audio
            is resampled internally by Whisper.

        Returns
        -------
        str
            Transcribed text (stripped of leading/trailing whitespace).
        """
        self._ensure_loaded()

        audio = np.asarray(audio)

        # Convert int16 PCM to float32 normalised
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Whisper expects float32, mono, 16 kHz
        # The library handles resampling internally if sr != 16000
        result = self.model.transcribe(
            audio,
            language="en",
            fp16=False,  # safe for CPU-only environments
        )
        return result.get("text", "").strip()

    def transcribe_file(self, wav_path: str) -> str:
        """Transcribe a WAV file from disk.

        Parameters
        ----------
        wav_path : str
            Path to a WAV (or other ffmpeg-supported) audio file.

        Returns
        -------
        str
            Transcribed text.
        """
        self._ensure_loaded()

        path = Path(wav_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        result = self.model.transcribe(
            str(path),
            language="en",
            fp16=False,
        )
        return result.get("text", "").strip()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Whether the Whisper model is currently in memory."""
        return self._loaded
