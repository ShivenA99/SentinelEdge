"""Live microphone audio capture for demo.

Provides a thread-safe audio capture interface that reads from the system
microphone via PyAudio and exposes chunks through a queue.  This allows
the WebSocket server to ingest live audio without blocking the event loop.

Note: PyAudio (``pip install pyaudio``) is required for live capture.
It is an optional dependency -- the demo works fine with pre-recorded
transcripts when PyAudio is not installed.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import numpy as np


class LiveMicCapture:
    """Capture audio from the system microphone.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz (default 16 kHz for Whisper).
    chunk_size : int
        Number of frames per read from the audio device.
    channels : int
        Number of audio channels (1 = mono).
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        chunk_size: int = 1024,
        channels: int = 1,
        input_device: str | int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.input_device = input_device
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stream = None
        self._pyaudio_instance = None
        self._backend: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start capturing audio in a background thread.

        Uses PyAudio when available, otherwise falls back to sounddevice.

        Raises
        ------
        ImportError
            If neither PyAudio nor sounddevice is installed.
        RuntimeError
            If the capture is already running.
        """
        if self._running:
            raise RuntimeError("Capture is already running")

        try:
            import pyaudio  # noqa: F401
            self._backend = "pyaudio"
        except ImportError:
            try:
                import sounddevice  # noqa: F401
                self._backend = "sounddevice"
            except ImportError as exc:
                raise ImportError(
                    "Live microphone capture requires PyAudio or sounddevice. "
                    "Install one with: pip install pyaudio  OR  pip install sounddevice"
                ) from exc

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="mic-capture"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop capturing and release the audio device."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def get_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get next audio chunk from the queue.

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait for a chunk.

        Returns
        -------
        np.ndarray or None
            Float32 audio chunk, or ``None`` if the timeout expired.
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain_queue(self) -> None:
        """Drop any buffered audio chunks currently waiting in the queue."""
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    @property
    def is_running(self) -> bool:
        """Whether the capture loop is active."""
        return self._running

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Background thread that reads from the microphone.

        Opens an input stream (PyAudio or sounddevice), reads chunks,
        and puts float32 audio on the queue for the main application
        to consume.
        """
        if self._backend == "sounddevice":
            self._capture_loop_sounddevice()
            return

        import pyaudio

        pa = pyaudio.PyAudio()
        self._pyaudio_instance = pa

        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=(self.input_device if isinstance(self.input_device, int) else None),
            )
            self._stream = stream

            while self._running:
                try:
                    raw_data = stream.read(
                        self.chunk_size, exception_on_overflow=False
                    )
                except OSError:
                    # Audio device error -- skip this chunk
                    continue

                # Convert int16 bytes to float32 in [-1, 1]
                audio = (
                    np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
                self.audio_queue.put(audio)

        finally:
            if self._stream is not None:
                try:
                    self._stream.stop_stream()
                    self._stream.close()
                except OSError:
                    pass
                self._stream = None

            pa.terminate()
            self._pyaudio_instance = None

    def _capture_loop_sounddevice(self) -> None:
        """Capture loop implemented with sounddevice InputStream."""
        import sounddevice as sd

        def _callback(indata, frames, _time_info, status):
            if status:
                return
            # sounddevice already provides float32 in [-1, 1] when dtype=float32
            audio = np.asarray(indata[:, 0], dtype=np.float32).copy()
            self.audio_queue.put(audio)

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self.chunk_size,
                device=self.input_device,
                callback=_callback,
            ) as stream:
                self._stream = stream
                while self._running:
                    time.sleep(0.05)
        finally:
            self._stream = None
