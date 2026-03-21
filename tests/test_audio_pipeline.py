"""Tests for audio pipeline components."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentinel_edge.audio.ring_buffer import RingBuffer
from sentinel_edge.audio.sentence_splitter import SentenceSplitter


class TestRingBuffer:
    def test_write_and_read(self):
        buf = RingBuffer(duration_seconds=1.0, sample_rate=16000)
        samples = np.ones(8000, dtype=np.int16)
        buf.write(samples)
        result = buf.read_last(0.5)
        assert len(result) == 8000
        assert np.all(result == 1)

    def test_circular_overwrite(self):
        buf = RingBuffer(duration_seconds=1.0, sample_rate=16000)
        # Write more than buffer size
        first = np.ones(16000, dtype=np.int16)
        second = np.ones(16000, dtype=np.int16) * 2
        buf.write(first)
        buf.write(second)
        result = buf.read_last(1.0)
        assert np.all(result == 2)  # First data overwritten

    def test_read_last_partial(self):
        buf = RingBuffer(duration_seconds=2.0, sample_rate=16000)
        samples = np.arange(32000, dtype=np.int16)
        buf.write(samples)
        result = buf.read_last(1.0)
        assert len(result) == 16000

    def test_clear(self):
        buf = RingBuffer(duration_seconds=1.0, sample_rate=16000)
        buf.write(np.ones(8000, dtype=np.int16))
        buf.clear()
        result = buf.read_last(1.0)
        assert np.all(result == 0)


class TestSentenceSplitter:
    def test_simple_sentence(self):
        splitter = SentenceSplitter()
        sentences = splitter.feed("Hello world. How are you? ")
        assert len(sentences) >= 2

    def test_incremental_feed(self):
        splitter = SentenceSplitter()
        result1 = splitter.feed("Hello")
        assert len(result1) == 0  # No complete sentence yet
        result2 = splitter.feed(" world. ")
        assert len(result2) == 1  # Now we have one

    def test_exclamation(self):
        splitter = SentenceSplitter()
        sentences = splitter.feed("Stop! This is urgent! ")
        assert len(sentences) >= 2

    def test_question(self):
        splitter = SentenceSplitter()
        sentences = splitter.feed("What is your account number? Tell me now. ")
        assert len(sentences) >= 2

    def test_flush(self):
        splitter = SentenceSplitter()
        splitter.feed("Partial sentence without ending")
        remaining = splitter.flush()
        assert remaining is not None
        assert "Partial" in remaining

    def test_reset(self):
        splitter = SentenceSplitter()
        splitter.feed("Some text")
        splitter.reset()
        assert splitter.buffer == ""
