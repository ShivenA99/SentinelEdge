"""Rule-based sentence segmentation for streaming transcripts.

Designed for incremental use: text is fed in as it arrives from the
transcriber and complete sentences are emitted as soon as a sentence
boundary is detected.
"""

from __future__ import annotations

import re


# Sentence-ending patterns:
#   - Standard punctuation: . ! ?
#   - Ellipsis / long pause: ...  (treated as a sentence boundary)
# We look for a sentence terminator followed by whitespace or end-of-string
# to avoid splitting on abbreviations like "U.S.A." mid-sentence.
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])"       # lookbehind for sentence-ending punctuation
    r"(?:\s+|$)"         # followed by whitespace or end-of-string
    r"|"
    r"(?<=\.\.\.)"       # lookbehind for ellipsis
    r"(?:\s+|$)"         # followed by whitespace or end-of-string
)


class SentenceSplitter:
    """Incrementally segments streaming transcript text into sentences.

    Usage::

        splitter = SentenceSplitter()
        for chunk in transcript_stream:
            new_sentences = splitter.feed(chunk)
            for sentence in new_sentences:
                process(sentence)
        # At end-of-stream, flush any remaining text
        leftover = splitter.flush()
        if leftover:
            process(leftover)
    """

    def __init__(self) -> None:
        self.buffer: str = ""
        self.sentences: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, text: str) -> list[str]:
        """Append *text* to the internal buffer and extract complete sentences.

        Parameters
        ----------
        text : str
            New transcript text (may be a partial sentence, a full
            sentence, or multiple sentences).

        Returns
        -------
        list[str]
            Newly completed sentences (may be empty if no boundary was
            found yet).
        """
        self.buffer += text
        completed: list[str] = []

        # Repeatedly split at the first sentence boundary
        while True:
            match = _SENTENCE_BOUNDARY.search(self.buffer)
            if match is None:
                break

            boundary_end = match.end()
            sentence = self.buffer[:boundary_end].strip()

            if sentence:
                completed.append(sentence)
                self.sentences.append(sentence)

            self.buffer = self.buffer[boundary_end:]

        return completed

    def flush(self) -> str | None:
        """Return any remaining buffered text as the final sentence.

        Returns
        -------
        str | None
            The leftover text, or *None* if the buffer is empty.
        """
        remaining = self.buffer.strip()
        if remaining:
            self.sentences.append(remaining)
            self.buffer = ""
            return remaining

        self.buffer = ""
        return None

    def reset(self) -> None:
        """Clear all internal state for a new transcript."""
        self.buffer = ""
        self.sentences.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def all_sentences(self) -> list[str]:
        """All sentences extracted so far (including flushed)."""
        return list(self.sentences)

    @property
    def pending(self) -> str:
        """Text currently waiting in the buffer (not yet a full sentence)."""
        return self.buffer
