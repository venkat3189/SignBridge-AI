"""Sign boundary detection and confidence filtering for the live stream."""
from src.config import SEQ_LEN
from src.feature_extraction import resample_sequence

MIN_FRAMES = 6      # ignore very short hand movements
MAX_FRAMES = 150    # force a decision for very long movements
END_GRACE = 6       # frames without a raised hand that end a sign


class SignSegmenter:
    """Collects frames while a hand is raised; when the hands drop for END_GRACE
    frames the sign is complete and returned as a (30, 182) sequence."""

    def __init__(self, min_frames=MIN_FRAMES, max_frames=MAX_FRAMES, end_grace=END_GRACE):
        self.min_frames, self.max_frames, self.end_grace = min_frames, max_frames, end_grace
        self.buf, self.gap = [], 0

    @property
    def state(self):
        return "signing" if self.buf else "idle"

    def reset(self):
        self.buf, self.gap = [], 0

    def update(self, vec, active):
        if active:
            self.buf.append(vec)
            self.gap = 0
            if len(self.buf) >= self.max_frames:
                return self._finish()
        elif self.buf:
            self.buf.append(vec)
            self.gap += 1
            if self.gap >= self.end_grace:
                return self._finish()
        return None

    def _finish(self):
        frames = self.buf[: len(self.buf) - self.gap] if self.gap else self.buf
        self.reset()
        if len(frames) < self.min_frames:
            return None
        return resample_sequence(frames, SEQ_LEN)


def accept(result, threshold, min_margin=0.0):
    """True when the prediction is confident enough to be added to the sentence."""
    return result["confidence"] >= threshold and result["margin"] >= min_margin
