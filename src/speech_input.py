
"""Local speech-to-text using SoundDevice and Faster-Whisper."""

import time
import numpy as np

SAMPLE_RATE = 16000
_model = None


def get_microphone():
    """Check whether an input microphone is available."""
    import sounddevice as sd

    devices = sd.query_devices()
    inputs = [
        (index, device)
        for index, device in enumerate(devices)
        if device["max_input_channels"] > 0
    ]

    if not inputs:
        raise RuntimeError(
            "No microphone detected. Connect or enable a microphone."
        )

    return inputs


def record(seconds=5, status_callback=None):
    """Record audio from the default microphone."""
    import sounddevice as sd

    get_microphone()

    chunks = []
    total_samples = int(seconds * SAMPLE_RATE)

    def callback(indata, frames, timing, status):
        if status:
            print("Microphone status:", status)

        chunks.append(indata.copy())

        if status_callback:
            recorded = sum(len(chunk) for chunk in chunks)
            progress = min(recorded / total_samples, 1.0)
            status_callback(progress)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=callback
        ):
            time.sleep(seconds)

    except Exception as exc:
        raise RuntimeError(f"Microphone recording failed: {exc}") from exc

    if not chunks:
        raise RuntimeError("No audio was captured.")

    audio = np.concatenate(chunks, axis=0).flatten()

    if len(audio) == 0:
        raise RuntimeError("The microphone returned empty audio.")

    return audio


def _get_model(size="base"):
    """Load Whisper only once."""
    global _model

    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel(
            size,
            device="cpu",
            compute_type="int8"
        )

    return _model


def transcribe(audio, language=None):
    """
    Convert audio to text.

    language:
        None = automatic detection
        en   = English
        hi   = Hindi
        te   = Telugu

    Returns:
        (recognized_text, detected_language)
    """

    audio = np.asarray(audio, dtype=np.float32).flatten()

    if len(audio) == 0:
        return "", language or ""

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains invalid values.")

    if float(np.max(np.abs(audio))) < 0.005:
        return "", language or ""

    segments, info = _get_model().transcribe(
        audio,
        language=language,
        vad_filter=True
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    return text, info.language
