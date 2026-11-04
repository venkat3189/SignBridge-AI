
"""Speech-to-text using browser audio and Faster-Whisper."""

import io
import wave
import time
from collections.abc import Mapping
from typing import Any, cast

import numpy as np

SAMPLE_RATE = 16000
_model = None


def decode_audio(uploaded_file):
    """Convert browser-recorded WAV audio into a 16 kHz mono float array."""

    try:
        with wave.open(io.BytesIO(uploaded_file.getvalue()), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            original_rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())

        if sample_width == 1:
            audio = (
                np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
                - 128
            ) / 128.0

        elif sample_width == 2:
            audio = (
                np.frombuffer(frames, dtype="<i2").astype(np.float32)
                / 32768.0
            )

        elif sample_width == 3:
            raw = np.frombuffer(frames, dtype=np.uint8)
            raw = raw.reshape(-1, 3)

            values = (
                raw[:, 0].astype(np.int32)
                | (raw[:, 1].astype(np.int32) << 8)
                | (raw[:, 2].astype(np.int32) << 16)
            )

            values = (values ^ 0x800000) - 0x800000
            audio = values.astype(np.float32) / 8388608.0

        elif sample_width == 4:
            audio = (
                np.frombuffer(frames, dtype="<i4").astype(np.float32)
                / 2147483648.0
            )

        else:
            raise ValueError("Unsupported audio format.")

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        # Convert to the 16 kHz sample rate expected by Whisper.
        if original_rate != SAMPLE_RATE:
            new_length = int(
                len(audio) * SAMPLE_RATE / original_rate
            )

            old_positions = np.arange(len(audio))
            new_positions = np.linspace(
                0, len(audio) - 1, new_length
            )

            audio = np.interp(
                new_positions,
                old_positions,
                audio
            ).astype(np.float32)

        return audio.astype(np.float32)

    except Exception as exc:
        raise RuntimeError(
            f"Could not process recorded audio: {exc}"
        ) from exc


def get_microphone():
    """Check whether a local microphone is available."""
    import sounddevice as sd

    devices = cast(list[Mapping[str, Any]], sd.query_devices())

    inputs = [
        (index, device)
        for index, device in enumerate(devices)
        if device["max_input_channels"] > 0
    ]

    if not inputs:
        raise RuntimeError("No microphone detected.")

    return inputs


def record(seconds=5, status_callback=None):
    """Record audio using SoundDevice for local use only."""
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
            status_callback(min(recorded / total_samples, 1.0))

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=callback
        ):
            time.sleep(seconds)

    except Exception as exc:
        raise RuntimeError(
            f"Microphone recording failed: {exc}"
        ) from exc

    if not chunks:
        raise RuntimeError("No audio was captured.")

    return np.concatenate(chunks, axis=0).flatten()


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
    Convert audio into text.

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
