"""Turns per-frame landmark sequences into model inputs."""
import numpy as np

from src.config import SEQ_LEN


def resample_sequence(frames, n=SEQ_LEN):
    """Pick n evenly spaced frames from a variable-length sequence."""
    frames = np.asarray(frames, dtype=np.float32)
    if len(frames) == 0:
        raise ValueError("empty sequence")
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return frames[idx]


def aggregate_features(seq):
    """Video-level statistics used by the Random Forest / SVM baselines.

    Input : (T, F) or (N, T, F)
    Output: (N, 5*F)  -> mean, std, min, max over time + mean absolute velocity
    """
    arr = np.asarray(seq, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    mean = arr.mean(axis=1)
    std = arr.std(axis=1)
    mn = arr.min(axis=1)
    mx = arr.max(axis=1)
    vel = np.abs(np.diff(arr, axis=1)).mean(axis=1)
    return np.concatenate([mean, std, mn, mx, vel], axis=1)
