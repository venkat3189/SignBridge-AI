"""Checks that every library SignBridge AI needs is installed and working.
Run:  python scripts/verify_setup.py
Add --camera to also test the webcam.
"""
import sys
import sqlite3
import importlib

results = []


def check(name, fn):
    try:
        detail = fn()
        results.append((name, True, detail or ""))
    except Exception as exc:  # report every failure instead of crashing
        results.append((name, False, f"{type(exc).__name__}: {exc}"))


def version_of(module_name):
    def _inner():
        mod = importlib.import_module(module_name)
        return getattr(mod, "__version__", "installed")
    return _inner


def test_mediapipe():
    import mediapipe as mp
    import numpy as np
    # Uses the legacy 'solutions' API, which this project depends on.
    with mp.solutions.holistic.Holistic(static_image_mode=True) as holistic:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        holistic.process(blank)
    return f"Holistic OK (mediapipe {mp.__version__})"


def test_tensorflow():
    import tensorflow as tf
    from tensorflow import keras
    model = keras.Sequential([keras.layers.Input((30, 10)), keras.layers.GRU(8),
                              keras.layers.Dense(3, activation="softmax")])
    return f"TF {tf.__version__}, Keras {keras.__version__}, GRU model builds ({model.count_params()} params)"


def test_tts():
    import pyttsx3
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    names = [v.name for v in voices]
    return f"{len(voices)} voice(s): {', '.join(names)}"


def test_audio():
    import sounddevice as sd
    inputs = [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
    if not inputs:
        raise RuntimeError("No microphone found")
    return f"{len(inputs)} input device(s), e.g. {inputs[0]}"


def test_whisper_import():
    from faster_whisper import WhisperModel  # noqa: F401
    return "import OK (model downloads on first use)"


def test_sqlite():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.execute("INSERT INTO t VALUES (1)")
    return f"SQLite {sqlite3.sqlite_version}"


def test_camera():
    import cv2
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read a frame from camera 0")
    return f"frame size {frame.shape[1]}x{frame.shape[0]}"


print(f"Python {sys.version.split()[0]}")
if not sys.version.startswith("3.11"):
    print("WARNING: this project is planned for Python 3.11\n")

check("numpy", version_of("numpy"))
check("opencv", version_of("cv2"))
check("pandas", version_of("pandas"))
check("scikit-learn", version_of("sklearn"))
check("plotly", version_of("plotly"))
check("streamlit", version_of("streamlit"))
check("joblib", version_of("joblib"))
check("mediapipe holistic", test_mediapipe)
check("tensorflow/keras", test_tensorflow)
check("text-to-speech", test_tts)
check("microphone", test_audio)
check("faster-whisper", test_whisper_import)
check("sqlite", test_sqlite)
if "--camera" in sys.argv:
    check("webcam", test_camera)

print()
for name, ok, detail in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name:20s} {detail}")

failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed.")
sys.exit(1 if failed else 0)