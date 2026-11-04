"""Trains a Personalised model: INCLUDE clips + your own webcam clips.

Also reports how the ORIGINAL INCLUDE-only model does on your clips (the live-vs-dataset gap).
Your last ~25% of clips per sign are held out for testing (same session, so optimistic).

Run:  python scripts/train_demo_model.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.config import MODELS_DIR, SELF_DIR, RECORDABLE_SIGNS
from src.data_preprocessing import load_landmark_dataset, sanitize
from src.feature_extraction import aggregate_features


def load_self():
    lookup = {sanitize(n): n for n in RECORDABLE_SIGNS}
    by_sign = {}
    for path in sorted(SELF_DIR.glob("*.npy")):
        key = path.stem.split("__")[0]
        if key in lookup:
            by_sign.setdefault(lookup[key], []).append(np.load(path))
    return by_sign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-weight", type=float, default=4.0, help="importance of your own clips")
    args = ap.parse_args()

    by_sign = load_self()
    by_sign = {s: c for s, c in by_sign.items() if len(c) >= 4}
    if len(by_sign) < 2:
        print("Need at least 2 signs with 4+ recorded clips. Run record_samples.py first.")
        return 1
    signs = sorted(by_sign)
    print("Signs:", signs, {s: len(c) for s, c in by_sign.items()})

    X_inc, labels, _, _ = load_landmark_dataset()
    keep = np.isin(labels, signs)
    X_inc, y_inc = X_inc[keep], np.array([signs.index(l) for l in labels[keep]])

    X_tr, y_tr, w_tr, X_te, y_te = [X_inc], [y_inc], [np.ones(len(y_inc))], [], []
    for i, s in enumerate(signs):
        clips = np.stack(by_sign[s])
        n_test = max(2, round(0.25 * len(clips)))
        X_tr.append(clips[:-n_test]); y_tr.append(np.full(len(clips) - n_test, i))
        w_tr.append(np.full(len(clips) - n_test, args.self_weight))
        X_te.append(clips[-n_test:]); y_te.append(np.full(n_test, i))
    X_tr, y_tr, w_tr = np.concatenate(X_tr), np.concatenate(y_tr), np.concatenate(w_tr)
    X_te, y_te = np.concatenate(X_te), np.concatenate(y_te)
    print(f"Training clips: {len(X_tr)} ({len(X_inc)} INCLUDE + {len(X_tr) - len(X_inc)} yours) | your test clips: {len(X_te)}")

    rf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    rf.fit(aggregate_features(X_tr), y_tr, sample_weight=w_tr)
    acc = accuracy_score(y_te, rf.predict(aggregate_features(X_te)))

    base_acc = None
    base_path, labels_path = MODELS_DIR / "static_model.pkl", MODELS_DIR / "labels.json"
    if base_path.exists() and labels_path.exists():
        base, classes = joblib.load(base_path), json.loads(labels_path.read_text())
        shared = [s for s in signs if s in classes]       # signs the original model knows
        mask = np.isin(y_te, [signs.index(s) for s in shared])
        if len(shared) >= 2 and mask.any():
            proba = base.predict_proba(aggregate_features(X_te[mask]))
            cols = [list(base.classes_).index(classes.index(s)) for s in shared]
            pred = np.array([signs.index(shared[i]) for i in np.argmax(proba[:, cols], axis=1)])
            base_acc = float(accuracy_score(y_te[mask], pred))

    joblib.dump({"model": rf, "classes": signs}, MODELS_DIR / "demo_model.pkl")
    (MODELS_DIR / "demo_metrics.json").write_text(json.dumps({
        "signs": signs, "n_test": int(len(y_te)), "baseline_accuracy": base_acc,
        "personalised_accuracy": float(acc),
        "note": ("Test clips were recorded by the same person in the same session as the training clips, "
                 "so the personalised score is optimistic and is not a generalisation claim. "
                 "The original-model score shows how the INCLUDE-trained model behaves on this webcam."),
    }, indent=2))
    print("\nOriginal INCLUDE-only model on your clips (signs it knows): "
          + (f"{base_acc * 100:.0f}%" if base_acc is not None else "n/a"))
    print(f"Personalised model on your held-out clips:              {acc * 100:.0f}%")
    print("Saved models/demo_model.pkl. Restart the app and choose Model = Personalised.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
