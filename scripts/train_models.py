"""Trains and evaluates all models on the extracted landmarks.

Models:  Random Forest, SVM (video-level statistics)  and  GRU (30-frame sequences)
Extra:   hands-only vs hands+pose ablation (Random Forest)
Outputs: models/*.pkl, models/dynamic_model/gru.keras, models/metrics.json,
         models/labels.json, models/norm_stats.npz, data/processed/confusion_*.csv

Run:  python scripts/train_models.py            (add --skip-gru for a fast test)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import ALL_SIGNS, DYNAMIC_DIR, MODELS_DIR, PROC_DIR, SEQ_LEN, N_POSE_FEATURES
from src.data_preprocessing import load_landmark_dataset, split_by_group
from src.feature_extraction import aggregate_features

SEED = 42
np.random.seed(SEED)


def scores(y_true, y_pred, n_classes):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(n_classes), average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f)}


def time_single(fn, sample, reps=40):
    fn(sample)  # warm-up
    t = time.perf_counter()
    for _ in range(reps):
        fn(sample)
    return (time.perf_counter() - t) / reps * 1000


def augment(X, y, copies=4):
    """Noise + small time warp/shift on standardised sequences."""
    outs, ys = [X], [y]
    T = X.shape[1]
    for _ in range(copies):
        Xa = X.copy()
        for i in range(len(Xa)):
            f = np.random.uniform(0.9, 1.1)
            shift = np.random.uniform(-2, 2)
            idx = np.clip(np.round(np.arange(T) * f + shift), 0, T - 1).astype(int)
            Xa[i] = Xa[i][idx]
        Xa += np.random.normal(0, 0.05, Xa.shape).astype(np.float32)
        outs.append(Xa)
        ys.append(y)
    return np.concatenate(outs), np.concatenate(ys)


def train_gru(Xtr, ytr, Xva, yva, n_classes, epochs):
    import tensorflow as tf
    import keras
    tf.random.set_seed(SEED)
    model = keras.Sequential([
        keras.layers.Input((Xtr.shape[1], Xtr.shape[2])),
        keras.layers.GRU(128, return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.GRU(64),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    has_val = len(Xva) > 0
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss" if has_val else "loss",
                                        patience=15, restore_best_weights=True)]
    Xa, ya = augment(Xtr, ytr)
    model.fit(Xa, ya, validation_data=(Xva, yva) if has_val else None,
              epochs=epochs, batch_size=32, callbacks=cb, verbose="2")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gru", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    DYNAMIC_DIR.mkdir(parents=True, exist_ok=True)
    X, labels, numbers, names = load_landmark_dataset()
    classes = [s for s in ALL_SIGNS if s in set(labels)]
    cid = {c: i for i, c in enumerate(classes)}
    y = np.array([cid[l] for l in labels])
    n_classes = len(classes)
    subset = split_by_group(labels, numbers)
    tr, va, te = (subset == "train"), (subset == "val"), (subset == "test")
    print(f"{len(X)} videos, {n_classes} signs | train {tr.sum()}  val {va.sum()}  test {te.sum()}")
    print(f"Chance level on test: {100 / n_classes:.1f}%")
    missing = [s for s in ALL_SIGNS if s not in set(labels)]
    if missing:
        print("WARNING: no landmark data for:", missing)

    metrics = {
        "dataset": {"source": "INCLUDE (AI4Bharat), subset", "n_videos": int(len(X)),
                    "n_classes": n_classes, "train": int(tr.sum()), "val": int(va.sum()),
                    "test": int(te.sum()), "chance_level": 1 / n_classes},
        "split_note": ("Own split: whole recording runs (consecutive file numbers) are assigned to "
                       "one subset. Not verified to be signer-independent."),
        "models": {}, "ablation": {}, "classes": classes,
    }

    A = aggregate_features(X)                     # (N, 5*182)
    tv = tr | va                                  # classical models use train+val
    results = {}

    # ---- Random Forest ----
    t = time.time()
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(A[tv], y[tv])
    train_s = time.time() - t
    pred = rf.predict(A[te])
    m = scores(y[te], pred, n_classes)
    m["train_seconds"] = train_s
    m["inference_ms"] = time_single(lambda s: rf.predict_proba(aggregate_features(s)), X[te][0])
    results["random_forest"] = (m, pred)
    joblib.dump(rf, MODELS_DIR / "static_model.pkl")

    # ---- SVM ----
    t = time.time()
    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale",
                                              probability=True, random_state=SEED)).fit(A[tv], y[tv])
    train_s = time.time() - t
    pred = svm.predict(A[te])
    m = scores(y[te], pred, n_classes)
    m["train_seconds"] = train_s
    m["inference_ms"] = time_single(lambda s: svm.predict_proba(aggregate_features(s)), X[te][0])
    results["svm"] = (m, pred)
    joblib.dump(svm, MODELS_DIR / "svm_model.pkl")

    # ---- Ablation: hands only vs hands + body pose (Random Forest) ----
    A_hands = aggregate_features(X[:, :, N_POSE_FEATURES:])
    rf_h = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(A_hands[tv], y[tv])
    metrics["ablation"] = {
        "random_forest_hands_only": scores(y[te], rf_h.predict(A_hands[te]), n_classes),
        "random_forest_hands_plus_pose": results["random_forest"][0] | {},
    }

    # ---- GRU ----
    if not args.skip_gru:
        mean = X[tr].reshape(-1, X.shape[2]).mean(axis=0)
        std = X[tr].reshape(-1, X.shape[2]).std(axis=0) + 1e-6
        Xn = ((X - mean) / std).astype(np.float32)
        np.savez(MODELS_DIR / "norm_stats.npz", mean=mean, std=std)
        t = time.time()
        gru = train_gru(Xn[tr], y[tr], Xn[va], y[va], n_classes, args.epochs)
        train_s = time.time() - t
        proba = gru.predict(Xn[te], verbose="0")
        pred = proba.argmax(axis=1)
        m = scores(y[te], pred, n_classes)
        m["train_seconds"] = train_s
        m["inference_ms"] = time_single(lambda s: gru(s, training=False), Xn[te][:1])
        results["gru"] = (m, pred)
        gru.save(DYNAMIC_DIR / "gru.keras")

    for name, (m, pred) in results.items():
        metrics["models"][name] = m
        cm = confusion_matrix(y[te], pred, labels=range(n_classes))
        np.savetxt(PROC_DIR / f"confusion_{name}.csv", cm, fmt="%d", delimiter=",")
        metrics["models"][name]["confusion_matrix"] = cm.tolist()
    (MODELS_DIR / "labels.json").write_text(json.dumps(classes, indent=2))
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n=== TEST SET RESULTS (unseen recording runs) ===")
    print(f"{'model':16s} {'acc':>6s} {'prec':>6s} {'recall':>6s} {'F1':>6s} {'ms/pred':>8s}")
    for name, (m, _) in results.items():
        print(f"{name:16s} {m['accuracy']:6.3f} {m['precision_macro']:6.3f} "
              f"{m['recall_macro']:6.3f} {m['f1_macro']:6.3f} {m['inference_ms']:8.1f}")
    ab = metrics["ablation"]
    print(f"\nAblation (RF) hands-only acc {ab['random_forest_hands_only']['accuracy']:.3f} "
          f"vs hands+pose acc {ab['random_forest_hands_plus_pose']['accuracy']:.3f}")
    print("\nSaved: models/metrics.json, labels.json, static_model.pkl, svm_model.pkl",
          "" if args.skip_gru else ", dynamic_model/gru.keras")


if __name__ == "__main__":
    main()
