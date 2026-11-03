"""Loads the trained models and turns a (30, 182) sequence into a prediction."""
import json
import time

import joblib
import numpy as np

from src.config import MODELS_DIR, DYNAMIC_DIR
from src.feature_extraction import aggregate_features


class GesturePredictor:
    def __init__(self):
        labels_path = MODELS_DIR / "labels.json"
        self.classes = json.loads(labels_path.read_text()) if labels_path.exists() else []
        self.models = {}
        self.load_errors = {}
        for key, name, fname in (("Random Forest", "rf", "static_model.pkl"), ("SVM", "svm", "svm_model.pkl")):
            path = MODELS_DIR / fname
            if path.exists():
                try:
                    self.models[key] = joblib.load(path)
                except Exception as exc:
                    self.load_errors[key] = str(exc)
        demo_path = MODELS_DIR / "demo_model.pkl"
        self._demo_classes = []
        if demo_path.exists():
            try:
                obj = joblib.load(demo_path)
                self.models["Personalised"] = obj["model"]
                self._demo_classes = obj["classes"]
            except Exception as exc:
                self.load_errors["Personalised"] = str(exc)
        gru_path = DYNAMIC_DIR / "gru.keras"
        stats_path = MODELS_DIR / "norm_stats.npz"
        if gru_path.exists() and stats_path.exists():
            try:
                from tensorflow import keras
                self.models["GRU"] = keras.models.load_model(gru_path)
                stats = np.load(stats_path)
                self._mean, self._std = stats["mean"], stats["std"]
            except Exception as exc:
                self.load_errors["GRU"] = str(exc)

    @property
    def ready(self):
        return bool(self.models) and bool(self.classes)

    def available(self):
        base = [n for n in ("Random Forest", "GRU", "SVM") if n in self.models]
        names = (["Personalised"] if "Personalised" in self.models else []) + base
        if len(base) >= 2:
            names.append("Ensemble")
        return names

    def _proba_one(self, name, seq, agg):
        n = len(self.classes)
        if name == "GRU":
            x = ((seq - self._mean) / self._std).astype(np.float32)[None]
            return np.asarray(self.models["GRU"](x, training=False))[0]
        model = self.models[name]
        out = np.zeros(n)
        if name == "Personalised":
            for ci, p in zip(model.classes_, model.predict_proba(agg)[0]):
                sign = self._demo_classes[ci]
                if sign in self.classes:
                    out[self.classes.index(sign)] = p
            return out
        out[model.classes_] = model.predict_proba(agg)[0]
        return out

    def predict(self, seq, model="Ensemble"):
        t = time.perf_counter()
        seq = np.asarray(seq, dtype=np.float32)
        agg = aggregate_features(seq)
        if model == "Ensemble":
            probs = np.mean([self._proba_one(n, seq, agg) for n in self.models if n != "Personalised"], axis=0)
        else:
            probs = self._proba_one(model, seq, agg)
        order = np.argsort(probs)[::-1]
        return {
            "label": self.classes[order[0]],
            "confidence": float(probs[order[0]]),
            "margin": float(probs[order[0]] - probs[order[1]]),
            "top3": [(self.classes[i], float(probs[i])) for i in order[:3]],
            "ms": (time.perf_counter() - t) * 1000,
            "model": model,
        }
