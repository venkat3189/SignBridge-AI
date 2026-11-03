import streamlit as st

from database import db
from dashboard import ui
from src.config import SELECTED_SIGNS, ALL_SIGNS


def _go(page):
    st.session_state["nav"] = page


def render(metrics, predictor):
    ui.hero("SignBridge AI",
            "Communication without barriers.",
            "Real-time Indian Sign Language recognition that turns signs into text and speech. "
            "Everything runs on your computer, and your camera video never leaves it.")
    c1, c2, _ = st.columns([1, 1, 3])
    c1.button("Open live translator", type="primary", on_click=_go, args=("Live Translator",), use_container_width=True)
    c2.button("View model analytics", on_click=_go, args=("Analytics",), use_container_width=True)
    st.write("")

    best_val, best_note = "—", "Train the models to see results"
    if metrics and metrics.get("models"):
        name, m = max(metrics["models"].items(), key=lambda kv: kv[1]["accuracy"])
        best_val = f"{m['accuracy'] * 100:.1f}%"
        best_note = f"{ui.MODEL_NAMES.get(name, name)} · held-out test clips"
    n_models = len(metrics["models"]) if metrics else 0
    ui.stat_row([
        ("Vocabulary", f"{len(ALL_SIGNS)} signs", "Indian Sign Language"),
        ("Best test accuracy", best_val, best_note),
        ("Models trained", str(n_models), "Random Forest · SVM · GRU"),
        ("Saved translations", str(db.count("translations")), "Stored locally in SQLite"),
    ])

    st.write("")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Recent translations")
        rows = db.recent_translations(limit=6)
        if rows:
            for r in rows:
                ui.html(f'<div class="row-line"><b>{ui.esc(r["sentence"])}</b>'
                        f'<span style="float:right;color:#9A9A94;font-size:.8rem">{ui.esc(r["ts"])}</span></div>')
        else:
            ui.empty_state("Translate a few signs and they will appear here.")
    with right:
        st.markdown("#### System status")
        ready = predictor.ready
        items = [
            (ready, "Recognition models loaded" if ready else "Models not found. Run train_models.py"),
            (True, "Database ready (local SQLite)"),
            (True, "Processing on this device. No video is uploaded."),
        ]
        for ok, text in items:
            ui.html(f'<div class="row-line"><span class="dot {"on" if ok else "off"}"></span>{ui.esc(text)}</div>')
        if predictor.load_errors:
            st.warning("Some models failed to load: " + ", ".join(predictor.load_errors))

    st.write("")
    st.markdown("#### Supported vocabulary")
    for cat, signs in SELECTED_SIGNS.items():
        ui.html(f'<div class="eyebrow" style="margin-top:.8rem">{ui.esc(cat)}</div>{ui.chips(signs)}')
