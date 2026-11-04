
import streamlit as st

from database import db
from dashboard import ui
from src.config import SELECTED_SIGNS, ALL_SIGNS, CUSTOM_SIGNS


def _go(page):
    st.session_state["nav"] = page


def _section(title, subtitle=""):
    ui.html(
        f'<div class="section-heading">{ui.esc(title)}</div>'
        f'<div class="section-caption">{ui.esc(subtitle)}</div>'
    )


def render(metrics, predictor):

    # --------------------------------------------------
    # HERO
    # --------------------------------------------------

        # --------------------------------------------------
    # HERO
    # --------------------------------------------------

    ui.html("""
        <div class="hero">
            <div class="eyebrow">
                INTELLIGENT COMMUNICATION PLATFORM
            </div>

            <div class="title" style="
                font-size: clamp(3rem, 5.2vw, 5rem);
                line-height: 1.05;
                letter-spacing: -0.055em;
                max-width: 100%;
            ">
                Every Gesture. Every Voice.<br>
                One Connection.
            </div>

            <div class="sub">
                Sign language recognition. Speech intelligence.
                One seamless experience.
            </div>
        </div>
    """)

    c1, c2, _ = st.columns([1.35, 1.25, 2.4])

    with c1:
        st.button(
            "Open Translator  ↗",
            type="primary",
            on_click=_go,
            args=("Live Translator",),
            use_container_width=True
        )

    with c2:
        st.button(
            "Explore Analytics",
            on_click=_go,
            args=("Analytics",),
            use_container_width=True
        )

    st.write("")

    # --------------------------------------------------
    # DYNAMIC STATISTICS
    # --------------------------------------------------

    best_val = "—"
    best_note = "Train models to view results"

    if metrics and metrics.get("models"):
        name, model = max(
            metrics["models"].items(),
            key=lambda item: item[1]["accuracy"]
        )

        best_val = f'{model["accuracy"] * 100:.1f}%'
        best_note = (
            f'{ui.MODEL_NAMES.get(name, name)} · Test accuracy'
        )

    n_models = len(metrics["models"]) if metrics else 0
    saved = db.count("translations")

    ui.stat_row([
        (
            "Sign vocabulary",
            str(len(ALL_SIGNS)),
            "Available recognition signs"
        ),
        (
            "Best model accuracy",
            best_val,
            best_note
        ),
        (
            "Trained models",
            f"{n_models:02d}",
            "RF · SVM · GRU"
        ),
        (
            "Saved translations",
            f"{saved:02d}",
            "Stored in local database"
        ),
    ])

    st.write("")
    st.write("")

    # --------------------------------------------------
    # EXPLORE PLATFORM
    # --------------------------------------------------

    _section(
        "Explore the platform",
        "Intelligent tools for effortless communication."
    )

    a, b, c = st.columns(3, gap="medium")

    features = [
        (
            a,
            "01",
            "Live Translator",
            "Turn hand gestures into meaningful text in real time.",
            "Live Translator"
        ),
        (
            b,
            "02",
            "Two-Way Communication",
            "Connect speech and sign language in one workspace.",
            "Two-way communication"
        ),
        (
            c,
            "03",
            "Model Analytics",
            "Explore recognition performance and model results.",
            "Analytics"
        )
    ]

    for col, number, title, description, destination in features:
        with col:
            ui.html(
                f'<div class="panel" style="min-height:185px">'
                f'<div class="eyebrow">MODULE {number}</div>'
                f'<div style="font-family:Manrope,sans-serif;'
                f'font-size:1.1rem;font-weight:750;color:white;'
                f'margin-bottom:.55rem">{ui.esc(title)}</div>'
                f'<div style="color:#999;font-size:.82rem;'
                f'line-height:1.65">{ui.esc(description)}</div>'
                f'</div>'
            )

            st.button(
                f"Open {title}  ↗",
                key=f"home_{number}",
                on_click=_go,
                args=(destination,),
                use_container_width=True
            )

    st.write("")
    st.write("")

    # --------------------------------------------------
    # RECENT TRANSLATIONS AND SYSTEM STATUS
    # --------------------------------------------------

    left, right = st.columns([1.5, 1], gap="large")

    with left:
        _section(
            "Recent translations",
            "Your latest recognised sentences."
        )

        rows = db.recent_translations(limit=6)

        if rows:
            for r in rows:
                ui.html(
                    f'<div class="row-line">'
                    f'<b>{ui.esc(r["sentence"])}</b>'
                    f'<span style="float:right;color:#858585;'
                    f'font-size:.75rem">'
                    f'{ui.esc(r["ts"])}</span></div>'
                )
        else:
            ui.empty_state(
                "Your translated sentences will appear here."
            )

    with right:
        _section(
            "System status",
            "Application readiness and privacy."
        )

        ready = predictor.ready

        items = [
            (
                ready,
                "Recognition models loaded"
                if ready else "Recognition models unavailable"
            ),
            (
                True,
                "Local database connected"
            ),
            (
                True,
                "Video processing stays on device"
            )
        ]

        for ok, message in items:
            ui.html(
                f'<div class="row-line">'
                f'<span class="dot {"on" if ok else "off"}"></span>'
                f'{ui.esc(message)}</div>'
            )

        if predictor.load_errors:
            st.warning(
                "Some models failed to load: "
                + ", ".join(predictor.load_errors)
            )

    st.write("")
    st.write("")

    # --------------------------------------------------
    # SUPPORTED VOCABULARY
    # --------------------------------------------------

    _section(
        "Recognition vocabulary",
        "Explore the signs currently supported by SignBridge AI."
    )

    for category, signs in SELECTED_SIGNS.items():
        ui.html(
            f'<div class="eyebrow" '
            f'style="margin-top:1.1rem">'
            f'{ui.esc(category)}</div>'
            f'{ui.chips(signs)}'
        )

    ui.html(
        '<div class="eyebrow" style="margin-top:1.1rem">'
        'PERSONAL CALIBRATION WORDS</div>'
        f'{ui.chips(CUSTOM_SIGNS)}'
    )
