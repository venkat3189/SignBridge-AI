
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import ui
from src.config import MODELS_DIR, SELECTED_SIGNS


BLACK = "#080808"
WHITE = "#FFFFFF"
GREY = "#A0A0A0"
DARK_GREY = "#252525"
MID_GREY = "#777777"


def _style(fig, height=360):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=15, r=20, t=45, b=20),
        font=dict(
            family="DM Sans, sans-serif",
            color=WHITE,
            size=12
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            y=1.12,
            font=dict(color="#CCCCCC")
        ),
        title_font=dict(
            family="Manrope, sans-serif",
            color=WHITE
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor=DARK_GREY,
            tickfont=dict(color="#BBBBBB")
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor=DARK_GREY,
            tickfont=dict(color="#BBBBBB")
        ),
        hoverlabel=dict(
            bgcolor="#202020",
            font_color=WHITE,
            bordercolor="#555555"
        )
    )

    return fig


def _heading(title, subtitle=""):
    ui.html(
        f'<div class="section-heading">{ui.esc(title)}</div>'
        f'<div class="section-caption">{ui.esc(subtitle)}</div>'
    )


def _panel_start(title):
    ui.html(
        f'<div class="panel">'
        f'<div class="section-heading">{ui.esc(title)}</div>'
    )


def render(metrics):

    # --------------------------------------------------
    # PAGE HEADER
    # --------------------------------------------------

    ui.page_title(
        "MODEL INTELLIGENCE",
        "Performance, measured.",
        "A transparent view of recognition accuracy, speed and reliability."
    )

    if not metrics or not metrics.get("models"):
        ui.empty_state(
            "Train your models to unlock performance analytics."
        )
        st.code("python scripts/train_models.py", language="bash")
        return

    ds = metrics["dataset"]
    models = metrics["models"]

    names = {
        key: ui.MODEL_NAMES.get(key, key)
        for key in models
    }

    metric_keys = [
        ("accuracy", "Accuracy"),
        ("precision_macro", "Precision"),
        ("recall_macro", "Recall"),
        ("f1_macro", "F1-score")
    ]

    # --------------------------------------------------
    # OVERVIEW METRICS
    # --------------------------------------------------

    best_key = max(
        models,
        key=lambda key: models[key]["accuracy"]
    )

    best_model = models[best_key]

    ui.stat_row([
        (
            "Dataset size",
            str(ds["n_videos"]),
            f'{ds["n_classes"]} recognised signs'
        ),
        (
            "Best accuracy",
            f'{best_model["accuracy"] * 100:.1f}%',
            names[best_key]
        ),
        (
            "Models evaluated",
            f"{len(models):02d}",
            "Recognition algorithms"
        ),
        (
            "Random baseline",
            f'{ds["chance_level"] * 100:.1f}%',
            "Chance-level accuracy"
        )
    ])

    st.write("")

    if metrics.get("split_note"):
        st.caption(metrics["split_note"])

    # --------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------

    _heading(
        "Model comparison",
        "Compare recognition quality across all trained models."
    )

    fig = go.Figure()

    monochrome = [
        "#FFFFFF",
        "#BDBDBD",
        "#858585",
        "#505050"
    ]

    for (key, label), colour in zip(metric_keys, monochrome):
        fig.add_bar(
            name=label,
            x=[names[k] for k in models],
            y=[models[k][key] for k in models],
            marker_color=colour,
            marker_line=dict(
                color="#FFFFFF",
                width=0.4
            ),
            text=[
                f'{models[k][key] * 100:.1f}%'
                for k in models
            ],
            textposition="outside",
            textfont=dict(color=WHITE)
        )

    fig.update_layout(
        barmode="group",
        yaxis=dict(
            range=[0, 1.12],
            tickformat=".0%",
            title="Performance"
        ),
        bargap=0.25,
        bargroupgap=0.08
    )

    st.plotly_chart(
        _style(fig, 410),
        use_container_width=True
    )

    st.write("")

    # --------------------------------------------------
    # LATENCY AND ABLATION
    # --------------------------------------------------

    c1, c2 = st.columns(2, gap="large")

    with c1:
        _heading(
            "Recognition latency",
            "Time required for one prediction on CPU."
        )

        latency_values = [
            models[k]["inference_ms"]
            for k in models
        ]

        fig = go.Figure(
            go.Bar(
                x=[names[k] for k in models],
                y=latency_values,
                marker_color=[
                    "#FFFFFF",
                    "#A0A0A0",
                    "#555555"
                ][:len(models)],
                text=[
                    f"{value:.1f} ms"
                    for value in latency_values
                ],
                textposition="outside",
                textfont=dict(color=WHITE)
            )
        )

        fig.update_layout(
            yaxis_title="Milliseconds",
            showlegend=False
        )

        st.plotly_chart(
            _style(fig, 350),
            use_container_width=True
        )

    with c2:
        _heading(
            "Feature ablation",
            "Evaluate the contribution of body-pose information."
        )

        ab = metrics.get("ablation", {})

        if ab:
            labels = [
                "Hands only",
                "Hands + pose"
            ]

            fig = go.Figure()

            for key, label, colour in [
                ("accuracy", "Accuracy", WHITE),
                ("f1_macro", "F1-score", MID_GREY)
            ]:
                vals = [
                    ab["random_forest_hands_only"][key],
                    ab["random_forest_hands_plus_pose"][key]
                ]

                fig.add_bar(
                    name=label,
                    x=labels,
                    y=vals,
                    marker_color=colour,
                    text=[
                        f"{value * 100:.1f}%"
                        for value in vals
                    ],
                    textposition="outside",
                    textfont=dict(color=WHITE)
                )

            fig.update_layout(
                barmode="group",
                yaxis=dict(
                    range=[0, 1.12],
                    tickformat=".0%"
                )
            )

            st.plotly_chart(
                _style(fig, 350),
                use_container_width=True
            )

            st.caption(
                "Random Forest · Same evaluation split."
            )
        else:
            ui.empty_state(
                "No feature ablation results available."
            )

    st.write("")

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    _heading(
        "Confusion matrix",
        "Inspect correct predictions and classification errors."
    )

    choice = st.selectbox(
        "Select recognition model",
        list(models),
        format_func=lambda key: names[key],
        key="cm_model"
    )

    classes = metrics["classes"]

    cm = np.array(
        models[choice]["confusion_matrix"]
    )

    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            colorscale=[
                [0, "#111111"],
                [0.35, "#555555"],
                [0.7, "#AAAAAA"],
                [1, "#FFFFFF"]
            ],
            colorbar=dict(
                title="Clips",
                tickfont=dict(color=WHITE),
                title_font=dict(color=WHITE)
            ),
            hovertemplate=(
                "True: %{y}<br>"
                "Predicted: %{x}<br>"
                "Clips: %{z}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        xaxis_title="Predicted label",
        yaxis_title="Actual label",
        yaxis_autorange="reversed"
    )

    st.plotly_chart(
        _style(fig, 620),
        use_container_width=True
    )

    st.write("")

    # --------------------------------------------------
    # PER SIGN RECALL
    # --------------------------------------------------

    _heading(
        "Per-sign recognition",
        "Identify signs that need additional training."
    )

    support = cm.sum(axis=1)

    recall = np.divide(
        np.diag(cm),
        support,
        out=np.zeros(len(classes)),
        where=support > 0
    )

    df = pd.DataFrame({
        "Sign": classes,
        "Recall": recall,
        "Test clips": support
    }).sort_values("Recall")

    fig = go.Figure(
        go.Bar(
            x=df["Recall"],
            y=df["Sign"],
            orientation="h",
            marker_color=[
                "#555555" if value < 0.5 else
                "#AAAAAA" if value < 0.8 else
                "#FFFFFF"
                for value in df["Recall"]
            ],
            text=[
                f"{value * 100:.1f}%"
                for value in df["Recall"]
            ],
            textposition="outside",
            textfont=dict(color=WHITE)
        )
    )

    fig.update_layout(
        xaxis=dict(
            range=[0, 1.12],
            tickformat=".0%",
            title="Recall"
        ),
        showlegend=False
    )

    st.plotly_chart(
        _style(fig, max(420, len(classes) * 32)),
        use_container_width=True
    )

    st.caption(
        "Low recall indicates signs that may benefit from more training examples."
    )

    # --------------------------------------------------
    # PERSONAL CALIBRATION
    # --------------------------------------------------

    demo_path = MODELS_DIR / "demo_metrics.json"

    if demo_path.exists():

        d = json.loads(
            demo_path.read_text(encoding="utf-8")
        )

        st.write("")

        _heading(
            "Personal calibration",
            "Compare your original model with your personalised model."
        )

        baseline = (
            "N/A"
            if d["baseline_accuracy"] is None
            else f'{d["baseline_accuracy"] * 100:.1f}%'
        )

        ui.stat_row([
            (
                "Original model",
                baseline,
                f'{d["n_test"]} webcam test clips'
            ),
            (
                "Personalised model",
                f'{d["personalised_accuracy"] * 100:.1f}%',
                f'{len(d["signs"])} personalised signs'
            )
        ])

        st.caption(d["note"])

    # --------------------------------------------------
    # RAW DATA
    # --------------------------------------------------

    st.write("")

    with st.expander("View detailed model metrics"):

        table = pd.DataFrame([
            {
                "Model": names[k],
                **{
                    label: models[k][key]
                    for key, label in metric_keys
                },
                "Latency (ms)": models[k]["inference_ms"],
                "Training time (s)": models[k]["train_seconds"]
            }
            for k in models
        ])

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------
    # VOCABULARY
    # --------------------------------------------------

    st.write("")

    _heading(
        "Supported vocabulary",
        "Signs currently included in the recognition dataset."
    )

    for category, signs in SELECTED_SIGNS.items():
        ui.html(
            f'<div class="eyebrow" '
            f'style="margin-top:1rem">'
            f'{ui.esc(category)}</div>'
            f'{ui.chips(signs)}'
        )
