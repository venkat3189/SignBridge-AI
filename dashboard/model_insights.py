import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import ui
from src.config import MODELS_DIR, SELECTED_SIGNS

BLUE, INK, GREY = "#2F5BFF", "#0B0B0C", "#B9BDC7"


def _style(fig, height=360):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=30, b=10),
                      font=dict(family="Inter, system-ui, sans-serif", color="#111"),
                      legend=dict(orientation="h", y=1.12), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def render(metrics):
    ui.page_title("Analytics", "How well does it actually work?",
                  "Every number on this page comes from evaluating the saved models on held-out test clips.")
    if not metrics or not metrics.get("models"):
        ui.empty_state("Run python scripts/train_models.py to generate real results.")
        return

    ds, models = metrics["dataset"], metrics["models"]
    names = {k: ui.MODEL_NAMES.get(k, k) for k in models}
    ui.stat_row([
        ("Videos used", str(ds["n_videos"]), f'{ds["n_classes"]} signs'),
        ("Train / val / test", f'{ds["train"]} / {ds["val"]} / {ds["test"]}', "Whole recording runs per subset"),
        ("Chance level", f'{ds["chance_level"] * 100:.1f}%', "Random guessing baseline"),
    ])
    st.caption(metrics.get("split_note", ""))

    st.markdown("#### Model comparison")
    metric_keys = [("accuracy", "Accuracy"), ("precision_macro", "Precision"),
                   ("recall_macro", "Recall"), ("f1_macro", "F1-score")]
    fig = go.Figure()
    for (key, label), colour in zip(metric_keys, [INK, BLUE, "#7C9BFF", GREY]):
        fig.add_bar(name=label, x=[names[k] for k in models], y=[models[k][key] for k in models],
                    marker_color=colour, text=[f"{models[k][key] * 100:.1f}%" for k in models], textposition="outside")
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1.1], tickformat=".0%"))
    st.plotly_chart(_style(fig), use_container_width=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### Recognition latency")
        fig = go.Figure(go.Bar(x=[names[k] for k in models], y=[models[k]["inference_ms"] for k in models],
                               marker_color=BLUE, text=[f'{models[k]["inference_ms"]:.1f} ms' for k in models],
                               textposition="outside"))
        fig.update_layout(yaxis_title="ms per prediction (CPU)")
        st.plotly_chart(_style(fig, 320), use_container_width=True)
    with c2:
        st.markdown("#### Hands only vs hands + body pose")
        ab = metrics.get("ablation", {})
        if ab:
            labels = ["Hands only", "Hands + pose"]
            fig = go.Figure()
            for key, label, colour in (("accuracy", "Accuracy", INK), ("f1_macro", "F1-score", BLUE)):
                vals = [ab["random_forest_hands_only"][key], ab["random_forest_hands_plus_pose"][key]]
                fig.add_bar(name=label, x=labels, y=vals, marker_color=colour,
                            text=[f"{v * 100:.1f}%" for v in vals], textposition="outside")
            fig.update_layout(barmode="group", yaxis=dict(range=[0, 1.1], tickformat=".0%"))
            st.plotly_chart(_style(fig, 320), use_container_width=True)
            st.caption("Random Forest, same split. Compares whether body pose adds information.")

    st.markdown("#### Confusion matrix")
    choice = st.selectbox("Model", list(models), format_func=lambda k: names[k], key="cm_model")
    classes = metrics["classes"]
    cm = np.array(models[choice]["confusion_matrix"])
    fig = go.Figure(go.Heatmap(z=cm, x=classes, y=classes, colorscale=[[0, "#FFFFFF"], [1, BLUE]],
                               hovertemplate="True: %{y}<br>Predicted: %{x}<br>Clips: %{z}<extra></extra>"))
    fig.update_layout(xaxis_title="Predicted", yaxis_title="True", yaxis_autorange="reversed")
    st.plotly_chart(_style(fig, 640), use_container_width=True)

    st.markdown("#### Per-sign recall")
    support = cm.sum(axis=1)
    recall = np.divide(np.diag(cm), support, out=np.zeros(len(classes)), where=support > 0)
    df = pd.DataFrame({"Sign": classes, "Recall": recall, "Test clips": support}).sort_values("Recall")
    fig = go.Figure(go.Bar(x=df["Recall"], y=df["Sign"], orientation="h", marker_color=BLUE))
    fig.update_layout(xaxis=dict(range=[0, 1], tickformat=".0%"))
    st.plotly_chart(_style(fig, 720), use_container_width=True)
    st.caption("Signs with few test clips give noisy estimates. Use this to see which signs the model confuses.")

    with st.expander("Raw numbers"):
        table = pd.DataFrame([{"Model": names[k], **{lbl: models[k][key] for key, lbl in metric_keys},
                               "Latency (ms)": models[k]["inference_ms"],
                               "Training time (s)": models[k]["train_seconds"]} for k in models])
        st.dataframe(table, use_container_width=True, hide_index=True)
    demo_path = MODELS_DIR / "demo_metrics.json"
    if demo_path.exists():
        import json
        d = json.loads(demo_path.read_text(encoding="utf-8"))
        st.markdown("#### Personal calibration experiment")
        ui.stat_row([
            ("Original model on my webcam", f'{d["baseline_accuracy"] * 100:.0f}%', f'{d["n_test"]} clips recorded by the user'),
            ("Personalised model", f'{d["personalised_accuracy"] * 100:.0f}%', f'{len(d["signs"])} signs: ' + ", ".join(d["signs"])),
        ])
        st.caption(d["note"])
    st.markdown("#### Supported vocabulary")
    for cat, signs in SELECTED_SIGNS.items():
        ui.html(f'<div class="eyebrow" style="margin-top:.6rem">{ui.esc(cat)}</div>{ui.chips(signs)}')
