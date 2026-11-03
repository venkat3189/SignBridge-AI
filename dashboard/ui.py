"""Shared styling and small UI helpers (Luxury Minimal look)."""
import html as _html
import json

import streamlit as st

from src.config import MODELS_DIR

MODEL_NAMES = {"random_forest": "Random Forest", "svm": "SVM", "gru": "GRU"}
ACCENT = "#2F5BFF"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.stApp { background: #FFFFFF; color: #111111; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1240px; }
h1, h2, h3, h4 { letter-spacing: -0.025em; font-weight: 650; color: #0B0B0C; }
[data-testid="stSidebar"] { background: #FAFAF8; border-right: 1px solid #EDEDE8; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding: .55rem .8rem; border-radius: 10px; width: 100%; transition: background .15s; border: 1px solid transparent; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #F0F0EC; }
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: #FFFFFF; border-color: #E6E6E0; box-shadow: 0 1px 2px rgba(0,0,0,.04); font-weight: 600; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 18px; }
[data-testid="stImage"] img { border-radius: 20px; }
.brand { display:flex; align-items:center; gap:.6rem; font-weight:700; font-size:1.15rem; letter-spacing:-.02em; margin-bottom:1.4rem; }
.brand .mark { width:28px; height:28px; border-radius:8px; background:#0B0B0C; color:#fff; display:flex; align-items:center; justify-content:center; font-size:.85rem; }
.brand small { display:block; font-weight:500; font-size:.7rem; color:#8A8A84; letter-spacing:.08em; text-transform:uppercase; }
.eyebrow { font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:#6B6B66; font-weight:600; margin-bottom:.55rem; }
.hero { padding: 1.4rem 0 1.6rem; }
.hero .title { font-size: 3.2rem; line-height: 1.04; font-weight: 700; letter-spacing: -.035em; color: #0B0B0C; max-width: 18ch; }
.hero .sub { font-size: 1.12rem; color: #5A5A55; max-width: 58ch; margin-top: 1rem; line-height: 1.6; }
.stat { background:#fff; border:1px solid #ECECE8; border-radius:18px; padding:1.25rem 1.35rem; box-shadow:0 1px 2px rgba(0,0,0,.03), 0 10px 28px rgba(0,0,0,.03); }
.stat .l { font-size:.74rem; letter-spacing:.1em; text-transform:uppercase; color:#7A7A74; font-weight:600; }
.stat .v { font-size:2.15rem; font-weight:700; letter-spacing:-.035em; margin:.3rem 0 .1rem; color:#0B0B0C; }
.stat .n { font-size:.82rem; color:#8A8A84; }
.chip { display:inline-block; padding:.3rem .75rem; margin:0 .35rem .45rem 0; border-radius:999px; background:#F4F4F1; color:#222; font-size:.85rem; border:1px solid #E8E8E3; }
.chip.blue { background:#EEF2FF; color:#2F5BFF; border-color:#DCE4FF; font-weight:550; }
.panel { background:#fff; border:1px solid #ECECE8; border-radius:20px; padding:1.5rem 1.6rem; box-shadow:0 1px 2px rgba(0,0,0,.03), 0 14px 36px rgba(0,0,0,.04); }
.big-sign { font-size:2.7rem; font-weight:700; letter-spacing:-.035em; min-height:3.4rem; color:#0B0B0C; line-height:1.15; }
.conf { height:6px; background:#EEEEEA; border-radius:99px; overflow:hidden; margin-top:.4rem; }
.conf-bar { height:100%; background:#2F5BFF; border-radius:99px; }
.conf-text { font-size:.8rem; color:#7A7A74; margin:.4rem 0 1rem; }
.status { display:inline-flex; align-items:center; gap:.5rem; font-size:.85rem; padding:.35rem .85rem; border-radius:99px; background:#F4F4F1; color:#444; margin-bottom:1.1rem; }
.status.ok { background:#E8F7EE; color:#14753C; }
.status.warn { background:#FFF1F0; color:#B42318; }
.status.live { background:#EEF2FF; color:#2F5BFF; }
.sentence { font-size:1.4rem; font-weight:560; line-height:1.4; letter-spacing:-.01em; min-height:2rem; color:#0B0B0C; }
.camera-off { aspect-ratio:4/3; border-radius:20px; border:1px dashed #D5D5CF; background:#FAFAF8; display:flex; flex-direction:column; align-items:center; justify-content:center; color:#8A8A84; gap:.4rem; }
.camera-off b { color:#333; font-size:1.05rem; }
.dot { width:8px; height:8px; border-radius:50%; display:inline-block; background:#C9C9C2; margin-right:.5rem; }
.dot.on { background:#22A559; }
.dot.off { background:#E5484D; }
.row-line { padding:.45rem 0; border-bottom:1px solid #F0F0EC; font-size:.92rem; color:#333; }
.bubble { padding:.7rem 1rem; border-radius:16px; margin:.35rem 0; max-width:85%; font-size:.95rem; line-height:1.45; }
.bubble.a { background:#0B0B0C; color:#fff; margin-left:auto; border-bottom-right-radius:4px; }
.bubble.b { background:#F4F4F1; color:#111; border-bottom-left-radius:4px; }
.bubble small { display:block; opacity:.55; font-size:.7rem; margin-top:.2rem; }
.stButton > button { border-radius:12px; border:1px solid #E2E2DC; background:#fff; color:#111; font-weight:550; padding:.55rem 1rem; transition:all .15s; }
.stButton > button:hover { border-color:#111; color:#111; transform:translateY(-1px); }
.stButton > button[kind="primary"] { background:#0B0B0C; color:#fff; border-color:#0B0B0C; }
.stButton > button[kind="primary"]:hover { background:#2F5BFF; border-color:#2F5BFF; color:#fff; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def html(s):
    """Render HTML without leading indentation (Markdown would treat it as code)."""
    st.markdown("\n".join(line.strip() for line in s.strip().splitlines()), unsafe_allow_html=True)


def esc(s):
    return _html.escape(str(s))


def hero(eyebrow, title, subtitle):
    html(f'<div class="hero"><div class="eyebrow">{esc(eyebrow)}</div>'
         f'<div class="title">{esc(title)}</div><div class="sub">{esc(subtitle)}</div></div>')


def page_title(eyebrow, title, subtitle=""):
    sub = f'<div class="sub" style="font-size:1rem;margin-top:.5rem">{esc(subtitle)}</div>' if subtitle else ""
    html(f'<div class="hero" style="padding:.4rem 0 1.2rem"><div class="eyebrow">{esc(eyebrow)}</div>'
         f'<div class="title" style="font-size:2.3rem">{esc(title)}</div>{sub}</div>')


def stat_card(label, value, note=""):
    return (f'<div class="stat"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div>'
            f'<div class="n">{esc(note)}</div></div>')


def stat_row(items):
    cols = st.columns(len(items), gap="medium")
    for col, (label, value, note) in zip(cols, items):
        with col:
            html(stat_card(label, value, note))


def chips(items, blue=False):
    cls = "chip blue" if blue else "chip"
    return "".join(f'<span class="{cls}">{esc(i)}</span>' for i in items)


def empty_state(message):
    html(f'<div class="camera-off" style="aspect-ratio:auto;padding:2.2rem"><b>Nothing here yet</b>{esc(message)}</div>')


@st.cache_data
def _load(path_str, mtime):
    return json.loads(open(path_str, encoding="utf-8").read())


def load_metrics():
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        return None
    try:
        return _load(str(path), path.stat().st_mtime)
    except Exception:
        return None
