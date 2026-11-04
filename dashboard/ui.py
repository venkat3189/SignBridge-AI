
"""Shared styling and UI helpers — SignBridge AI Premium Monochrome."""

import html as _html
import json

import streamlit as st

from src.config import MODELS_DIR

MODEL_NAMES = {
    "random_forest": "Random Forest",
    "svm": "Support Vector Machine",
    "gru": "GRU"
}

ACCENT = "#FFFFFF"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --black: #080808;
    --surface: #111111;
    --surface-2: #171717;
    --surface-3: #202020;
    --white: #FFFFFF;
    --soft-white: #EDEDED;
    --muted: #929292;
    --line: #292929;
    --radius: 18px;
}

html, body, .stApp, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--black);
    color: var(--white);
}

#MainMenu, footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1380px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
}

h1, h2, h3, h4 {
    font-family: 'Manrope', sans-serif;
    color: var(--white);
    font-weight: 700;
    letter-spacing: -0.045em;
}

p, label, span, li {
    color: inherit;
}

[data-testid="stSidebar"] {
    background: #0D0D0D;
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 7px;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    width: 100%;
    padding: .72rem .85rem;
    border-radius: 11px;
    border: 1px solid transparent;
    transition: all .2s ease;
    color: #A8A8A8;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #191919;
    color: white;
}

[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
    display: none;
}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: #F2F2F2;
    color: #080808;
    border-color: #F2F2F2;
    font-weight: 700;
}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
    color: #080808;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: var(--radius);
    border-color: var(--line);
}

[data-testid="stImage"] img {
    border-radius: var(--radius);
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 2rem;
}

.brand .mark {
    width: 39px;
    height: 39px;
    border-radius: 11px;
    background: white;
    color: black;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Manrope', sans-serif;
    font-size: 1.2rem;
    font-weight: 800;
}

.brand-name {
    color: white;
    font-family: 'Manrope', sans-serif;
    font-size: 1.02rem;
    font-weight: 800;
    letter-spacing: .08em;
}

.brand small {
    display: block;
    color: #858585;
    font-size: .62rem;
    font-weight: 600;
    letter-spacing: .17em;
    margin-top: 2px;
}

.eyebrow {
    font-size: .68rem;
    letter-spacing: .19em;
    text-transform: uppercase;
    color: #999999;
    font-weight: 700;
    margin-bottom: .75rem;
}

.hero {
    padding: 1.1rem 0 2rem;
}

.hero .title {
    font-family: 'Manrope', sans-serif;
    font-size: clamp(2.7rem, 4.6vw, 4.3rem);
    line-height: 1.08;
    font-weight: 800;
    letter-spacing: -.055em;
    color: white;
    max-width: 1150px;
    width: 100%;
}

.hero .sub {
    font-size: 1.02rem;
    color: #A0A0A0;
    max-width: 620px;
    margin-top: 1.1rem;
    line-height: 1.75;
}

.section-heading {
    color: white;
    font-family: 'Manrope', sans-serif;
    font-size: 1.08rem;
    font-weight: 750;
    letter-spacing: -.025em;
    margin-bottom: .25rem;
}

.section-caption {
    color: #858585;
    font-size: .82rem;
    margin-bottom: 1rem;
}

.stat {
    background: linear-gradient(145deg, #171717, #101010);
    border: 1px solid #292929;
    border-radius: 17px;
    padding: 1.35rem;
    min-height: 148px;
    transition: all .25s ease;
}

.stat:hover {
    transform: translateY(-4px);
    border-color: #777777;
    background: #1A1A1A;
}

.stat .l {
    font-size: .67rem;
    letter-spacing: .13em;
    text-transform: uppercase;
    color: #A0A0A0;
    font-weight: 700;
}

.stat .v {
    font-family: 'Manrope', sans-serif;
    font-size: clamp(1.8rem, 3vw, 2.5rem);
    font-weight: 800;
    letter-spacing: -.06em;
    margin: .65rem 0 .2rem;
    color: white;
}

.stat .n {
    font-size: .76rem;
    color: #858585;
    line-height: 1.5;
}

.panel {
    background: linear-gradient(145deg, #171717, #101010);
    border: 1px solid #292929;
    border-radius: 19px;
    padding: 1.45rem;
    box-shadow: 0 12px 40px rgba(0,0,0,.16);
}

.big-sign {
    font-size: 2.7rem;
    font-weight: 800;
    letter-spacing: -.05em;
    min-height: 3.4rem;
    color: white;
    line-height: 1.15;
}

.conf {
    height: 5px;
    background: #303030;
    border-radius: 99px;
    overflow: hidden;
    margin-top: .5rem;
}

.conf-bar {
    height: 100%;
    background: white;
    border-radius: 99px;
}

.conf-text {
    font-size: .8rem;
    color: #999999;
    margin: .4rem 0 1rem;
}

.status {
    display: inline-flex;
    align-items: center;
    gap: .5rem;
    font-size: .78rem;
    padding: .42rem .8rem;
    border-radius: 99px;
    background: #242424;
    color: #E5E5E5;
    border: 1px solid #363636;
    margin-bottom: 1rem;
}

.status.ok, .status.live, .status.warn {
    background: #242424;
    color: white;
    border-color: #505050;
}

.sentence {
    font-size: 1.35rem;
    font-weight: 650;
    line-height: 1.5;
    letter-spacing: -.025em;
    min-height: 2rem;
    color: white;
}

.camera-off {
    aspect-ratio: 4/3;
    border-radius: 18px;
    border: 1px dashed #505050;
    background: #141414;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: #929292;
    gap: .5rem;
}

.camera-off b {
    color: white;
    font-size: 1rem;
}

.dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
    background: #777777;
    margin-right: .55rem;
}

.dot.on {
    background: #FFFFFF;
    box-shadow: 0 0 9px rgba(255,255,255,.5);
}

.dot.off {
    background: #666666;
}

.row-line {
    padding: .8rem .1rem;
    border-bottom: 1px solid #292929;
    font-size: .88rem;
    color: #E0E0E0;
    line-height: 1.5;
}

.bubble {
    padding: .8rem 1rem;
    border-radius: 14px;
    margin: .4rem 0;
    max-width: 85%;
    font-size: .92rem;
    line-height: 1.5;
}

.bubble.a {
    background: #F1F1F1;
    color: #080808;
    margin-left: auto;
    border-bottom-right-radius: 4px;
}

.bubble.b {
    background: #222222;
    color: #FFFFFF;
    border-bottom-left-radius: 4px;
}

.bubble small {
    display: block;
    opacity: .58;
    font-size: .68rem;
    margin-top: .3rem;
}

.chip {
    display: inline-block;
    padding: .38rem .78rem;
    margin: 0 .35rem .45rem 0;
    border-radius: 7px;
    background: #202020;
    color: #E8E8E8;
    font-size: .78rem;
    border: 1px solid #363636;
}

.chip.blue {
    background: #F1F1F1;
    color: #080808;
    border-color: #FFFFFF;
    font-weight: 700;
}

.stButton > button {
    border-radius: 10px;
    border: 1px solid #414141;
    background: #171717;
    color: white;
    font-weight: 650;
    padding: .65rem 1.1rem;
    min-height: 43px;
    transition: all .2s ease;
}

.stButton > button:hover {
    border-color: white;
    background: #292929;
    color: white;
    transform: translateY(-2px);
}

.stButton > button[kind="primary"] {
    background: white;
    color: #080808;
    border-color: white;
    font-weight: 750;
}

.stButton > button[kind="primary"]:hover {
    background: #DCDCDC;
    border-color: #DCDCDC;
    color: black;
}

.stTextInput input, .stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input {
    background: #171717 !important;
    color: white !important;
    border-color: #383838 !important;
    border-radius: 10px !important;
}

.stTextInput label, .stTextArea label,
.stSelectbox label, .stSlider label {
    color: #D0D0D0 !important;
}

.stProgress > div > div > div > div {
    background: white;
}

[data-testid="stMetric"] {
    background: #171717;
    border: 1px solid #303030;
    border-radius: 15px;
    padding: 1rem;
}

[data-testid="stMetricLabel"] {
    color: #A0A0A0;
}

[data-testid="stMetricValue"] {
    color: white;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: #292929;
}

@media(max-width: 768px) {
    .block-container {
        padding: 1.2rem 1rem 3rem;
    }

    .hero .title {
        font-size: 2.65rem;
    }

    .hero .sub {
        font-size: .92rem;
    }

    .stat {
        padding: 1rem;
        min-height: 125px;
    }
}


/* ==========================================
   SIGNBRIDGE AI — GLOBAL PREMIUM OVERRIDES
========================================== */

/* Sidebar navigation: force readable text */
[data-testid="stSidebar"] [role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 9px !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    background: transparent !important;
    color: #D0D0D0 !important;
    border: 1px solid transparent !important;
    border-radius: 12px !important;
    padding: 13px 15px !important;
    min-height: 46px;
    transition: all .2s ease;
}

[data-testid="stSidebar"] [role="radiogroup"] label p,
[data-testid="stSidebar"] [role="radiogroup"] label span,
[data-testid="stSidebar"] [role="radiogroup"] label div {
    color: inherit !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #202020 !important;
    color: #FFFFFF !important;
    border-color: #383838 !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: #FFFFFF !important;
    color: #080808 !important;
    border-color: #FFFFFF !important;
    font-weight: 700 !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p,
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) span,
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) div {
    color: #080808 !important;
}

/* Sidebar branding */
[data-testid="stSidebar"] {
    background: #0C0C0C !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #D0D0D0;
}

.brand {
    padding: 10px 4px 22px;
    border-bottom: 1px solid #292929;
    margin-bottom: 20px;
}

/* Main content */
.stApp {
    background: #080808 !important;
    color: #FFFFFF !important;
}

[data-testid="stMain"] {
    background: #080808 !important;
}

.block-container {
    max-width: 1380px !important;
}

/* Headings and descriptions */
.stMarkdown p,
.stCaption,
[data-testid="stCaptionContainer"] {
    color: #B0B0B0;
}

h1, h2, h3, h4 {
    color: #FFFFFF !important;
}

/* Premium cards */
.panel, .stat {
    transition: border-color .2s ease, transform .2s ease;
}

.panel:hover {
    border-color: #505050;
}

/* Text inputs */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    background: #171717 !important;
    color: #FFFFFF !important;
    border: 1px solid #393939 !important;
    caret-color: #FFFFFF;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #858585 !important;
}

/* Select boxes */
[data-baseweb="select"] > div {
    background: #171717 !important;
    border-color: #393939 !important;
    color: #FFFFFF !important;
}

[data-baseweb="select"] * {
    color: #FFFFFF !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #111111;
    border: 1px solid #303030;
    border-radius: 13px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #FFFFFF !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #303030;
    border-radius: 12px;
    overflow: hidden;
}

/* Alerts */
[data-testid="stAlert"] {
    background: #171717;
    border: 1px solid #383838;
    color: #FFFFFF;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    min-height: 44px;
    border-radius: 11px;
    font-weight: 650;
}

/* Focus visibility */
button:focus-visible,
input:focus-visible,
textarea:focus-visible {
    outline: 2px solid #FFFFFF !important;
    outline-offset: 3px;
}

/* Mobile layout */
@media(max-width: 768px) {
    .block-container {
        padding: 1.2rem .9rem 3rem !important;
    }

    .hero .title {
        font-size: 2.5rem !important;
    }

    .stat {
        min-height: 115px;
    }
}

</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def html(s):
    """Render HTML without leading indentation."""
    st.markdown(
        "\n".join(line.strip() for line in s.strip().splitlines()),
        unsafe_allow_html=True
    )


def esc(s):
    return _html.escape(str(s))


def hero(eyebrow, title, subtitle):
    html(
        f'<div class="hero">'
        f'<div class="eyebrow">{esc(eyebrow)}</div>'
        f'<div class="title">{esc(title)}</div>'
        f'<div class="sub">{esc(subtitle)}</div>'
        f'</div>'
    )


def page_title(eyebrow, title, subtitle=""):
    sub = (
        f'<div class="sub" style="font-size:.95rem;margin-top:.6rem">'
        f'{esc(subtitle)}</div>'
        if subtitle else ""
    )

    html(
        f'<div class="hero" style="padding:.4rem 0 1.4rem">'
        f'<div class="eyebrow">{esc(eyebrow)}</div>'
        f'<div class="title" style="font-size:clamp(2.2rem,4vw,3.4rem)">'
        f'{esc(title)}</div>{sub}</div>'
    )


def stat_card(label, value, note=""):
    return (
        f'<div class="stat">'
        f'<div class="l">{esc(label)}</div>'
        f'<div class="v">{esc(value)}</div>'
        f'<div class="n">{esc(note)}</div>'
        f'</div>'
    )


def stat_row(items):
    cols = st.columns(len(items), gap="medium")

    for col, (label, value, note) in zip(cols, items):
        with col:
            html(stat_card(label, value, note))


def chips(items, blue=False):
    cls = "chip blue" if blue else "chip"

    return "".join(
        f'<span class="{cls}">{esc(i)}</span>'
        for i in items
    )


def empty_state(message):
    html(
        f'<div class="camera-off" '
        f'style="aspect-ratio:auto;padding:2.2rem">'
        f'<b>Nothing here yet</b>'
        f'<span>{esc(message)}</span></div>'
    )


@st.cache_data
def _load(path_str, mtime):
    with open(path_str, encoding="utf-8") as file:
        return json.loads(file.read())


def load_metrics():
    path = MODELS_DIR / "metrics.json"

    if not path.exists():
        return None

    try:
        return _load(str(path), path.stat().st_mtime)
    except Exception:
        return None
