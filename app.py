"""SignBridge AI - run with:  streamlit run app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

st.set_page_config(page_title="SignBridge AI", page_icon="🤟", layout="wide")

from database import db
from dashboard import about, communication, history, home, model_insights, translator, ui
from src.gesture_prediction import GesturePredictor

PAGES = ["Overview", "Live Translator", "Two-Way", "Analytics", "History", "About"]


@st.cache_resource
def get_predictor():
    return GesturePredictor()

db.init_db()
ui.inject_css()
metrics = ui.load_metrics()
db.sync_model_runs(metrics)
predictor = get_predictor()

st.session_state.setdefault("nav", PAGES[0])
with st.sidebar:
    ui.html('<div class="brand"><div class="mark">SB</div><div>SignBridge<small>AI Platform</small></div></div>')
    page = st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
    st.write("")
    ui.html('<div style="font-size:.78rem;color:#8A8A84;line-height:1.5;margin-top:2rem">'
            'Runs locally.<br>Your camera video never leaves this computer.</div>')

if page == "Overview":
    home.render(metrics, predictor)
elif page == "Live Translator":
    translator.render(predictor)
elif page == "Two-Way":
    communication.render()
elif page == "Analytics":
    model_insights.render(metrics)
elif page == "History":
    history.render()
else:
    about.render()
