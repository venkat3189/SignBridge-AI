
"""SignBridge AI - run with: streamlit run app.py"""

import sys
from pathlib import Path

# Add project root to Python path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SignBridge AI",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# PROJECT IMPORTS
# --------------------------------------------------

from database import db
from dashboard import (
    about,
    communication,
    history,
    home,
    model_insights,
    translator,
    ui
)
from src.gesture_prediction import GesturePredictor


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

PAGES = [
    "Overview",
    "Live Translator",
    "Two-Way",
    "Analytics",
    "History",
    "About"
]


# --------------------------------------------------
# LOAD RECOGNITION MODEL
# --------------------------------------------------

@st.cache_resource
def get_predictor():
    return GesturePredictor()


# --------------------------------------------------
# INITIALIZE APPLICATION
# --------------------------------------------------

db.init_db()

ui.inject_css()

metrics = ui.load_metrics()

db.sync_model_runs(metrics)

predictor = get_predictor()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.session_state.setdefault("nav", PAGES[0])

with st.sidebar:

    # Brand
    ui.html("""
        <div class="brand">
            <div class="mark">SB</div>
            <div>
                <div class="brand-name">SIGNBRIDGE</div>
                <small>INTELLIGENT COMMUNICATION</small>
            </div>
        </div>
    """)

    st.write("")

    # Navigation
    page = st.radio(
        "Navigation",
        PAGES,
        key="nav",
        label_visibility="collapsed"
    )

    st.write("")

    # Privacy information
    ui.html("""
        <div style="
            margin-top:3rem;
            padding:1rem .5rem;
            border-top:1px solid #292929;
        ">
            <div style="
                color:#FFFFFF;
                font-size:.75rem;
                font-weight:700;
                letter-spacing:.12em;
            ">
                PRIVATE BY DESIGN
            </div>

            <div style="
                color:#999999;
                font-size:.78rem;
                line-height:1.7;
                margin-top:.5rem;
            ">
                Local processing.<br>
                Your camera video stays on this device.
            </div>
        </div>
    """)


# --------------------------------------------------
# MAIN CONTENT - PAGE ROUTING
# IMPORTANT: Keep this outside st.sidebar
# --------------------------------------------------

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

elif page == "About":

    about.render()
