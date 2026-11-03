import pandas as pd
import streamlit as st

from database import db
from dashboard import ui


def render():
    ui.page_title("History", "Everything you have translated.", "Stored only on this computer.")
    search = st.text_input("Search", placeholder="Search sentences or words", label_visibility="collapsed")
    rows = db.recent_translations(limit=500, search=search)
    if not rows:
        ui.empty_state("No saved translations yet. Press Speak in the live translator to save one.")
        return
    df = pd.DataFrame(rows).rename(columns={"ts": "Time", "words": "Signs", "sentence": "Sentence",
                                            "mean_confidence": "Avg confidence", "model": "Model"}).drop(columns=["id"])
    df["Avg confidence"] = (df["Avg confidence"].fillna(0) * 100).round(0).astype(int).astype(str) + "%"
    st.dataframe(df, use_container_width=True, hide_index=True)
    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button("Export CSV", df.to_csv(index=False).encode("utf-8"), "signbridge_history.csv",
                       "text/csv", use_container_width=True)
    if c2.button("Clear history", use_container_width=True):
        db.clear_translations()
        st.rerun()
