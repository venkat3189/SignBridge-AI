import streamlit as st

from dashboard import ui

AUTHOR = "Your Name"   # <- change this to your name before you submit


def render():
    ui.page_title("About", "SignBridge AI",
                  "An Indian Sign Language to text and speech translator, built for the Neural Strom hackathon (FT-04).")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### How it works")
        for i, (t, d) in enumerate([
            ("Capture", "OpenCV reads the webcam. Frames stay in memory and are never saved or uploaded."),
            ("Track", "MediaPipe finds both hands and the upper body, giving 182 numbers per frame."),
            ("Segment", "A sign starts when a hand is raised and ends when the hands drop."),
            ("Classify", "Random Forest, SVM and a GRU network predict one of 30 signs with a confidence score."),
            ("Filter", "Low-confidence predictions are rejected and repeated detections are ignored."),
            ("Build", "Rule-based grammar groups signs into punctuated sentences without adding words."),
            ("Speak", "pyttsx3 reads the sentence aloud using the voices installed on your PC."),
        ], 1):
            ui.html(f'<div class="row-line"><b>{i}. {t}</b><br><span style="color:#6B6B66">{ui.esc(d)}</span></div>')
    with c2:
        st.markdown("#### Technology")
        ui.html(ui.chips(["Python 3.11", "OpenCV", "MediaPipe", "scikit-learn", "TensorFlow / Keras",
                          "Streamlit", "Plotly", "SQLite", "pyttsx3", "faster-whisper"], blue=True))
        st.markdown("#### Honest limitations")
        for t in [
            "Recognises only the 30 trained signs, one sign at a time. It is not full ISL translation.",
            "The sentence builder orders and punctuates signs. It does not translate ISL grammar.",
            "Training used the INCLUDE dataset, signed by a small group of signers. Accuracy on other people, lighting and cameras is untested unless stated in the README.",
            "The test split is our own, grouped by recording run. It is not verified signer-independent.",
            "Speech recognition needs an internet connection once to download its model.",
        ]:
            ui.html(f'<div class="row-line">{ui.esc(t)}</div>')
        st.markdown("#### Dataset and credits")
        st.markdown("Signs come from the **INCLUDE** dataset (AI4Bharat and IIT Madras, Sridhar et al., "
                    "ACM Multimedia 2020), shared on Zenodo under CC BY 4.0.")
        st.markdown(f"Built by **{AUTHOR}**.")
