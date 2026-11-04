
import streamlit as st

from dashboard import ui


AUTHOR = "Your Name"  # Replace with your name before submission


def render():

    # --------------------------------------------------
    # PAGE HEADER
    # --------------------------------------------------

    ui.page_title(
        "ABOUT SIGNBRIDGE",
        "Technology with a human purpose.",
        "An intelligent Indian Sign Language recognition platform designed to make everyday communication more accessible."
    )

    # --------------------------------------------------
    # INTRODUCTION
    # --------------------------------------------------

    ui.html("""
    <div class="panel" style="
        background:linear-gradient(125deg,#191919,#090909);
        border:1px solid #343434;
        padding:2rem;
        margin-bottom:1.5rem;
    ">
        <div class="eyebrow" style="color:#AFAFAF">
            THE PROJECT
        </div>

        <div style="
            font-family:Manrope,sans-serif;
            font-size:2rem;
            font-weight:800;
            color:#FFFFFF;
            letter-spacing:-.045em;
            line-height:1.15;
            max-width:650px;
        ">
            Every gesture deserves to be understood.
        </div>

        <div style="
            color:#B8B8B8;
            font-size:.95rem;
            line-height:1.8;
            margin-top:1rem;
            max-width:650px;
        ">
            SignBridge AI combines computer vision, machine learning
            and speech intelligence to convert recognised signs into
            readable text and spoken communication.
        </div>

        <div style="
            display:flex;
            flex-wrap:wrap;
            gap:9px;
            margin-top:1.5rem;
        ">
            <span class="chip">INDIAN SIGN LANGUAGE</span>
            <span class="chip">COMPUTER VISION</span>
            <span class="chip">SPEECH INTELLIGENCE</span>
        </div>
    </div>
    """)

    # --------------------------------------------------
    # HOW IT WORKS
    # --------------------------------------------------

    ui.html("""
    <div class="section-heading">Inside the intelligence</div>
    <div class="section-caption">
        Seven stages transform a hand gesture into communication.
    </div>
    """)

    stages = [
        (
            "01",
            "Capture",
            "OpenCV reads webcam frames. Video remains in memory and is not uploaded."
        ),
        (
            "02",
            "Track",
            "MediaPipe tracks both hands and upper-body landmarks, producing 182 features per frame."
        ),
        (
            "03",
            "Segment",
            "A sign begins when a hand is raised and ends when the hands are lowered."
        ),
        (
            "04",
            "Classify",
            "Random Forest, SVM and GRU models predict one of the 30 trained signs."
        ),
        (
            "05",
            "Filter",
            "Low-confidence predictions are rejected and repeated detections are filtered."
        ),
        (
            "06",
            "Build",
            "Rule-based sentence building organises recognised signs into punctuated sentences."
        ),
        (
            "07",
            "Speak",
            "pyttsx3 converts the generated sentence into speech using installed voices."
        )
    ]

    for start in range(0, len(stages), 2):

        columns = st.columns(2, gap="medium")

        for column, item in zip(
            columns,
            stages[start:start + 2]
        ):

            number, title, description = item

            with column:
                ui.html(
                    f"""
                    <div class="panel" style="
                        background:linear-gradient(145deg,#191919,#101010);
                        border:1px solid #303030;
                        min-height:155px;
                        margin-bottom:1rem;
                        padding:1.3rem;
                    ">
                        <div style="
                            display:flex;
                            justify-content:space-between;
                            align-items:center;
                        ">
                            <span style="
                                color:#FFFFFF;
                                font-size:1.3rem;
                                font-weight:800;
                            ">{number}</span>

                            <span style="
                                color:#666;
                                font-size:.68rem;
                                letter-spacing:.15em;
                                font-weight:700;
                            ">PROCESS</span>
                        </div>

                        <div style="
                            color:#FFFFFF;
                            font-size:1.1rem;
                            font-weight:750;
                            margin-top:1rem;
                        ">{title}</div>

                        <div style="
                            color:#A0A0A0;
                            font-size:.83rem;
                            line-height:1.7;
                            margin-top:.45rem;
                        ">{ui.esc(description)}</div>
                    </div>
                    """
                )

    # --------------------------------------------------
    # TECHNOLOGY STACK
    # --------------------------------------------------

    st.write("")

    ui.html("""
    <div class="section-heading">Built with precision.</div>
    <div class="section-caption">
        The technologies powering the SignBridge AI ecosystem.
    </div>
    """)

    technologies = [
        ("01", "Python 3.11", "Core programming"),
        ("02", "OpenCV", "Video processing"),
        ("03", "MediaPipe", "Landmark tracking"),
        ("04", "scikit-learn", "Machine learning"),
        ("05", "TensorFlow / Keras", "Deep learning"),
        ("06", "Streamlit", "Application interface"),
        ("07", "Plotly", "Model visualisation"),
        ("08", "SQLite", "Local data storage"),
        ("09", "pyttsx3", "Text to speech"),
        ("10", "Faster-Whisper", "Speech recognition")
    ]

    for start in range(0, len(technologies), 2):

        columns = st.columns(2, gap="medium")

        for column, item in zip(
            columns,
            technologies[start:start + 2]
        ):

            number, name, purpose = item

            with column:
                ui.html(
                    f"""
                    <div class="panel" style="
                        background:#141414;
                        border:1px solid #303030;
                        border-radius:13px;
                        padding:1rem 1.2rem;
                        margin-bottom:.7rem;
                    ">
                        <div style="
                            color:#777;
                            font-size:.7rem;
                            font-weight:700;
                            letter-spacing:.12em;
                        ">{number}</div>

                        <div style="
                            color:#FFFFFF;
                            font-size:1rem;
                            font-weight:700;
                            margin-top:.4rem;
                        ">{ui.esc(name)}</div>

                        <div style="
                            color:#929292;
                            font-size:.78rem;
                            margin-top:.25rem;
                        ">{ui.esc(purpose)}</div>
                    </div>
                    """
                )

    # --------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------

    st.write("")

    ui.html("""
    <div class="section-heading">Built with transparency.</div>
    <div class="section-caption">
        Current capabilities and known limitations.
    </div>
    """)

    limitations = [
        (
            "Limited vocabulary",
            "Currently recognises only 30 trained signs, one sign at a time."
        ),
        (
            "Sentence construction",
            "The sentence builder orders and punctuates recognised signs. It does not perform full ISL grammar translation."
        ),
        (
            "Dataset diversity",
            "Training uses the INCLUDE dataset, which contains recordings from a limited group of signers."
        ),
        (
            "Evaluation scope",
            "The test split is grouped by recording run and is not verified to be signer-independent."
        ),
        (
            "Speech recognition",
            "Faster-Whisper requires an initial model download. Subsequent transcription can run locally."
        )
    ]

    for title, description in limitations:

        ui.html(
            f"""
            <div style="
                display:flex;
                gap:1rem;
                padding:1rem 0;
                border-bottom:1px solid #303030;
            ">
                <div style="
                    color:#FFFFFF;
                    font-size:1.1rem;
                    font-weight:800;
                ">—</div>

                <div>
                    <div style="
                        color:#FFFFFF;
                        font-weight:700;
                        font-size:.92rem;
                    ">{ui.esc(title)}</div>

                    <div style="
                        color:#999;
                        font-size:.82rem;
                        line-height:1.7;
                        margin-top:.3rem;
                    ">{ui.esc(description)}</div>
                </div>
            </div>
            """
        )

    # --------------------------------------------------
    # DATASET AND CREDITS
    # --------------------------------------------------

    st.write("")

    ui.html("""
    <div class="panel" style="
        background:linear-gradient(135deg,#191919,#0B0B0B);
        border:1px solid #383838;
        padding:1.6rem;
    ">
        <div class="eyebrow" style="color:#999">
            DATASET & ACKNOWLEDGEMENTS
        </div>

        <div style="
            color:#FFFFFF;
            font-size:1.25rem;
            font-weight:800;
        ">
            Research-powered recognition.
        </div>

        <div style="
            color:#AAAAAA;
            font-size:.85rem;
            line-height:1.8;
            margin-top:.7rem;
        ">
            The recognised signs are based on the INCLUDE dataset
            from AI4Bharat and IIT Madras, described by Sridhar et al.,
            ACM Multimedia 2020.
        </div>

        <div style="
            color:#FFFFFF;
            font-size:.8rem;
            margin-top:1rem;
            font-weight:650;
        ">
            Dataset licence: CC BY 4.0
        </div>
    </div>
    """)

    # --------------------------------------------------
    # PROJECT FOOTER
    # --------------------------------------------------

    st.write("")

    ui.html(
        f"""
        <div style="
            text-align:center;
            padding:2rem 1rem;
            border-top:1px solid #303030;
        ">
            <div style="
                color:#FFFFFF;
                font-size:1.15rem;
                font-weight:800;
                letter-spacing:-.03em;
            ">
                SIGNBRIDGE AI
            </div>

            <div style="
                color:#888;
                font-size:.8rem;
                margin-top:.5rem;
            ">
                Designed to connect. Built to communicate.
            </div>

            <div style="
                color:#666;
                font-size:.75rem;
                margin-top:1rem;
            ">
                Developed by {ui.esc(AUTHOR)}
            </div>
        </div>
        """
    )
