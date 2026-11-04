
import pandas as pd
import streamlit as st

from database import db
from dashboard import ui


def render():

    # --------------------------------------------------
    # HEADER
    # --------------------------------------------------

    ui.page_title(
        "COMMUNICATION ARCHIVE",
        "Every conversation. Preserved.",
        "Your recognised signs and translated sentences, stored securely on this device."
    )

    # --------------------------------------------------
    # SEARCH
    # --------------------------------------------------

    search_col, _ = st.columns([4, 1])

    with search_col:
        search = st.text_input(
            "Search conversations",
            placeholder="Search sentences, signs or keywords...",
            key="history_search"
        )

    rows = db.recent_translations(
        limit=500,
        search=search
    )

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    total = len(rows)

    if rows:
        confidences = [
            float(r.get("mean_confidence") or 0)
            for r in rows
        ]
        average = sum(confidences) / len(confidences)
        unique_models = len({
            r.get("model")
            for r in rows
            if r.get("model")
        })
    else:
        average = 0
        unique_models = 0

    ui.stat_row([
        ("Records found", f"{total:02d}", "Matching conversations"),
        ("Average confidence", f"{average:.0%}", "Across displayed records"),
        ("Models used", f"{unique_models:02d}", "Recognition models")
    ])

    st.write("")

    # --------------------------------------------------
    # EMPTY STATE
    # --------------------------------------------------

    if not rows:
        ui.empty_state(
            "No conversations found. Your translated sentences will appear here."
        )
        return

    # --------------------------------------------------
    # COMMUNICATION LOG
    # --------------------------------------------------

    ui.html(
        '<div class="section-heading">Communication log</div>'
        '<div class="section-caption">'
        'Your latest saved translations.</div>'
    )

    for index, r in enumerate(rows):

        sentence = ui.esc(r.get("sentence", ""))
        timestamp = ui.esc(r.get("ts", ""))
        signs = r.get("words") or []
        model = ui.esc(r.get("model") or "Not specified")

        confidence = float(
            r.get("mean_confidence") or 0
        )

        if isinstance(signs, str):
            signs_text = signs
        else:
            signs_text = " · ".join(str(word) for word in signs)

        signs_text = ui.esc(signs_text)

        confidence_width = max(
            0,
            min(confidence * 100, 100)
        )

        ui.html(
            f'''
            <div class="panel" style="
                padding:1.1rem 1.3rem;
                margin-bottom:.7rem;
                border-radius:14px;
                background:linear-gradient(120deg,#181818,#101010);
            ">
                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                    gap:12px;
                    flex-wrap:wrap;
                ">
                    <div style="
                        color:#999;
                        font-size:.68rem;
                        font-weight:700;
                        letter-spacing:.14em;
                    ">
                        TRANSLATION {index + 1:03d}
                    </div>

                    <div style="
                        color:#999;
                        font-size:.73rem;
                    ">
                        {timestamp}
                    </div>
                </div>

                <div style="
                    color:#FFFFFF;
                    font-family:Manrope,sans-serif;
                    font-size:1.12rem;
                    font-weight:700;
                    margin:.8rem 0;
                    line-height:1.5;
                ">
                    {sentence}
                </div>

                <div style="
                    color:#999;
                    font-size:.78rem;
                    line-height:1.6;
                ">
                    SIGNS: {signs_text or "Not available"}
                </div>

                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                    gap:10px;
                    margin-top:1rem;
                    flex-wrap:wrap;
                ">
                    <div style="color:#999;font-size:.73rem">
                        MODEL: {model}
                    </div>

                    <div style="color:#FFFFFF;font-size:.75rem;font-weight:700">
                        {confidence:.0%} CONFIDENCE
                    </div>
                </div>

                <div style="
                    height:3px;
                    background:#333;
                    border-radius:10px;
                    margin-top:.6rem;
                    overflow:hidden;
                ">
                    <div style="
                        height:100%;
                        width:{confidence_width:.1f}%;
                        background:#FFFFFF;
                        border-radius:10px;
                    "></div>
                </div>
            </div>
            '''
        )

    # --------------------------------------------------
    # EXPORT
    # --------------------------------------------------

    st.write("")

    ui.html(
        '<div class="section-heading">Export your records</div>'
        '<div class="section-caption">'
        'Download your conversation history for future reference.'
        '</div>'
    )

    df = pd.DataFrame(rows).rename(
        columns={
            "ts": "Time",
            "words": "Signs",
            "sentence": "Sentence",
            "mean_confidence": "Avg confidence",
            "model": "Model"
        }
    )

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df["Avg confidence"] = (
    pd.to_numeric(
        df["Avg confidence"],
        errors="coerce"
    )
    .fillna(0.0)
    .apply(lambda value: f"{value:.0%}")
)

    if "Signs" in df.columns:
        df["Signs"] = df["Signs"].apply(
            lambda value: (
                " · ".join(map(str, value))
                if isinstance(value, (list, tuple))
                else str(value or "")
            )
        )

    c1, c2 = st.columns([1, 1])

    with c1:
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "signbridge_history.csv",
            "text/csv",
            use_container_width=True
        )

    with c2:
        if st.button(
            "Clear history",
            use_container_width=True
        ):
            st.session_state["confirm_clear_history"] = True

    if st.session_state.get("confirm_clear_history"):
        st.warning(
            "This will permanently clear all saved translations."
        )

        yes, no = st.columns(2)

        with yes:
            if st.button(
                "Confirm deletion",
                type="primary",
                use_container_width=True
            ):
                db.clear_translations()
                st.session_state["confirm_clear_history"] = False
                st.rerun()

        with no:
            if st.button(
                "Cancel",
                use_container_width=True
            ):
                st.session_state["confirm_clear_history"] = False
                st.rerun()
