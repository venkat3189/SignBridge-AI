
import streamlit as st

from database import db
from dashboard import ui
from src import speech_output
from src.sentence_builder import build_sentence


LANGS = {
    "Auto-detect": None,
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te"
}


def render():
    ui.page_title(
        "Two-way communication",
        "Let both sides be heard.",
        "Hearing person to text with speech recognition, "
        "and signed sentences to speech."
    )

    left, right = st.columns(2, gap="large")

    # --------------------------------------------------
    # SPEECH TO TEXT
    # --------------------------------------------------

    with left:
        st.markdown("#### Speech to text")

        st.caption(
            "Record your voice using your browser microphone. "
            "The recorded audio is converted into text using Whisper."
        )

        lang_name = st.selectbox(
            "Spoken language",
            list(LANGS),
            key="stt_lang"
        )

        audio_file = st.audio_input(
    "Record your speech",
    key="browser_microphone"
)

        if st.button(
            "Transcribe recording",
            type="primary",
            use_container_width=True,
            disabled=audio_file is None
        ):
            try:
                from src import speech_input

                status = st.empty()

                status.info("⏳ Processing your recording...")

                audio = speech_input.decode_audio(audio_file)

                with st.spinner("Transcribing speech with Whisper..."):
                    text, detected = speech_input.transcribe(
                        audio,
                        LANGS[lang_name]
                    )

                if text:
                    db.add_conversation(
                        "speech_to_text",
                        text,
                        detected
                    )

                    st.session_state["last_stt"] = text

                    status.success(
                        f"✓ Speech recognized ({detected})"
                    )

                else:
                    status.warning(
                        "No clear speech detected. "
                        "Please record again."
                    )

            except Exception as exc:
                st.error(
                    f"Speech recognition failed: {exc}"
                )

        if st.session_state.get("last_stt"):
            ui.html(
                f'<div class="panel">'
                f'<div class="eyebrow">Heard</div>'
                f'<div class="sentence">'
                f'{ui.esc(st.session_state["last_stt"])}'
                f'</div></div>'
            )

    # --------------------------------------------------
    # SIGN TO SPEECH
    # --------------------------------------------------

    with right:
        st.markdown("#### Sign to speech")

        st.caption(
            "Speak the sentence built in the live translator, "
            "or type something to say aloud."
        )

        default = build_sentence(
            st.session_state.get("words", [])
        )

        text = st.text_area(
            "Sentence",
            value=default,
            height=120,
            key=f"tts_text_{default}"
        )

        voices = speech_output.list_voices()
        voice_id = None

        if voices:
            names = [name for _, name in voices]

            selected_voice = st.selectbox(
                "Voice",
                names,
                key="comm_voice"
            )

            voice_id = voices[names.index(selected_voice)][0]

        if st.button(
            "Speak",
            use_container_width=True
        ):
            ok, msg = speech_output.speak(
                text,
                voice_id
            )

            if ok:
                db.add_conversation(
                    "sign_to_speech",
                    text,
                    ""
                )

                st.success("Speech played successfully.")

            else:
                st.error(msg)

    # --------------------------------------------------
    # CONVERSATION HISTORY
    # --------------------------------------------------

    st.write("")

    st.markdown("#### Conversation history")

    rows = list(
        reversed(db.recent_conversations(30))
    )

    if not rows:
        ui.empty_state(
            "Spoken and signed messages will appear here."
        )

    for r in rows:
        cls = (
            "a"
            if r["direction"] == "sign_to_speech"
            else "b"
        )

        who = (
            "Signed → spoken"
            if cls == "a"
            else "Heard → text"
        )

        ui.html(
            f'<div style="display:flex">'
            f'<div class="bubble {cls}">'
            f'{ui.esc(r["text"])}'
            f'<small>{who} · {ui.esc(r["ts"])}</small>'
            f'</div></div>'
        )
