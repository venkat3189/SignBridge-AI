import time

import cv2
import numpy as np
import streamlit as st

from database import db
from dashboard import ui
from src import ai_polish, speech_output
from src.config import RECORDABLE_SIGNS
from src.confidence_filter import SignSegmenter, accept
from src.sentence_builder import build_sentence

STATUS_TEXT = {
    "idle": ("Waiting for hands", ""),
    "signing": ("Reading sign…", "live"),
    "ok": ("Sign recognised", "ok"),
    "reject": ("Gesture not recognised. Please try again.", "warn"),
}


@st.cache_resource
def get_extractor():
    from src.hand_tracking import LandmarkExtractor
    return LandmarkExtractor(static_image_mode=False, hands_complexity=1, pose_complexity=0)


@st.cache_data
def get_voices():
    return speech_output.list_voices()


def _init_state():
    defaults = {"words": [], "confs": [], "last": None, "last_time": 0.0, "last_spoken": "",
                "last_word": "", "last_word_time": 0.0}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _undo():
    if st.session_state.words:
        st.session_state.words.pop()
        if st.session_state.confs:
            st.session_state.confs.pop()


def _clear():
    st.session_state.words, st.session_state.confs = [], []
    st.session_state.last_spoken = ""


def _add_manual():
    st.session_state.words.append(st.session_state.manual_word)
    st.session_state.confs.append(1.0)


def panel_html(status_key, seg_frames=0, thr=0.20):
    last = st.session_state.last
    label = last["label"] if last and last["accepted"] else ("—" if not last else "?")
    conf = last["confidence"] if last else 0.0
    ms = f' · {last["ms"]:.0f} ms' if last else ""
    text, cls = STATUS_TEXT[status_key]
    if status_key == "signing":
        text = f"Reading sign… ({seg_frames} frames)"
    words = st.session_state.words
    sentence = build_sentence(words, question=st.session_state.get("as_question", False)) or "—"
    top3 = ""
    if last:
        top3 = " · ".join(f"{ui.esc(n)} {p * 100:.0f}%" for n, p in last["top3"])
        if not last["accepted"]:
            top3 += f'<br><span style="color:#B42318">Below threshold ({thr * 100:.0f}%), so it was not added.</span>'
    conf_line = f"{conf * 100:.0f}% confidence{ms}"
    pol = st.session_state.get("polished")
    ai_html = ""
    if pol and pol["key"] == (tuple(words), bool(st.session_state.get("as_question", False))):
        if pol["used_ai"]:
            ai_html = (f'<div class="eyebrow" style="margin-top:1.1rem">AI-polished</div>'
                       f'<div class="sentence" style="color:#2F5BFF">{ui.esc(pol["text"])}</div>')
        elif pol["message"]:
            ai_html = f'<div class="conf-text" style="margin-top:.8rem">{ui.esc(pol["message"])}</div>'
    chips = ui.chips(words, blue=True) if words else '<span style="color:#9A9A94">No words yet</span>'
    return (f'<div class="panel"><div class="eyebrow">Recognised sign</div>'
            f'<div class="big-sign">{ui.esc(label)}</div>'
            f'<div class="conf"><div class="conf-bar" style="width:{conf * 100:.0f}%"></div></div>'
            f'<div class="conf-text">{conf_line}'
            f'{"<br>Top 3: " + top3 if top3 else ""}</div>'
            f'<div class="status {cls}">{ui.esc(text)}</div>'
            f'<div class="eyebrow">Translation buffer</div><div style="margin-bottom:1rem">{chips}</div>'
            f'<div class="eyebrow">Sentence</div><div class="sentence">{ui.esc(sentence)}</div>{ai_html}</div>')


def _draw(frame, info):
    from mediapipe.python.solutions import drawing_utils, hands, pose

    draw = drawing_utils
    if info["pose"]:
        draw.draw_landmarks(frame, info["pose"], list(pose.POSE_CONNECTIONS),
                            draw.DrawingSpec(color=(190, 190, 190), thickness=1, circle_radius=1),
                            draw.DrawingSpec(color=(210, 210, 210), thickness=1))
    for hl in info["hands"] or []:
        draw.draw_landmarks(frame, hl, list(hands.HAND_CONNECTIONS),
                            draw.DrawingSpec(color=(255, 99, 37), thickness=2, circle_radius=3),
                            draw.DrawingSpec(color=(255, 255, 255), thickness=2))


def render(predictor):
    _init_state()
    ui.page_title("Live translator", "Sign in front of the camera.",
                  "Raise your hands, sign, then lower them. Each completed sign is added to the sentence.")
    if not predictor.ready:
        st.error("No trained models found. Run `python scripts/train_models.py` first.")
        return

    left, right = st.columns([3, 2], gap="large")
    with left:
        frame_ph = st.empty()
        c1, c2 = st.columns([1, 2])
        run = c1.toggle("Camera on", key="cam_on")
        with c2.expander("Recognition settings"):
            model = st.selectbox("Model", predictor.available(), key="model_choice",
                                 help="Personalised = trained with your own webcam clips (best for your demo). Use a threshold of about 0.5 with it.")
            threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.20, 0.05, key="threshold",
                                  help="Predictions below this are rejected. Tune it using your own tests.")
            cam_index = st.number_input("Camera index", 0, 5, 0, key="cam_index")
    with right:
        panel_ph = st.empty()
        b1, b2, b3 = st.columns(3)
        speak_clicked = b1.button("Speak", type="primary", use_container_width=True)
        b2.button("Undo", on_click=_undo, use_container_width=True)
        b3.button("Clear", on_click=_clear, use_container_width=True)
        st.toggle("Ask as a question", key="as_question",
                  help="Sign language marks questions with facial expression, which this app does not read. Switch this on to phrase the sentence as a question.")
        polish_clicked = st.button("Polish with AI", use_container_width=True,
                                   help="Optional. Sends only the recognised words (text) to an AI service. Needs internet.")
        with st.expander("AI sentence polish (optional)"):
            st.selectbox("Provider", ["Anthropic (Claude)", "OpenAI-compatible"], key="ai_provider")
            st.text_input("API key", type="password", key="ai_key",
                          help="Kept only in this browser session. You can also set ANTHROPIC_API_KEY or OPENAI_API_KEY.")
            st.text_input("Model (optional)", key="ai_model", placeholder="leave empty for the default")
            st.caption("Only the recognised words are sent as text. Video never leaves your computer. "
                       "The AI may only rephrase; replies that add or drop signed words are rejected.")
        with st.expander("Correct or add a word"):
            st.selectbox("Word", RECORDABLE_SIGNS, key="manual_word")
            st.button("Add word", on_click=_add_manual)
        voices = get_voices()
        voice_id = None
        if voices:
            names = [n for _, n in voices]
            voice_id = voices[names.index(st.selectbox("Voice", names, key="voice"))][0]

    if polish_clicked:
        q = bool(st.session_state.get("as_question", False))
        words = list(st.session_state.words)
        with st.spinner("Polishing…"):
            res = ai_polish.polish(words, build_sentence(words, question=q), q,
                                   st.session_state.get("ai_provider", "Anthropic (Claude)"),
                                   st.session_state.get("ai_key", ""), st.session_state.get("ai_model", ""))
        res["key"] = (tuple(words), q)
        st.session_state["polished"] = res
        st.toast(res["message"] or "Done.")

    if speak_clicked:
        words = st.session_state.words
        sentence = build_sentence(words, question=st.session_state.get("as_question", False))
        pol = st.session_state.get("polished")
        if pol and pol["used_ai"] and pol["key"] == (tuple(words), bool(st.session_state.get("as_question", False))):
            sentence = pol["text"]
        if not words:
            st.toast("Nothing to speak yet.")
        else:
            ok, msg = speech_output.speak(sentence, voice_id)
            if sentence != st.session_state.last_spoken:   # save each new sentence once
                db.add_translation(words, sentence, float(np.mean(st.session_state.confs or [0])), model)
                st.session_state.last_spoken = sentence
                st.toast("Spoken and saved to history." if ok else msg)
            elif not ok:
                st.toast(msg)

    if not run:
        frame_ph.markdown('<div class="camera-off"><b>Camera is off</b>Turn on “Camera on” to start translating.</div>',
                          unsafe_allow_html=True)
        panel_ph.markdown(panel_html("idle", thr=threshold), unsafe_allow_html=True)
        return

    cap = cv2.VideoCapture(int(cam_index), cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        st.error("Could not open the camera. Close other apps that use it, or change the camera index.")
        return
    extractor, seg = get_extractor(), SignSegmenter()
    shown = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                st.error("The camera stopped sending frames.")
                break
            vec, info = extractor.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            seq = seg.update(vec, info["active"])
            _draw(frame, info)
            frame_ph.image(cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            time.sleep(0.03)
            if seq is not None:
                res = predictor.predict(seq, model)
                res["accepted"] = accept(res, threshold)
                st.session_state.last, st.session_state.last_time = res, time.time()
                now = time.time()
                if res["accepted"]:
                    dup = res["label"] == st.session_state.last_word and now - st.session_state.last_word_time < 1.5
                    if not dup:
                        st.session_state.words.append(res["label"])
                        st.session_state.confs.append(res["confidence"])
                    st.session_state.last_word, st.session_state.last_word_time = res["label"], now
                shown = None
            if seg.state == "signing":
                key = "signing"
            elif st.session_state.last and time.time() - st.session_state.last_time < 3:
                key = "ok" if st.session_state.last["accepted"] else "reject"
            else:
                key = "idle"
            if key == "signing" or (key, len(st.session_state.words)) != shown:
                panel_ph.markdown(panel_html(key, len(seg.buf), threshold), unsafe_allow_html=True)
                shown = (key, len(st.session_state.words))
    finally:
        cap.release()
