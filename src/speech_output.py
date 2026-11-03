"""Offline text-to-speech with pyttsx3 (uses the voices installed in Windows)."""


def list_voices():
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = [(v.id, v.name) for v in engine.getProperty("voices")]
        engine.stop()
        return voices
    except Exception:
        return []


def speak(text, voice_id=None, rate=170):
    """Speak text. Returns (ok, message). Blocks until speech finishes."""
    if not text or not text.strip():
        return False, "Nothing to speak."
    try:
        import pyttsx3
        engine = pyttsx3.init()
        if voice_id:
            engine.setProperty("voice", voice_id)
        engine.setProperty("rate", rate)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        return True, "Spoken."
    except Exception as exc:
        return False, f"Text-to-speech failed: {exc}"
