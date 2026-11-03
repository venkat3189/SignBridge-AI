"""Rule-based sentence construction. Never adds words that were not recognised."""

# Complete utterances: rendered as their own sentence with the right punctuation.
PHRASES = {
    "Hello": "Hello!",
    "How are you": "How are you?",
    "Alright": "Alright.",
    "Good Morning": "Good morning.",
    "Good afternoon": "Good afternoon.",
    "Good evening": "Good evening.",
    "Good night": "Good night.",
    "Thank you": "Thank you.",
    "Pleased": "Pleased.",
}


def display_word(word):
    """English rendering of a sign label (plural 'you' is shown as 'you all')."""
    return "you all" if word == "you (plural)" else word


def build_sentence(words):
    """Turn a list of recognised signs into punctuated text.

    Phrase signs become their own sentence; other signs are grouped in the
    order they were signed, capitalised and ended with a full stop.
    """
    sentences, clause = [], []

    def flush():
        if not clause:
            return
        parts = [display_word(w) for w in clause]
        parts = [p if p == "I" else p.lower() for p in parts]
        text = " ".join(parts)
        sentences.append(text[0].upper() + text[1:] + ".")
        clause.clear()

    for w in words:
        if w in PHRASES:
            flush()
            sentences.append(PHRASES[w])
        else:
            clause.append(w)
    flush()
    return " ".join(sentences)
