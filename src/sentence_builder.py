"""Rule-based sentence construction.

Only grammar is added (verb endings, "do/does" in questions, the article "a" for
doctor, punctuation, word order). Content words are never invented or dropped.
Indian Sign Language often puts the verb last ("I WATER WANT"); this module
reorders that into English ("I want water.").
"""

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
    "Yes": "Yes.",
    "No": "No.",
}

PRONOUNS = {"I", "you", "he", "she", "it", "we", "you (plural)", "they"}
THIRD_SINGULAR = {"he", "she", "it"}
VERBS = {"Need": "need", "Want": "want", "Exercise": "exercise"}
OBJECT_FORMS = {"Doctor": "a doctor"}       # nouns that take an article after a verb


def display_word(word):
    """English rendering of a sign label (plural 'you' is shown as 'you all')."""
    return "you all" if word == "you (plural)" else word


def _lower(word):
    w = display_word(word)
    return w if w == "I" else w.lower()


def _cap(text):
    return text[0].upper() + text[1:] if text else text


def _clause(tokens, question=False):
    if tokens == ["Help"]:
        return "Help!"
    rest = list(tokens)
    subject = next((t for t in rest if t in PRONOUNS), None)
    if subject:
        rest.remove(subject)
    verb = next((t for t in rest if t in VERBS), None)
    end = "?" if question else "."

    if verb:
        rest.remove(verb)
        objects = [OBJECT_FORMS.get(t, _lower(t)) for t in rest]
        base = VERBS[verb]
        if subject and question:
            aux = "does" if subject in THIRD_SINGULAR else "do"
            words = [aux, _lower(subject), base] + objects
        elif subject:
            form = base + "s" if subject in THIRD_SINGULAR else base
            words = [_lower(subject), form] + objects
        else:
            words = [base] + objects
        return _cap(" ".join(words)) + end

    return _cap(" ".join(_lower(t) for t in tokens)) + end


def build_sentence(words, question=False):
    """Turn a list of recognised signs into punctuated English text.

    question=True phrases the last group of signs as a question (the app cannot
    see facial expression, so the user chooses this).
    """
    items, clause = [], []
    for w in words:
        if w in PHRASES:
            if clause:
                items.append(("clause", clause))
                clause = []
            items.append(("phrase", PHRASES[w]))
        else:
            clause.append(w)
    if clause:
        items.append(("clause", clause))

    out = []
    for i, (kind, value) in enumerate(items):
        if kind == "phrase":
            out.append(value)
        else:
            out.append(_clause(value, question and i == len(items) - 1))
    return " ".join(out)
