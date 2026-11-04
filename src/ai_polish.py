"""Optional AI sentence polish.

Only the recognised sign words (text) are sent to the provider; video never leaves the
device. The reply is validated: if it adds a content word that was not signed, or drops
a signed word, it is rejected and the rule-based sentence is kept.
Uses only the standard library (urllib), so no extra packages are needed.
"""
import json
import os
import re
import urllib.error
import urllib.request

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-haiku-5-5"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"

SYSTEM = (
    "You help turn words recognised from Indian Sign Language into a natural English sentence. "
    "Indian Sign Language often puts the verb last and drops small grammar words. "
    "Rules: use ONLY the recognised words (in any grammatical form) plus small function words "
    "(a, the, to, is, do, please, ...). Never add new facts, names, places or content words. "
    "Never drop a recognised word. Keep the meaning and the order of ideas. "
    "If the draft is already correct, or the words do not form a sensible sentence, return the draft unchanged. "
    "Reply with the sentence only: one line, no quotes, no explanation."
)

FUNCTION_WORDS = {
    "a", "an", "the", "to", "is", "are", "am", "be", "do", "does", "can", "could", "will",
    "would", "please", "not", "and", "of", "for", "with", "in", "on", "at", "some", "any",
}
PRONOUN_FORMS = {
    "I": {"i", "me", "my", "mine"}, "you": {"you", "your", "yours"},
    "he": {"he", "him", "his"}, "she": {"she", "her", "hers"}, "it": {"it", "its"},
    "we": {"we", "us", "our", "ours"}, "you (plural)": {"you", "your", "yours", "all"},
    "they": {"they", "them", "their", "theirs"},
}


def _stem(word):
    w = word.lower()
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    return w.rstrip("e")


def validate(output, words):
    """Return (ok, reason). Rejects added content words and dropped signed words."""
    if not output or "\n" in output.strip():
        return False, "empty or multi-line reply"
    tokens = re.findall(r"[A-Za-z']+", output)
    if not tokens or len(tokens) > len(words) * 4 + 6:
        return False, "reply too long"
    allowed, required = set(FUNCTION_WORDS), []
    for w in words:
        if w in PRONOUN_FORMS:
            allowed |= PRONOUN_FORMS[w]
            required.append(PRONOUN_FORMS[w])
        else:
            for t in re.findall(r"[A-Za-z]+", w):
                if t.lower() in FUNCTION_WORDS:
                    continue
                allowed.add(t.lower())
                required.append({t.lower()})
    allowed_stems = {_stem(a) for a in allowed}
    for t in tokens:
        if t.lower() not in allowed and _stem(t) not in allowed_stems:
            return False, f"added a word that was not signed: '{t}'"
    out_stems = {_stem(t) for t in tokens} | {t.lower() for t in tokens}
    for forms in required:
        if not any(f in out_stems or _stem(f) in out_stems for f in forms):
            return False, f"dropped a signed word ({sorted(forms)[0]})"
    return True, ""


def _post(url, headers, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"content-type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _call(provider, api_key, model, user, base_url, timeout):
    if provider.startswith("Anthropic"):
        data = _post(base_url or ANTHROPIC_URL,
                     {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                     {"model": model or ANTHROPIC_MODEL, "max_tokens": 80, "system": SYSTEM,
                      "messages": [{"role": "user", "content": user}]}, timeout)
        return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
    data = _post(base_url or OPENAI_URL, {"Authorization": f"Bearer {api_key}"},
                 {"model": model or OPENAI_MODEL,
                  "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}, timeout)
    return data["choices"][0]["message"]["content"]


def polish(words, draft, question=False, provider="Anthropic (Claude)", api_key="", model="",
           base_url="", timeout=15):
    """Returns {"text", "used_ai", "message"}. Always safe: falls back to the draft."""
    fallback = {"text": draft, "used_ai": False, "message": ""}
    if not words:
        return {**fallback, "message": "Nothing to polish yet."}
    env = "ANTHROPIC_API_KEY" if provider.startswith("Anthropic") else "OPENAI_API_KEY"
    api_key = (api_key or os.environ.get(env, "")).strip()
    if not api_key:
        return {**fallback, "message": "Add an API key in 'AI sentence polish' first."}
    user = (f"Recognised signs, in order: {', '.join(words)}\n"
            f"Rule-based draft: {draft}\n"
            f"The signer marked this as a question: {'yes' if question else 'no'}\n"
            "Return the best sentence.")
    try:
        text = _call(provider, api_key, model.strip(), user, base_url, timeout).strip().strip('"').strip()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")[:150]
        return {**fallback, "message": f"AI service error {exc.code}: {detail}"}
    except urllib.error.URLError:
        return {**fallback, "message": "No internet connection. Using the offline sentence."}
    except Exception as exc:
        return {**fallback, "message": f"AI polish failed: {exc}"}
    ok, reason = validate(text, words)
    if not ok:
        return {**fallback, "message": f"AI reply rejected ({reason}). Using the offline sentence."}
    return {"text": text, "used_ai": True, "message": "AI-polished."}
