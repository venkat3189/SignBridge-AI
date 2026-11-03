"""SQLite persistence layer (local file, no server needed)."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from src.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS translations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    words TEXT NOT NULL,
    sentence TEXT NOT NULL,
    mean_confidence REAL,
    model TEXT
);
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    direction TEXT NOT NULL,      -- 'sign_to_speech' or 'speech_to_text'
    text TEXT NOT NULL,
    language TEXT
);
CREATE TABLE IF NOT EXISTS practice_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    target_sign TEXT NOT NULL,
    predicted_sign TEXT,
    correct INTEGER NOT NULL,
    confidence REAL
);
CREATE TABLE IF NOT EXISTS model_runs (
    model TEXT PRIMARY KEY,
    accuracy REAL, precision_macro REAL, recall_macro REAL, f1_macro REAL,
    inference_ms REAL, n_test INTEGER, updated TEXT
);
"""


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _conn() as c:
        c.executescript(SCHEMA)


def add_translation(words, sentence, mean_confidence, model):
    with _conn() as c:
        c.execute("INSERT INTO translations (ts, words, sentence, mean_confidence, model) VALUES (?,?,?,?,?)",
                  (_now(), json.dumps(list(words)), sentence, mean_confidence, model))


def recent_translations(limit=200, search=""):
    q = "SELECT id, ts, words, sentence, mean_confidence, model FROM translations"
    args = []
    if search:
        q += " WHERE sentence LIKE ? OR words LIKE ?"
        args += [f"%{search}%", f"%{search}%"]
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with _conn() as c:
        rows = [dict(r) for r in c.execute(q, args)]
    for r in rows:
        try:
            r["words"] = ", ".join(json.loads(r["words"]))
        except Exception:
            pass
    return rows


def clear_translations():
    with _conn() as c:
        c.execute("DELETE FROM translations")


def add_conversation(direction, text, language=""):
    with _conn() as c:
        c.execute("INSERT INTO conversations (ts, direction, text, language) VALUES (?,?,?,?)",
                  (_now(), direction, text, language))


def recent_conversations(limit=30):
    with _conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT ts, direction, text, language FROM conversations ORDER BY id DESC LIMIT ?", (limit,))]


def add_practice(target, predicted, correct, confidence):
    with _conn() as c:
        c.execute("INSERT INTO practice_results (ts, target_sign, predicted_sign, correct, confidence) VALUES (?,?,?,?,?)",
                  (_now(), target, predicted, int(correct), confidence))


def practice_rows(limit=200):
    with _conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT ts, target_sign, predicted_sign, correct, confidence FROM practice_results ORDER BY id DESC LIMIT ?",
            (limit,))]


def count(table):
    if table not in {"translations", "conversations", "practice_results", "model_runs"}:
        raise ValueError(table)
    with _conn() as c:
        return c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def sync_model_runs(metrics):
    """Mirror the real evaluation results from metrics.json into the database."""
    if not metrics:
        return
    n_test = metrics.get("dataset", {}).get("test")
    with _conn() as c:
        c.execute("DELETE FROM model_runs")
        for name, m in metrics.get("models", {}).items():
            c.execute("INSERT INTO model_runs VALUES (?,?,?,?,?,?,?,?)",
                      (name, m["accuracy"], m["precision_macro"], m["recall_macro"], m["f1_macro"],
                       m["inference_ms"], n_test, _now()))
