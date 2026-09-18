"""Decider protocol: local traditional-ML baselines plus the laya decision model.

One protocol: score(row) -> result dict with option_ids, probabilities over
the declared options, timing, and a method tag.

  tfidf-cosine   cosine similarity, evidence+question vs each option description
  naive-bayes    MultinomialNB trained on the option descriptions as classes
  jaccard        keyword-overlap (Jaccard) baseline
  knn            KNeighborsClassifier (cosine, distance-weighted) on options
  laya           fully fine-tuned non-autoregressive decision model (base method)

The local systems need no training data: the runtime options themselves are
the only classes / neighbors available. Their probabilities are uncalibrated
similarity scores (documented in `notes`); laya is the calibrated base.
"""

from __future__ import annotations

import json
import math
import re
import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


def state_to_text(state) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, allow_nan=False)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _softmax(values: list[float], temperature: float) -> list[float]:
    scaled = [value * temperature for value in values]
    maximum = max(scaled)
    weights = [math.exp(v - maximum) for v in scaled]
    total = sum(weights)
    return [weight / total for weight in weights]


def _result(row: dict, option_ids: list[str], probabilities: list[float], started: float, method: str, notes: str) -> dict:
    return {
        "id": row["id"],
        "option_ids": option_ids,
        "probabilities": probabilities,
        "total_seconds": time.perf_counter() - started,
        "input_tokens": len(_tokens(state_to_text(row["state"]) + " " + row["question"])),
        "method": method,
        "notes": notes,
        "probability_status": "conditional option score; uncalibrated as decision confidence",
    }


def score_tfidf_cosine(row: dict) -> dict:
    started = time.perf_counter()
    query = state_to_text(row["state"]) + " " + row["question"]
    docs = [option["description"] for option in row["options"]]
    non_empty = [doc for doc in docs if doc]
    if not non_empty:
        sims = [0.0] * len(docs)
    else:
        vectorizer = TfidfVectorizer().fit(non_empty)
        sims = cosine_similarity(vectorizer.transform([query]), vectorizer.transform(docs))[0]
    return _result(
        row, [option["id"] for option in row["options"]], _softmax(list(sims), 12.0), started,
        "tfidf-cosine-v1",
        "cosine similarity of evidence+question vs each option description; softmax(12*sim)",
    )


def score_naive_bayes(row: dict) -> dict:
    started = time.perf_counter()
    query = state_to_text(row["state"]) + " " + row["question"]
    docs = [option["description"] for option in row["options"]]
    corpus, labels = [], []
    for i, doc in enumerate(docs):
        # repeat each description: one doc per class leaves NB no word counts
        corpus.extend([doc] * 3)
        labels.extend([i] * 3)
    vectorizer = CountVectorizer()
    clf = MultinomialNB().fit(vectorizer.fit_transform(corpus), labels)
    probs = list(clf.predict_proba(vectorizer.transform([query]))[0])
    return _result(
        row, [option["id"] for option in row["options"]], probs, started,
        "naive-bayes-v1",
        "MultinomialNB on option descriptions (3x per class); predict_proba on evidence+question",
    )


def score_jaccard(row: dict) -> dict:
    started = time.perf_counter()
    query = set(_tokens(state_to_text(row["state"]) + " " + row["question"]))
    sims = []
    for option in row["options"]:
        doc_tokens = set(_tokens(option["description"]))
        union = len(query | doc_tokens)
        sims.append(len(query & doc_tokens) / union if union else 0.0)
    return _result(
        row, [option["id"] for option in row["options"]], _softmax(sims, 20.0), started,
        "jaccard-v1",
        "keyword Jaccard overlap of evidence+question vs each option description; softmax(20*sim)",
    )


def score_knn(row: dict) -> dict:
    started = time.perf_counter()
    query = state_to_text(row["state"]) + " " + row["question"]
    docs = [option["description"] for option in row["options"]]
    non_empty = [(i, doc) for i, doc in enumerate(docs) if doc]
    if len(non_empty) < 2:
        probs = [1.0 / len(docs)] * len(docs)
    else:
        vectorizer = TfidfVectorizer().fit([doc for _, doc in non_empty])
        X = vectorizer.transform([doc for _, doc in non_empty])
        y = [i for i, _ in non_empty]
        k = min(3, len(non_empty))
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
        knn.fit(X, y)
        raw = knn.predict_proba(vectorizer.transform([query]))[0]
        probs = [0.0] * len(docs)
        for cls, p in zip(knn.classes_, raw):
            probs[cls] = p
        total = sum(probs)
        probs = [p / total for p in probs]
    return _result(
        row, [option["id"] for option in row["options"]], probs, started,
        "knn-v1",
        "KNeighborsClassifier (cosine, distance-weighted, k=min(3,n)) on option descriptions; predict_proba on evidence+question",
    )


# --- laya: a small fully fine-tuned non-autoregressive decision model (base) ---
import threading
from pathlib import Path

_laya_agent = None
_laya_load_error = None
_laya_lock = threading.Lock()
# In the deployed container the weights live on the persistent /data volume, so
# they are downloaded once (first boot) and reused across restarts. Locally,
# prefer the git-ignored models/laya copy, else fall back to the HF repo id.
_LAYA_DATA = Path("/data/laya")
_LAYA_LOCAL = Path(__file__).parent / "models" / "laya"


def _laya_source() -> str:
    if _LAYA_DATA.parent.exists():
        # /data is mounted (deployed container): fetch into the volume once.
        if not _LAYA_DATA.exists():
            from huggingface_hub import snapshot_download

            snapshot_download("convaiinnovations/laya", local_dir=str(_LAYA_DATA))
        return str(_LAYA_DATA)
    if _LAYA_LOCAL.exists():
        return str(_LAYA_LOCAL)
    return "convaiinnovations/laya"


def _get_laya():
    """Thread-safe lazy singleton for the laya agent.

    Loading is the one-time cost (a fresh MPS/CPU init is slow, and a missing
    local copy triggers a one-time download), so we load once and reuse.
    prewarm_laya() is fired in a background thread at server startup so the
    first real request does not pay the load on the request thread; if a
    request arrives before the prewarm finishes, it simply waits on the lock
    and gets the already-loaded agent (no double load).
    """
    global _laya_agent, _laya_load_error
    if _laya_agent is None and _laya_load_error is None:
        with _laya_lock:
            if _laya_agent is None and _laya_load_error is None:
                try:
                    import laya

                    _laya_agent = laya.load(_laya_source())
                except Exception as error:  # noqa: BLE001 - reported in the system's column
                    _laya_load_error = f"{type(error).__name__}: {error}"
    if _laya_load_error:
        raise RuntimeError(_laya_load_error)
    return _laya_agent


def prewarm_laya() -> None:
    """Kick off the one-time laya load; safe to call from a background thread."""
    _get_laya()


def score_laya(row: dict) -> dict:
    """Run one row as a single 'choice' question through the laya model.

    mapping: row.state -> laya state, row.question -> instructions,
    row.options -> criteria {option id: description}. laya scores every option
    at its own [MASK] marker in one forward pass and returns a temperature-
    calibrated softmax over the options (plus a calibrated confidence).
    """
    started = time.perf_counter()
    agent = _get_laya()
    # keep state as-is (str/dict/list are all valid laya states)
    questions = {
        "decision": {
            "type": "choice",
            "instructions": row["question"],
            "criteria": {option["id"]: option["description"] for option in row["options"]},
        }
    }
    answer = agent.predict(row["state"], questions)["answers"]["decision"]
    option_ids = [option["id"] for option in row["options"]]
    probabilities = [float(answer["probabilities"][oid]) for oid in option_ids]
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities] if total > 0 else [1.0 / len(probabilities)] * len(probabilities)
    result = _result(
        row, option_ids, probabilities, started,
        "laya-v1",
        "laya non-autoregressive decision model (ModernBERT-large + decision head), "
        "one choice question, marker-scored softmax over options, temperature-calibrated",
    )
    # laya's headline extra: a calibrated confidence the others do not have
    result["confidence"] = answer.get("confidence")
    return result


DECIDERS = [
    ("tfidf-cosine", score_tfidf_cosine),
    ("naive-bayes", score_naive_bayes),
    ("jaccard", score_jaccard),
    ("knn", score_knn),
    ("laya", score_laya),
]


def score_all(row: dict) -> dict:
    """Run every decider on one row; a failing system reports its own error."""
    results = {}
    for name, fn in DECIDERS:
        try:
            results[name] = fn(row)
        except Exception as error:
            results[name] = {"id": row["id"], "method": name, "error": str(error)}
    return results
