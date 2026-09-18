"""Decider protocol: local traditional-ML baselines plus the LLM readout.

One protocol: score(row, endpoint) -> result dict with option_ids,
probabilities over the declared options, timing, and a method tag.

  llm            openjev-lite direct readout via the served model
  tfidf-cosine   cosine similarity, evidence+question vs each option description
  naive-bayes    MultinomialNB trained on the option descriptions as classes
  jaccard        keyword-overlap (Jaccard) baseline

The local systems need no training data: the runtime options themselves are
the only classes / neighbors available. Their probabilities are
uncalibrated similarity scores (documented in `notes`), like the LLM readout.
"""

from __future__ import annotations

import json
import math
import re
import time

import openjev_lite
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB


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


def score_llm(row: dict, endpoint: dict) -> dict:
    result = openjev_lite.score_row(row, endpoint)
    result["method"] = "llm-direct-logprobs"
    return result


def score_tfidf_cosine(row: dict, endpoint: dict | None = None) -> dict:
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


def score_naive_bayes(row: dict, endpoint: dict | None = None) -> dict:
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


def score_jaccard(row: dict, endpoint: dict | None = None) -> dict:
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


DECIDERS = [
    ("llm", score_llm),
    ("tfidf-cosine", score_tfidf_cosine),
    ("naive-bayes", score_naive_bayes),
    ("jaccard", score_jaccard),
]


def score_all(row: dict, endpoint: dict) -> dict:
    """Run every decider on one row; a failing system reports its own error."""
    results = {}
    for name, fn in DECIDERS:
        try:
            results[name] = fn(row, endpoint)
        except Exception as error:
            results[name] = {"id": row["id"], "method": name, "error": str(error)}
    return results
