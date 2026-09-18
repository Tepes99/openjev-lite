"""openjev direct-mode readout over an OpenAI-compatible chat endpoint.

Imitates openjev (TheoLeeCJ/openjev) in minimal form: same input schema,
byte-identical prompt, same 2-16 option validation, same output fields.
One deviation from upstream: instead of a local forward pass reading
full-vocabulary logits, this reads the served model's top-N logprobs.
A letter absent from the top-N gets probability 0 (truncated readout,
documented in the output's `readout` field).

Stdlib only. Endpoint defaults come from ~/.pi/agent/models.json (the
"home-vllm" provider), overridable via JEV_BASE_URL / JEV_API_KEY /
JEV_MODEL / JEV_MODELS_FILE.

Usage:
    uv run python openjev_lite.py --input decisions.jsonl --output results.jsonl
"""

import argparse
import hashlib
import json
import math
import os
import ssl
import time
import urllib.request
from pathlib import Path

import certifi

LETTERS = "ABCDEFGHIJKLMNOP"
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
PROMPT_VERSION = "direct-options-v1"
DEFAULT_MODELS_FILE = Path.home() / ".pi" / "agent" / "models.json"
DEFAULT_PROVIDER = "home-vllm"
# Server-observed cap for top_logprobs on the home endpoint; fits 16 options.
TOP_LOGPROBS = 20


def validate_row(row: dict) -> None:
    required = {"id", "state", "question", "options"}
    if not required <= row.keys():
        raise ValueError(f"Row is missing fields: {sorted(required - row.keys())}")
    if not all(isinstance(row[key], str) and row[key] for key in ("id", "question")):
        raise ValueError("id and question must be nonempty strings")
    state = row["state"]
    if not isinstance(state, (str, dict, list)) or not state:
        raise ValueError("state must be a nonempty string, object, or array")
    try:
        json.dumps(state, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("state must be finite JSON-compatible data") from error
    options = row["options"]
    if not isinstance(options, list) or not 2 <= len(options) <= len(LETTERS):
        raise ValueError("options must contain 2-16 entries")
    ids = []
    for option in options:
        if not isinstance(option, dict) or not isinstance(option.get("id"), str) or not isinstance(option.get("description"), str):
            raise ValueError("Each option needs string id and description fields")
        ids.append(option["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Option IDs must be unique")


def build_messages(row: dict) -> list[dict]:
    """Byte-identical prompt to openjev's direct_messages()."""
    validate_row(row)
    payload = {
        "evidence": row["state"],
        "criterion": row["question"],
        "options": [
            {"letter": LETTERS[index], "description": option["description"]}
            for index, option in enumerate(row["options"])
        ],
    }
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def softmax(values: list[float]) -> list[float]:
    """Like upstream, but tolerates -inf (letter missing from top-N)."""
    finite = [value for value in values if math.isfinite(value)]
    if len(values) < 2 or not finite:
        raise ValueError("Need at least two scores with at least one finite")
    maximum = max(finite)
    weights = [math.exp(value - maximum) if math.isfinite(value) else 0.0 for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def load_endpoint() -> dict:
    models_file = Path(os.environ.get("JEV_MODELS_FILE", DEFAULT_MODELS_FILE))
    provider = None
    base_url = os.environ.get("JEV_BASE_URL")
    api_key = os.environ.get("JEV_API_KEY")
    model = os.environ.get("JEV_MODEL")
    if any(value is None for value in (base_url, api_key, model)):
        provider = json.loads(models_file.read_text())["providers"][DEFAULT_PROVIDER]
        base_url = base_url or provider["baseUrl"]
        api_key = api_key or provider["apiKey"]
        model = model or provider["models"][0]["id"]
    return {"base_url": base_url.rstrip("/"), "api_key": api_key, "model": model}


def request_completion(endpoint: dict, messages: list[dict]) -> dict:
    body = json.dumps(
        {
            "model": endpoint["model"],
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": TOP_LOGPROBS,
            # Required: the served model is a reasoning model and thinking
            # tokens would poison the first-token readout.
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    request = urllib.request.Request(
        f"{endpoint['base_url']}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {endpoint['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    context = ssl.create_default_context(cafile=certifi.where()) if endpoint["base_url"].startswith("https") else None
    with urllib.request.urlopen(request, timeout=300, context=context) as response:
        return json.loads(response.read())


def score_row(row: dict, endpoint: dict) -> dict:
    started = time.perf_counter()
    messages = build_messages(row)
    response = request_completion(endpoint, messages)
    total_seconds = time.perf_counter() - started
    try:
        first_token = response["choices"][0]["logprobs"]["content"][0]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"Row {row['id']}: no first-token logprobs in response") from error
    logprobs = {entry["token"]: entry["logprob"] for entry in first_token.get("top_logprobs") or []}
    option_logprobs = [logprobs.get(LETTERS[i], float("-inf")) for i in range(len(row["options"]))]
    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": softmax(option_logprobs),
        "option_logprobs": option_logprobs,
        "input_tokens": response.get("usage", {}).get("prompt_tokens"),
        "total_seconds": total_seconds,
        "prompt_sha256": hashlib.sha256(
            json.dumps(messages, ensure_ascii=False).encode()
        ).hexdigest(),
        "prompt_version": PROMPT_VERSION,
        "model": {"source": endpoint["base_url"], "model": endpoint["model"]},
        "readout": f"top-{TOP_LOGPROBS} logprobs at first token; letters outside the top-N get probability 0 (truncated readout)",
        "probability_status": "conditional option score; uncalibrated as decision confidence",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("Input is empty")
    endpoint = load_endpoint()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        for row in rows:
            destination.write(json.dumps(score_row(row, endpoint)) + "\n")
            destination.flush()


if __name__ == "__main__":
    main()
