"""Minimal client for llama.cpp's llama-server /completion endpoint (raw prompts, greedy)."""
from __future__ import annotations

import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
WORKERS = int(os.environ.get("LLAMA_WORKERS", "16"))


def complete(prompt: str, n_predict: int, retries: int = 3) -> dict:
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
            "cache_prompt": True, "seed": 0}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(f"{URL}/completion", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=1800))
            return {"text": r["content"], "stop": r.get("stop_type", "?"),
                    "n_tokens": r.get("tokens_predicted", 0), "prompt_tokens": r.get("tokens_evaluated", 0)}
        except Exception:  # noqa: BLE001 - retried, then raised
            if attempt == retries - 1:
                raise
            time.sleep(5)


def complete_many(jobs: list[tuple[str, int]], on_done=None) -> list[dict]:
    """jobs: (prompt, n_predict). Results in order; on_done(i, result) is called as each ends."""
    out = [None] * len(jobs)

    def run(i):
        out[i] = complete(*jobs[i])
        if on_done:
            on_done(i, out[i])

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(run, range(len(jobs))))
    return out
