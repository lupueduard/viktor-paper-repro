"""Minimal client for a llama.cpp `llama-server` (OpenAI-compatible chat endpoint)."""
from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


class LlamaServer:
    def __init__(self, url: str = "http://127.0.0.1:8080", workers: int = 8, timeout: float = 600):
        self.url = url.rstrip("/")
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.timeout = timeout

    def chat(self, system: str, user: str, *, temperature: float = 0.0, top_p: float = 1.0,
             max_tokens: int = 48, seed: int | None = None) -> str:
        body = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "cache_prompt": True,
        }
        if seed is not None:
            body["seed"] = seed
        if temperature == 0.0:
            body["top_k"] = 1
        data = json.dumps(body).encode()
        for attempt in range(5):
            try:
                req = urllib.request.Request(f"{self.url}/v1/chat/completions", data=data,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
                return out["choices"][0]["message"]["content"] or ""
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
        return ""

    def chat_many(self, system: str, users: list[str], **kw) -> list[str]:
        return list(self.pool.map(lambda u: self.chat(system, u, **kw), users))
