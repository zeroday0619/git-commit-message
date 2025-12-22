"""Ollama provider implementation.

Mirrors the Gemini provider structure: a single provider class that exposes
`generate_text` and `count_tokens`. Provider-agnostic orchestration lives in
`_llm.py`.
"""

from __future__ import annotations

from os import environ
from typing import Final

import re
import requests
from tiktoken import Encoding, get_encoding

from ._llm import LLMTextResult, LLMUsage


_DEFAULT_OLLAMA_HOST: Final[str] = "http://localhost:11434"


def _resolve_ollama_host(host: str | None) -> str:
    """Resolve the Ollama host URL from arg, env, or default."""

    return host or environ.get("OLLAMA_HOST") or _DEFAULT_OLLAMA_HOST


def _clean_response(raw_response: str) -> str:
    """Strip leading reasoning/thinking wrappers some models return.

    Only removes reasoning blocks that appear at the START of the response.
    Content inside the actual commit message body is preserved.
    """

    cleaned = raw_response

    # Remove leading reasoning blocks only (at the start of response).
    # These patterns match blocks that start from the beginning.
    cleaned = re.sub(r"^\s*<think>.*?</think>\s*", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"^\s*<thinking>.*?</thinking>\s*", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"^\s*<thought>.*?</thought>\s*", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"^\s*\[INST\].*?\[/INST\]\s*", "", cleaned, flags=re.IGNORECASE | re.DOTALL)

    # Remove standalone opening/closing reasoning tags only if they appear on their own line at start
    cleaned = re.sub(r"^\s*</?(?:think|thinking|thought|reasoning|analysis|rationale)>\s*\n?", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)

    # Remove leading horizontal rules often used as separators
    cleaned = re.sub(r"^\s*---+\s*\n", "", cleaned, flags=re.MULTILINE)

    # Normalize excessive blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _call_ollama(*, prompt: str, model: str, host: str) -> str:
    """Call Ollama /api/generate and return cleaned text."""

    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to connect to Ollama at {host}. Make sure Ollama is running: {exc}"
        ) from exc

    try:
        data = response.json()
        raw_response = data.get("response", "").strip()
        return _clean_response(raw_response)
    except ValueError as exc:
        raise RuntimeError(f"Failed to parse Ollama response: {exc}") from exc


def _get_encoding() -> Encoding:
    """Get a fallback encoding for token counting."""

    try:
        return get_encoding("cl100k_base")
    except Exception:
        return get_encoding("gpt2")


class OllamaProvider:
    """Ollama provider implementation for the LLM protocol."""

    name: str = "ollama"

    def __init__(self, /, *, host: str | None = None) -> None:
        self._host = _resolve_ollama_host(host)

    def generate_text(
        self,
        /,
        *,
        model: str,
        instructions: str,
        user_text: str,
    ) -> LLMTextResult:
        """Generate text using Ollama (non-streaming)."""

        full_prompt = f"{instructions}\n\n{user_text}"
        response_text = _call_ollama(
            prompt=full_prompt,
            model=model,
            host=self._host,
        )

        return LLMTextResult(
            text=response_text,
            response_id=None,
            usage=LLMUsage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            ),
        )

    def count_tokens(
        self,
        /,
        *,
        model: str,
        text: str,
    ) -> int:
        """Approximate token count using tiktoken; fallback to whitespace split."""

        try:
            encoding = _get_encoding()
            return len(encoding.encode(text))
        except Exception:
            return len(text.split())
