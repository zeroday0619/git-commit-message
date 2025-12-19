"""Generate Git commit messages using Ollama local models.

Provides functions to generate commit messages using locally hosted Ollama models.
"""

from __future__ import annotations

from babel import Locale
from os import environ
from typing import Final, TYPE_CHECKING

import re
import requests

if TYPE_CHECKING:
    from ._llm import LLMTextResult

from tiktoken import Encoding, get_encoding


def _get_encoding() -> Encoding:
    """Get a fallback encoding for token counting."""
    try:
        return get_encoding("cl100k_base")
    except Exception:
        # Fallback to a basic encoding if cl100k_base is not available
        return get_encoding("gpt2")


_DEFAULT_MODEL: Final[str] = "ingu627/exaone4.0:1.2b"
_DEFAULT_OLLAMA_HOST: Final[str] = "http://localhost:11434"
_DEFAULT_LANGUAGE: Final[str] = "en-GB"


def _language_display(
    language: str,
    /,
) -> str:
    """Return a human-friendly language display like 'ko-KR, Korean (South Korea)'."""

    try:
        locale = Locale.parse(language, sep="-")
    except Exception:
        return language

    tag_parts = [
        locale.language,
        locale.script,
        locale.territory,
        locale.variant,
    ]
    tag = "-".join(part for part in tag_parts if part)
    if not tag:
        return language

    english_name = locale.get_display_name("en") or ""
    if not english_name:
        return f"[{tag}]"

    return f"{english_name.capitalize()} [{tag}]"


def _build_system_prompt(
    single_line: bool,
    subject_max: int | None,
    language: str,
    /,
) -> str:
    display_language: str = _language_display(language)
    max_len = subject_max or 72
    if single_line:
        return (
            f"You are an expert Git commit message generator. "
            f"Always use '{display_language}' spelling and style. "
            f"Return a single-line imperative subject only (<= {max_len} chars). "
            f"Do not include a body, bullet points, or any rationale. Do not include any line breaks. "
            f"Consider the user-provided auxiliary context if present. "
            f"Return only the commit message text (no code fences or prefixes like 'Commit message:')."
        )
    return (
        f"You are an expert Git commit message generator. "
        f"Always use '{display_language}' spelling and style. "
        f"The subject line is mandatory: you MUST start the output with the subject as the very first non-empty line, "
        f"in imperative mood, and keep it <= {max_len} chars. Insert exactly one blank line after the subject. "
        f"Never start with bullets, headings, labels, or any other text. Then include a body in this format.\n\n"
        f"Example format (do not include the --- lines in the output):\n\n"
        f"---\n\n"
        f"<Subject line>\n\n"
        f"- <detail 1>\n"
        f"- <detail 2>\n"
        f"- <detail N>\n\n"
        f"<Rationale label translated into the target language>: <1-2 concise sentences explaining the intent and why>\n\n"
        f"---\n\n"
        f"Guidelines:\n"
        f"- The first non-empty line MUST be the subject line; include exactly one blank line after it.\n"
        f"- Never place bullets, headings, or labels before the subject line.\n"
        f"- Use '-' bullets; keep each bullet short (<= 1 line).\n"
        f"- Prefer imperative mood verbs (Add, Fix, Update, Remove, Refactor, Document, etc.).\n"
        f"- Focus on what changed and why; avoid copying diff hunks verbatim.\n"
        f"- The only allowed label is the equivalent of 'Rationale:' translated into the target language; do not add other headings or prefaces.\n"
        f"- All text (subject, bullets, rationale label, rationale content) MUST be in the target language: '{display_language}'. Do not mix other languages.\n"
        f"- Do not include the '---' delimiter lines, code fences, or any surrounding labels like 'Commit message:'.\n"
        f"- Do not copy or reuse any example text verbatim; produce original content based on the provided diff and context.\n"
        f"- If few details are necessary, include at least one bullet summarising the key change.\n"
        f"- If you cannot provide any body content, still output the subject line; the subject line must never be omitted.\n"
        f"- Consider the user-provided auxiliary context if present.\n"
        f"Return only the commit message text in the above format (no code fences or extra labels)."
    )


def _resolve_model(
    model: str | None,
    /,
) -> str:
    """Resolve the model name."""

    return (
        model
        or environ.get("GIT_COMMIT_MESSAGE_MODEL")
        or environ.get("OLLAMA_MODEL")
        or _DEFAULT_MODEL
    )


def _resolve_language(
    language: str | None,
    /,
) -> str:
    """Resolve the target language/locale tag used for output style."""

    return language or environ.get("GIT_COMMIT_MESSAGE_LANGUAGE") or _DEFAULT_LANGUAGE


def _resolve_ollama_host(
    host: str | None,
    /,
) -> str:
    """Resolve the Ollama host URL."""

    return (
        host
        or environ.get("OLLAMA_HOST")
        or _DEFAULT_OLLAMA_HOST
    )


class CommitMessageResult:
    """Hold the generated commit message and debugging information.

    Notes
    -----
    Treat all fields as read-only by convention.
    """

    __slots__ = (
        "message",
        "model",
        "prompt",
        "response_text",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
    )

    def __init__(
        self,
        /,
        *,
        message: str,
        model: str,
        prompt: str,
        response_text: str,
        total_tokens: int | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        self.message = message
        self.model = model
        self.prompt = prompt
        self.response_text = response_text
        self.total_tokens = total_tokens
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


def _build_combined_prompt(
    diff: str,
    hint: str | None,
    content_label: str = "Changes (diff)",
    /,
) -> str:
    """Compose a combined string of hint and content for debug/info output."""

    hint_content: str | None = (
        f"# Auxiliary context (user-provided)\n{hint}" if hint else None
    )
    content: str = f"# {content_label}\n{diff}"
    return "\n\n".join([part for part in (hint_content, content) if part is not None])


def _call_ollama(
    prompt: str,
    model: str,
    host: str,
    /,
) -> str:
    """Call Ollama API to generate text.

    Parameters
    ----------
    prompt
        The input prompt/message.
    model
        The model name to use.
    host
        The Ollama host URL.

    Returns
    -------
    str
        The generated text response.

    Raises
    ------
    RuntimeError
        If the API call fails.
    """

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
            f"Failed to connect to Ollama at {host}. "
            f"Make sure Ollama is running: {exc}"
        ) from exc

    try:
        data = response.json()
        raw_response = data.get("response", "").strip()
        
        # Remove thinking/reasoning tags that some models include
        # Pattern: </thought>, </think>, <thought>, </reasoning>, etc.
        import re
        
        # Remove everything before and including </thought> or </think> tags
        if "</thought>" in raw_response:
            raw_response = raw_response.split("</thought>", 1)[-1].strip()
        if "</think>" in raw_response:
            raw_response = raw_response.split("</think>", 1)[-1].strip()
        
        # Remove <thought>...</thought> blocks
        raw_response = re.sub(r'<thought>.*?</thought>', '', raw_response, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove <think>...</think> blocks
        raw_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove <thinking>...</thinking> blocks
        raw_response = re.sub(r'<thinking>.*?</thinking>', '', raw_response, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove [INST]...[/INST] blocks (some models use this)
        raw_response = re.sub(r'\[INST\].*?\[/INST\]', '', raw_response, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up any remaining XML-like tags related to reasoning
        raw_response = re.sub(r'</?(?:reasoning|analysis|rationale)>', '', raw_response, flags=re.IGNORECASE)
        
        # Remove --- delimiter lines (often used in examples or by models)
        raw_response = re.sub(r'^---+\s*$', '', raw_response, flags=re.MULTILINE)
        
        # Strip extra whitespace and clean up multiple blank lines
        raw_response = re.sub(r'\n{3,}', '\n\n', raw_response.strip())
        
        return raw_response
    except ValueError as exc:
        raise RuntimeError(
            f"Failed to parse Ollama response: {exc}"
        ) from exc


def generate_commit_message(
    diff: str,
    hint: str | None,
    model: str | None = None,
    single_line: bool = False,
    subject_max: int | None = None,
    language: str | None = None,
    host: str | None = None,
    /,
) -> str:
    """Generate a commit message using Ollama.

    Parameters
    ----------
    diff
        The staged changes as diff text.
    hint
        Optional auxiliary description of the changes.
    model
        The model name to use. If None, resolves from environment or uses default.
    single_line
        If True, return only a single-line subject.
    subject_max
        Maximum subject line length (default: 72).
    language
        Target language/locale IETF tag for the output.
    host
        Ollama host URL. If None, uses environment variable or default.

    Returns
    -------
    str
        The generated commit message.

    Raises
    ------
    RuntimeError
        If the API call or message generation fails.
    """

    chosen_model: str = _resolve_model(model)
    chosen_language: str = _resolve_language(language)
    chosen_host: str = _resolve_ollama_host(host)

    system_prompt = _build_system_prompt(single_line, subject_max, chosen_language)

    # Build user message with context
    user_message_parts: list[str] = []
    if hint:
        user_message_parts.append(f"# Auxiliary context (user-provided)\n{hint}")
    user_message_parts.append(f"# Changes (diff)\n{diff}")
    user_message = "\n\n".join(user_message_parts)

    # Combine system and user prompts
    full_prompt = f"{system_prompt}\n\n{user_message}"

    # Call Ollama
    response_text = _call_ollama(full_prompt, chosen_model, chosen_host)

    if not response_text:
        raise RuntimeError("An empty commit message was generated.")

    return response_text


def generate_commit_message_with_info(
    diff: str,
    hint: str | None,
    model: str | None = None,
    single_line: bool = False,
    subject_max: int | None = None,
    language: str | None = None,
    host: str | None = None,
    /,
) -> CommitMessageResult:
    """Generate a commit message using Ollama with debugging information.

    Returns
    -------
    CommitMessageResult
        The generated message and debugging information.
    """

    chosen_model: str = _resolve_model(model)
    chosen_language: str = _resolve_language(language)
    chosen_host: str = _resolve_ollama_host(host)

    system_prompt = _build_system_prompt(single_line, subject_max, chosen_language)

    # Build combined prompt for debug output
    combined_prompt = _build_combined_prompt(diff, hint)

    # Build user message with context
    user_message_parts: list[str] = []
    if hint:
        user_message_parts.append(f"# Auxiliary context (user-provided)\n{hint}")
    user_message_parts.append(f"# Changes (diff)\n{diff}")
    user_message = "\n\n".join(user_message_parts)

    # Combine system and user prompts
    full_prompt = f"{system_prompt}\n\n{user_message}"

    # Call Ollama
    response_text = _call_ollama(full_prompt, chosen_model, chosen_host)

    if not response_text:
        raise RuntimeError("An empty commit message was generated.")

    return CommitMessageResult(
        message=response_text,
        model=chosen_model,
        prompt=combined_prompt,
        response_text=response_text,
    )


class OllamaProvider:
    """Ollama provider implementation for LLM protocol."""

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
        """Generate text using Ollama API.

        Parameters
        ----------
        model
            The model name to use.
        instructions
            System instructions/prompt.
        user_text
            User message/content.

        Returns
        -------
        LLMTextResult
            Generated text with metadata.
        """
        from ._llm import LLMTextResult, LLMUsage

        # Combine system and user prompts
        full_prompt = f"{instructions}\n\n{user_text}"

        # Call Ollama
        response_text = _call_ollama(full_prompt, model, self._host)

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
        """Count tokens in text.

        Ollama doesn't provide a native token counting API.
        Uses tiktoken with cl100k_base encoding as an approximation.
        """
        try:
            encoding = _get_encoding()
            return len(encoding.encode(text))
        except Exception:
            # Fallback: rough estimate based on whitespace
            return len(text.split())
