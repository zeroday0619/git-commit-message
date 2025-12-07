"""Generate Git commit messages by calling an OpenAI GPT model.

Migrated to use OpenAI Responses API (client.responses.create).
"""

from __future__ import annotations

from babel import Locale
from openai import OpenAI
from openai.types.responses import Response, ResponseInputParam
from os import environ
from tiktoken import Encoding, encoding_for_model, get_encoding
from typing import Final


_DEFAULT_MODEL: Final[str] = "gpt-5-mini"
_DEFAULT_LANGUAGE: Final[str] = "en-GB"


def _encoding_for_model(
    model: str,
    /,
) -> Encoding:
    try:
        return encoding_for_model(model)
    except Exception:
        return get_encoding("cl100k_base")


def _count_tokens(
    text: str,
    *,
    model: str,
) -> int:
    encoding = _encoding_for_model(model)
    return len(encoding.encode(text))


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


def _instructions(
    single_line: bool,
    subject_max: int | None,
    language: str,
    /,
) -> str:
    """Create the system/developer instructions string for the Responses API."""
    return _build_system_prompt(single_line, subject_max, language)


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
        "response_id",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    )

    def __init__(
        self,
        /,
        *,
        message: str,
        model: str,
        prompt: str,
        response_text: str,
        response_id: str | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
    ) -> None:
        self.message = message
        self.model = model
        self.prompt = prompt
        self.response_text = response_text
        self.response_id = response_id
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


def _resolve_model(
    model: str | None,
    /,
) -> str:
    """Resolve the model name."""

    return (
        model
        or environ.get("GIT_COMMIT_MESSAGE_MODEL")
        or environ.get("OPENAI_MODEL")
        or _DEFAULT_MODEL
    )


def _resolve_language(
    language: str | None,
    /,
) -> str:
    """Resolve the target language/locale tag used for output style."""

    return language or environ.get("GIT_COMMIT_MESSAGE_LANGUAGE") or _DEFAULT_LANGUAGE


def _build_responses_input(
    diff: str,
    hint: str | None,
    /,
) -> ResponseInputParam:
    """Compose Responses API input items, separating auxiliary context and diff.

    Returns
    -------
    ResponseInputParam
        The list of input items to send to the Responses API.
    """

    hint_content: str | None = (
        f"# Auxiliary context (user-provided)\n{hint}" if hint else None
    )
    diff_content: str = f"# Changes (diff)\n{diff}"

    input_items: ResponseInputParam = []
    if hint_content:
        input_items.append(
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": hint_content},
                ],
            }
        )
    input_items.append(
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": diff_content},
            ],
        }
    )

    return input_items


def _split_diff_into_hunks(
    diff: str,
    /,
) -> list[str]:
    lines = diff.splitlines(keepends=True)
    hunks: list[str] = []
    file_header: list[str] = []
    current_hunk: list[str] | None = None

    for line in lines:
        if line.startswith("diff --git "):
            if current_hunk:
                hunks.append("".join(current_hunk))
                current_hunk = None
            file_header = [line]
            continue

        if line.startswith("@@"):
            if current_hunk:
                hunks.append("".join(current_hunk))
            base_header = file_header[:] if file_header else []
            current_hunk = base_header + [line]
            continue

        if current_hunk is not None:
            current_hunk.append(line)
            continue

        if file_header:
            file_header.append(line)
            continue

        # Lines outside a diff header/hunk; keep as standalone hunk
        current_hunk = [line]

    if current_hunk:
        hunks.append("".join(current_hunk))

    return hunks


def _build_diff_chunks(
    hunks: list[str],
    chunk_tokens: int,
    model: str,
    /,
) -> list[str]:
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive when chunking is enabled")

    chunks: list[str] = []
    current: list[str] = []

    for hunk in hunks:
        candidate = "".join(current + [hunk])
        token_count = _count_tokens(candidate, model=model)

        if token_count <= chunk_tokens:
            current.append(hunk)
            continue

        if current:
            chunks.append("".join(current))
            current = [hunk]
        else:
            single_tokens = _count_tokens(hunk, model=model)
            if single_tokens > chunk_tokens:
                raise ValueError(
                    "chunk_tokens is too small to fit a single diff hunk; increase the value or disable chunking"
                )
            current = [hunk]

    if current:
        chunks.append("".join(current))

    return chunks


def _build_chunk_summary_prompt() -> str:
    return (
        "You are an expert developer summarising Git diffs. "
        "Write detailed English bullet points describing what changed and why. "
        "Do not copy large code blocks verbatim; focus on behavior and intent. "
        "Be verbose when useful; this summary will later be used to craft a commit message."
    )


def _summarise_diff_chunks(
    chunks: list[str],
    model: str,
    client: OpenAI,
    /,
) -> list[tuple[str, Response]]:
    if not chunks:
        return []

    instructions = _build_chunk_summary_prompt()
    summaries: list[tuple[str, Response]] = []

    for chunk in chunks:
        resp = client.responses.create(
            model=model,
            instructions=instructions,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"# Diff chunk\n{chunk}",
                        }
                    ],
                }
            ],
        )

        text: str = (resp.output_text or "").strip()
        if not text:
            raise RuntimeError("An empty chunk summary was generated.")

        summaries.append((text, resp))

    return summaries


def _generate_commit_from_summaries(
    summaries: list[str],
    hint: str | None,
    model: str,
    single_line: bool,
    subject_max: int | None,
    language: str,
    client: OpenAI,
    /,
) -> tuple[str, Response]:
    instructions = _instructions(single_line, subject_max, language)
    sections: list[str] = []

    if hint:
        sections.append(f"# Auxiliary context (user-provided)\n{hint}")

    if summaries:
        numbered = [
            f"Summary {idx + 1}:\n{summary}" for idx, summary in enumerate(summaries)
        ]
        sections.append(
            "# Combined summaries of the commit (in English)\n" + "\n\n".join(numbered)
        )
    else:
        sections.append("# No summaries available")

    user_content = "\n\n".join(sections)

    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_content,
                    }
                ],
            }
        ],
    )

    text: str = (resp.output_text or "").strip()
    if not text:
        raise RuntimeError("An empty commit message was generated from summaries.")

    return text, resp


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


def generate_commit_message(
    diff: str,
    hint: str | None,
    model: str | None,
    single_line: bool = False,
    subject_max: int | None = None,
    language: str | None = None,
    chunk_tokens: int | None = 0,
    /,
) -> str:
    """Generate a commit message using an OpenAI GPT model."""

    chosen_model: str = _resolve_model(model)
    chosen_language: str = _resolve_language(language)
    api_key = environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("The OPENAI_API_KEY environment variable is required.")

    client = OpenAI(api_key=api_key)

    normalized_chunk_tokens = 0 if chunk_tokens is None else chunk_tokens

    if normalized_chunk_tokens != -1:
        hunks = _split_diff_into_hunks(diff)
        if normalized_chunk_tokens == 0:
            chunks = ["".join(hunks) if hunks else diff]
        elif normalized_chunk_tokens > 0:
            chunks = _build_diff_chunks(
                hunks,
                normalized_chunk_tokens,
                chosen_model,
            )
        else:
            chunks = ["".join(hunks) if hunks else diff]

        summary_pairs = _summarise_diff_chunks(
            chunks,
            chosen_model,
            client,
        )
        summary_texts = [text for text, _ in summary_pairs]
        text, _ = _generate_commit_from_summaries(
            summary_texts,
            hint,
            chosen_model,
            single_line,
            subject_max,
            chosen_language,
            client,
        )
    else:
        input_items = _build_responses_input(diff, hint)

        resp = client.responses.create(
            model=chosen_model,
            instructions=_instructions(single_line, subject_max, chosen_language),
            input=input_items,
        )

        text = (resp.output_text or "").strip()

    if not text:
        raise RuntimeError("An empty commit message was generated.")
    return text


def generate_commit_message_with_info(
    diff: str,
    hint: str | None,
    model: str | None,
    single_line: bool = False,
    subject_max: int | None = None,
    language: str | None = None,
    chunk_tokens: int | None = 0,
    /,
) -> CommitMessageResult:
    """Return the OpenAI GPT call result together with debugging information.

    Returns
    -------
    CommitMessageResult
        The generated message, token usage, and prompt/response text.
    """

    chosen_model: str = _resolve_model(model)
    chosen_language: str = _resolve_language(language)
    api_key = environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("The OPENAI_API_KEY environment variable is required.")

    client = OpenAI(api_key=api_key)

    normalized_chunk_tokens = 0 if chunk_tokens is None else chunk_tokens

    if normalized_chunk_tokens != -1:
        hunks = _split_diff_into_hunks(diff)
        if normalized_chunk_tokens == 0:
            chunks = ["".join(hunks) if hunks else diff]
        elif normalized_chunk_tokens > 0:
            chunks = _build_diff_chunks(
                hunks,
                normalized_chunk_tokens,
                chosen_model,
            )
        else:
            chunks = ["".join(hunks) if hunks else diff]

        summary_pairs = _summarise_diff_chunks(
            chunks,
            chosen_model,
            client,
        )
        summary_texts = [text for text, _ in summary_pairs]
        response_text, final_resp = _generate_commit_from_summaries(
            summary_texts,
            hint,
            chosen_model,
            single_line,
            subject_max,
            chosen_language,
            client,
        )

        total_tokens: int | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        if final_resp.usage:
            total_tokens = (total_tokens or 0) + (final_resp.usage.total_tokens or 0)
            prompt_tokens = (prompt_tokens or 0) + (final_resp.usage.input_tokens or 0)
            completion_tokens = (completion_tokens or 0) + (
                final_resp.usage.output_tokens or 0
            )

        for _, resp in summary_pairs:
            usage = resp.usage
            if usage is None:
                continue
            total_tokens = (total_tokens or 0) + (usage.total_tokens or 0)
            prompt_tokens = (prompt_tokens or 0) + (usage.input_tokens or 0)
            completion_tokens = (completion_tokens or 0) + (usage.output_tokens or 0)

        combined_prompt = _build_combined_prompt(
            "\n".join(summary_texts),
            hint,
            "Combined summaries (English)",
        )

        response_id: str | None = final_resp.id

    else:
        combined_prompt = _build_combined_prompt(diff, hint)
        input_items = _build_responses_input(diff, hint)

        resp = client.responses.create(
            model=chosen_model,
            instructions=_instructions(single_line, subject_max, chosen_language),
            input=input_items,
        )

        response_text = (resp.output_text or "").strip()
        response_id = resp.id
        usage = resp.usage
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        total_tokens: int | None = None
        if usage is not None:
            total_tokens = usage.total_tokens
            prompt_tokens = usage.input_tokens
            completion_tokens = usage.output_tokens

    if not response_text:
        raise RuntimeError("An empty commit message was generated.")

    return CommitMessageResult(
        message=response_text,
        model=chosen_model,
        prompt=combined_prompt,
        response_text=response_text,
        response_id=response_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
