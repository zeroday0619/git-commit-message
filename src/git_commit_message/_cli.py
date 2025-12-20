"""Command-line interface entry point.

Collect staged changes from the repository and call an LLM provider
to generate a commit message, or create a commit straight away.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from os import environ
from pathlib import Path
from sys import exit as sys_exit
from sys import stderr
from typing import Final

from ._git import (
    commit_with_message,
    get_repo_root,
    get_staged_diff,
    has_staged_changes,
)
from ._llm import (
    CommitMessageResult,
    UnsupportedProviderError,
    generate_commit_message,
    generate_commit_message_with_info,
)


def _env_chunk_tokens_default() -> int | None:
    """Return chunk token default from env if valid, else None."""

    raw: str | None = environ.get("GIT_COMMIT_MESSAGE_CHUNK_TOKENS")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _build_parser() -> ArgumentParser:
    """Create the CLI argument parser.

    Returns
    -------
    ArgumentParser
        A configured argument parser.
    """

    parser: ArgumentParser = ArgumentParser(
        prog="git-commit-message",
        description=(
            "Generate a commit message based on the staged changes."
        ),
    )

    parser.add_argument(
        "description",
        nargs="?",
        help="Optional auxiliary description of the changes.",
    )

    parser.add_argument(
        "--commit",
        action="store_true",
        help="Commit immediately with the generated message.",
    )

    parser.add_argument(
        "--edit",
        action="store_true",
        help="Open an editor to amend the message before committing. Use with '--commit'.",
    )

    parser.add_argument(
        "--provider",
        default=None,
        help=(
            "LLM provider to use (default: openai). "
            "You may also set GIT_COMMIT_MESSAGE_PROVIDER. "
            "The CLI flag overrides the environment variable."
        ),
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name to use. If unspecified, uses GIT_COMMIT_MESSAGE_MODEL or a provider-specific default (openai: gpt-5-mini; google: gemini-2.5-flash)."
        ),
    )

    parser.add_argument(
        "--language",
        dest="language",
        default=None,
        help=(
            "Target language/locale IETF tag for the output (default: en-GB). "
            "You may also set GIT_COMMIT_MESSAGE_LANGUAGE."
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the request/response and token usage.",
    )

    parser.add_argument(
        "--one-line",
        dest="one_line",
        action="store_true",
        help="Use only a single-line subject.",
    )

    parser.add_argument(
        "--max-length",
        dest="max_length",
        type=int,
        default=None,
        help="Maximum subject (first line) length (default: 72).",
    )

    parser.add_argument(
        "--chunk-tokens",
        dest="chunk_tokens",
        type=int,
        default=None,
        help=(
            "Target token budget per diff chunk. "
            "0 forces a single chunk with summarisation; -1 disables summarisation (legacy one-shot). "
            "If omitted, uses GIT_COMMIT_MESSAGE_CHUNK_TOKENS when set (default: 0)."
        ),
    )

    return parser


def _run(
    args: Namespace,
    /,
) -> int:
    """Main execution logic.

    Parameters
    ----------
    args
        Parsed CLI arguments.

    Returns
    -------
    int
        Process exit code. 0 indicates success; any other value indicates failure.
    """

    repo_root: Path = get_repo_root()

    if not has_staged_changes(repo_root):
        print("No staged changes. Run 'git add' and try again.", file=stderr)
        return 2

    diff_text: str = get_staged_diff(repo_root)

    hint: str | None = args.description if isinstance(args.description, str) else None

    chunk_tokens: int | None = args.chunk_tokens
    if chunk_tokens is None:
        chunk_tokens = _env_chunk_tokens_default()
    if chunk_tokens is None:
        chunk_tokens = 0

    result: CommitMessageResult | None = None
    try:
        if args.debug:
            result = generate_commit_message_with_info(
                diff_text,
                hint,
                args.model,
                getattr(args, "one_line", False),
                getattr(args, "max_length", None),
                getattr(args, "language", None),
                chunk_tokens,
                getattr(args, "provider", None),
            )
            message = result.message
        else:
            message = generate_commit_message(
                diff_text,
                hint,
                args.model,
                getattr(args, "one_line", False),
                getattr(args, "max_length", None),
                getattr(args, "language", None),
                chunk_tokens,
                getattr(args, "provider", None),
            )
    except UnsupportedProviderError as exc:
        print(str(exc), file=stderr)
        return 3
    except Exception as exc:  # noqa: BLE001 - to preserve standard output messaging
        print(f"Failed to generate commit message: {exc}", file=stderr)
        return 3

    # Option: force single-line message
    if getattr(args, "one_line", False):
        # Use the first non-empty line only
        for line in (ln.strip() for ln in message.splitlines()):
            if line:
                message = line
                break
        else:
            message = ""

    if not args.commit:
        if args.debug and result is not None:
            # Print debug information
            print(f"==== {result.provider} Usage ====")
            print(f"provider: {result.provider}")
            print(f"model: {result.model}")
            print(f"response_id: {getattr(result, 'response_id', '(n/a)')}")
            if result.total_tokens is not None:
                print(
                    f"tokens: prompt={result.prompt_tokens} completion={result.completion_tokens} total={result.total_tokens}"
                )
            else:
                print("tokens: (provider did not return usage)")
            print("\n==== Prompt ====")
            print(result.prompt)
            print("\n==== Response ====")
            print(result.response_text)
            print("\n==== Commit Message ====")
            print(message)
        else:
            print(message)
        return 0

    if args.debug and result is not None:
        # Also print debug info before commit
        print(f"==== {result.provider} Usage ====")
        print(f"provider: {result.provider}")
        print(f"model: {result.model}")
        print(f"response_id: {getattr(result, 'response_id', '(n/a)')}")
        if result.total_tokens is not None:
            print(
                f"tokens: prompt={result.prompt_tokens} completion={result.completion_tokens} total={result.total_tokens}"
            )
        else:
            print("tokens: (provider did not return usage)")
        print("\n==== Prompt ====")
        print(result.prompt)
        print("\n==== Response ====")
        print(result.response_text)
        print("\n==== Commit Message ====")
        print(message)

    if args.edit:
        rc: int = commit_with_message(message, True, repo_root)
    else:
        rc = commit_with_message(message, False, repo_root)

    return rc


def main() -> None:
    """Script entry point.

    Parse command-line arguments, delegate to the execution logic, and exit with its code.
    """

    parser: Final[ArgumentParser] = _build_parser()
    args: Namespace = parser.parse_args()

    if args.edit and not args.commit:
        print("'--edit' must be used together with '--commit'.", file=stderr)
        sys_exit(2)

    code: int = _run(args)
    sys_exit(code)
