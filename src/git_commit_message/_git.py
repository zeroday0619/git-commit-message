"""Git-related helper functions.

Provides repository root discovery, extraction of staged changes, and
creating commits from a message.
"""

from __future__ import annotations

from pathlib import Path
from subprocess import CalledProcessError, check_call, check_output, run


def get_repo_root(
    cwd: Path | None = None,
    /,
) -> Path:
    """Find the repository root from the current working directory.

    Parameters
    ----------
    cwd
        Starting directory for the search. Defaults to the current working directory.

    Returns
    -------
    Path
        The repository root path.
    """

    start: Path = cwd or Path.cwd()
    try:
        out: bytes = check_output(
            [
                "git",
                "rev-parse",
                "--show-toplevel",
            ],
            cwd=str(start),
        )
    except CalledProcessError as exc:  # noqa: TRY003
        raise RuntimeError("Not a Git repository.") from exc

    root = Path(out.decode().strip())
    return root


def has_staged_changes(
    cwd: Path,
    /,
) -> bool:
    """Check whether there are staged changes."""

    try:
        check_call(
            ["git", "diff", "--cached", "--quiet", "--exit-code"],
            cwd=str(cwd),
        )
        return False
    except CalledProcessError:
        return True


def get_staged_diff(
    cwd: Path,
    /,
) -> str:
    """Return the staged changes as diff text."""

    out: bytes = check_output(
        [
            "git",
            "diff",
            "--cached",
            "--patch",
            "--minimal",
            "--no-color",
        ],
        cwd=str(cwd),
    )
    return out.decode()


def commit_with_message(
    message: str,
    edit: bool,
    cwd: Path,
    /,
) -> int:
    """Create a commit with the given message.

    Parameters
    ----------
    message
        Commit message.
    edit
        If True, use the `--edit` flag to open an editor for amendments.
    cwd
        Git working directory.

    Returns
    -------
    int
        The subprocess exit code.
    """

    cmd: list[str] = ["git", "commit", "-m", message]
    if edit:
        cmd.append("--edit")

    try:
        completed = run(cmd, cwd=str(cwd), check=False)
        return int(completed.returncode)
    except OSError as exc:  # e.g., editor launch failure, etc.
        raise RuntimeError(f"Failed to run 'git commit': {exc}") from exc
