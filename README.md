# git-commit-message

Staged changes -> GPT or Ollama commit message generator.

[![asciicast](https://asciinema.org/a/jk0phFqNnc5vaCiIZEYBwZOyN.svg)](https://asciinema.org/a/jk0phFqNnc5vaCiIZEYBwZOyN)

## Install (PyPI)

Install the latest released version from PyPI:

```sh
# User environment (recommended)
python -m pip install --user git-commit-message

# Or system/virtualenv as appropriate
python -m pip install git-commit-message

# Or with pipx for isolated CLI installs
pipx install git-commit-message

# Upgrade to the newest version
python -m pip install --upgrade git-commit-message
```

Quick check:

```sh
git-commit-message --help
```

Set your API key (POSIX sh):

```sh
export OPENAI_API_KEY="sk-..."
```

Or for the Google provider:

```sh
export GOOGLE_API_KEY="..."
```

Note (fish): In fish, set it as follows.

```fish
set -x OPENAI_API_KEY "sk-..."
```

### Ollama (local models)

If you prefer to use Ollama for local model inference without API costs:

1. Install Ollama from https://ollama.ai
2. Start the Ollama server:

```sh
ollama serve
```

3. Pull a model in another terminal:

```sh
ollama pull mistral
```

Then use git-commit-message with the `--provider ollama` option:

```sh
git-commit-message --provider ollama
```

Or set it as the default provider:

```sh
export GIT_COMMIT_MESSAGE_PROVIDER=ollama
export OLLAMA_MODEL=mistral
```

## Install (editable)

```sh
python -m pip install -e .
```

## Usage

- Print commit message only:

```sh
git add -A
git-commit-message "optional extra context about the change"
```

- Force single-line subject only:

```sh
git-commit-message --one-line "optional context"
```

- Select provider (default: openai):

```sh
git-commit-message --provider openai "optional context"
```

- Select provider (Google Gemini via google-genai):

```sh
git-commit-message --provider google "optional context"
```

- Limit subject length (default 72):

```sh
git-commit-message --one-line --max-length 50 "optional context"
```

- Chunk long diffs by token budget (0 = single chunk + summary, -1 = disable chunking):

```sh
# force a single summary pass over the whole diff (default)
git-commit-message --chunk-tokens 0 "optional context"

# chunk the diff into ~4000-token pieces before summarising
git-commit-message --chunk-tokens 4000 "optional context"

# disable summarisation and use the legacy one-shot prompt
git-commit-message --chunk-tokens -1 "optional context"
```

- Commit immediately with editor:

```sh
git-commit-message --commit --edit "refactor parser for speed"
```

- Print debug info (prompt/response + token usage):

```sh
git-commit-message --debug "optional context"
```

- Select output language/locale (default: en-GB):

```sh
# American English
git-commit-message --language en-US "optional context"

# Korean
git-commit-message --language ko-KR

# Japanese
git-commit-message --language ja-JP
```

- Select AI provider (OpenAI or Ollama):

```sh
# Use Ollama (requires running `ollama serve`)
git-commit-message --provider ollama "optional context"

# Use OpenAI (default)
git-commit-message --provider openai "optional context"

# Or set via environment variable (default: openai)
export GIT_COMMIT_MESSAGE_PROVIDER=ollama
git-commit-message "optional context"
```

- Configure Ollama host (if running on a different machine):

```sh
git-commit-message --provider ollama --ollama-host http://192.168.1.100:11434
```

Notes:

- The model is instructed to write using the selected language/locale.
- In multi-line mode, the only allowed label ("Rationale:") is also translated into the target language.

Environment:

- `OPENAI_API_KEY`: required when provider is `openai`
- `GOOGLE_API_KEY`: required when provider is `google`
- `GIT_COMMIT_MESSAGE_PROVIDER`: optional (default: `openai`). `--provider` overrides this value.
- `GIT_COMMIT_MESSAGE_MODEL`: optional model override (defaults: `openai` -> `gpt-5-mini`, `google` -> `gemini-2.5-flash`, `ollama` -> `gpt-oss:20b`)
- `OPENAI_MODEL`: optional OpenAI-only model override
- `OLLAMA_MODEL`: optional Ollama-only model override
- `OLLAMA_HOST`: optional Ollama server URL (default: `http://localhost:11434`)
- `GIT_COMMIT_MESSAGE_LANGUAGE`: optional (default: `en-GB`)
- `GIT_COMMIT_MESSAGE_CHUNK_TOKENS`: optional token budget per diff chunk (default: 0 = single chunk + summary; -1 disables summarisation)

Notes:

- If token counting fails for your provider while chunking, try `--chunk-tokens 0` (default) or `--chunk-tokens -1`.

## AI‑generated code notice

Parts of this project were created with assistance from AI tools (e.g. large language models).
All AI‑assisted contributions were reviewed and adapted by maintainers before inclusion.
If you need provenance for specific changes, please refer to the Git history and commit messages.
