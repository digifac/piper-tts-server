# Contributing to piper-tts-server

Thanks for your interest in contributing! This document describes how to set up a development environment, run the test suite and linter, and submit changes.

## Development setup

The project uses a `src/` layout with [PEP 621](https://peps.python.org/pep-0621/) metadata in [`pyproject.toml`](./pyproject.toml). Python **3.10+** is required.

```bash
git clone https://github.com/digifac/piper-tts-server.git
cd piper-tts-server

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # CPU inference + dev/test tooling (pytest, httpx, ruff)
# or, for local GPU inference:
pip install -e ".[gpu,dev]"    # adds onnxruntime-gpu (you still need cuDNN/cuBLAS at runtime)
```

Run the server locally:

```bash
python -m piper_tts_server     # listens on 0.0.0.0:8000 by default
```

Configuration is read from environment variables with the `TTS_` prefix or a `.env` file — see [`.env.example`](./.env.example).

## Linting and tests

Before submitting any change, make sure both checks pass (this mirrors what CI runs):

```bash
ruff check .     # lint
pytest -q        # test suite (CPU path; onnxruntime-gpu is not required)
```

- **Ruff** handles linting. Keep the codebase free of new warnings.
- **Pytest** covers the device-selection logic and the HTTP API. Tests run against the CPU inference path, so no GPU or `onnxruntime-gpu` install is needed to contribute.

## Submitting changes

1. Fork the repository and create a feature branch from `main`.
2. Make your changes with clear, focused commits (Conventional Commits style — e.g. `feat:`, `fix:`, `docs:` — is appreciated).
3. Run `ruff check .` and `pytest -q`; both must pass.
4. Open a pull request against `main` describing the change and linking any related issue.

## License

By contributing, you agree that your contributions will be licensed under the [GNU General Public License v3.0 or later](https://www.gnu.org/licenses/gpl-3.0.html) (GPL-3.0-or-later), consistent with upstream Piper TTS. See [`LICENSE`](./LICENSE).
