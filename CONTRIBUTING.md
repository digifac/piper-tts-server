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

## Making a release

Releases are beta (`0.x`) for now, driven by git tags — there is no automatic version bump on every push to `main`. The [`Release`](./.github/workflows/release.yml) workflow runs when you push a tag matching `v0.*`: it builds both Docker images (CPU and GPU), pushes them to [GHCR](https://ghcr.io/digifac/piper-tts-server), and creates the GitHub release with auto-generated notes plus pull commands.

To cut a new release:

```bash
# 1. (Optional) Bump the version in pyproject.toml if you want it to match the tag.
#    The image tags come from the git tag, not from this file — but keeping them in sync is nice.

# 2. Commit any pending changes on main and push.
git checkout main && git pull --rebase origin main
git push origin main

# 3. Create and push the tag (e.g. v0.1.0, then v0.1.1 / v0.2.0 ...).
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

Watch the run in **Actions → Release**: both image builds must succeed before the release job creates (or updates) the GitHub release at [`/releases`](https://github.com/digifac/piper-tts-server/releases). Re-pushing the same tag re-runs the workflow and `softprops/action-gh-release` updates the existing release in place rather than duplicating it.

> To support stable `v1.*+` releases later, widen the tag pattern at the top of `.github/workflows/release.yml`.

## License

By contributing, you agree that your contributions will be licensed under the [GNU General Public License v3.0 or later](https://www.gnu.org/licenses/gpl-3.0.html) (GPL-3.0-or-later), consistent with upstream Piper TTS. See [`LICENSE`](./LICENSE).
