"""Command-line entrypoint for piper-tts-server."""

from __future__ import annotations

import logging


def main() -> None:  # pragma: no cover - thin wrapper around uvicorn
    """Run the server with uvicorn using configured host/port."""
    import uvicorn

    from .app import app
    from .config import get_settings

    settings = get_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":  # pragma: no cover
    main()
