"""Development entrypoint.

In production run the ASGI app directly::

    uvicorn app.main:app --host 0.0.0.0 --port 8000 --ws-max-size 16777216
"""

from __future__ import annotations

import os

import uvicorn

from app.core.config import get_settings


def main() -> None:
    settings = get_settings()

    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=not settings.is_production,
        # Must be >= AUDIO__MAX_BYTES or large recordings are dropped by the
        # WebSocket layer before the app ever sees them.
        ws_max_size=settings.audio.max_bytes + 1024 * 1024,
        log_config=None,  # logging is configured in app.core.logging
    )


if __name__ == "__main__":
    main()
