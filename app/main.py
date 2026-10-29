"""Application factory and ASGI entrypoint.

Run with::

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import health, voice
from app.container import ServiceContainer
from app.core.config import Settings, get_settings
from app.core.context import bind_context
from app.core.errors import VoiceAssistantError
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)

TRACE_HEADER = "X-Trace-Id"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.logging)

    container = ServiceContainer(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "Starting %s v%s", settings.app_name, settings.version
        )
        await container.startup()
        try:
            yield
        finally:
            await container.shutdown()

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        lifespan=lifespan,
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
        openapi_url=None if settings.is_production else "/openapi.json",
    )
    app.state.container = container
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=[TRACE_HEADER],
    )

    @app.middleware("http")
    async def trace_requests(request: Request, call_next):
        incoming = request.headers.get(TRACE_HEADER)

        with bind_context(trace_id=incoming) as trace_id:
            response = await call_next(request)

        response.headers[TRACE_HEADER] = trace_id
        return response

    @app.exception_handler(VoiceAssistantError)
    async def handle_domain_error(
        request: Request, exc: VoiceAssistantError
    ) -> JSONResponse:
        logger.warning("Domain error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=400,
            content={"error": {"code": exc.code, "message": exc.client_message}},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": {"code": "internal_error", "message": "Something went wrong."}
            },
        )

    app.include_router(health.router)
    app.include_router(voice.router, prefix=settings.api_prefix)

    return app


app = create_app()
