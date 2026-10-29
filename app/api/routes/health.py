"""Liveness and readiness probes.

``/health/live``  - process is up (never touches dependencies).
``/health/ready`` - prompts loaded, model loaded, store and agent reachable.
                    Returns 503 when not ready, so orchestrators hold traffic.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status

from app.api.deps import ContainerDep, SettingsDep

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health(settings: SettingsDep, container: ContainerDep) -> dict[str, object]:
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.version,
        "environment": settings.environment.value,
        "uptime_seconds": round(container.uptime_seconds, 1),
    }


@router.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/health/ready")
async def ready(container: ContainerDep, response: Response) -> dict[str, object]:
    report = await container.readiness()

    if not report["ready"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return report


@router.get("/metrics", include_in_schema=False)
async def metrics(settings: SettingsDep, container: ContainerDep) -> Response:
    """Prometheus scrape target. Keep it on the internal network."""

    if not settings.metrics_enabled:
        raise HTTPException(status_code=404)

    return Response(
        container.metrics.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
