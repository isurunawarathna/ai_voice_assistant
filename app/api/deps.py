"""FastAPI dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request, WebSocket
from starlette.requests import HTTPConnection

from app.container import ServiceContainer
from app.core.config import Settings


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


def get_ws_container(websocket: WebSocket) -> ServiceContainer:
    return websocket.app.state.container


def get_app_settings(connection: HTTPConnection) -> Settings:
    """Settings the app was *built* with, not whatever the global cache holds."""

    return connection.app.state.settings


ContainerDep = Annotated[ServiceContainer, Depends(get_container)]
WSContainerDep = Annotated[ServiceContainer, Depends(get_ws_container)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]
