"""API routers package."""

from app.routers.demo import router as demo_router
from app.routers.keys import router as keys_router
from app.routers.stream import router as stream_router

__all__ = ["demo_router", "keys_router", "stream_router"]


