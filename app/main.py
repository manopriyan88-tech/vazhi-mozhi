import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.core.config import settings
from app.routers.demo import router as demo_router
from app.routers.keys import router as keys_router
from app.routers.stream import router as stream_router

app = FastAPI(
    title="Adaptive API Kitchen & Multi-Agent Orchestrator",
    description="Dynamic tool-generating Gemini agent backend with real-time WebSocket streaming.",
    version="0.1.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(demo_router, prefix="/api/demo", tags=["Demo"])
app.include_router(keys_router, prefix="/api/keys", tags=["API Kitchen Vault"])
app.include_router(stream_router)


@app.get("/", include_in_schema=False)
async def serve_index():
    """Serves the interactive frontend UI dashboard."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_path = os.path.join(base_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"message": "Adaptive API Kitchen & Multi-Agent Orchestrator"}


@app.get("/frontend_bridge.js", include_in_schema=False)
async def serve_frontend_bridge():
    """Serves the client bridge library."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bridge_path = os.path.join(base_dir, "frontend_bridge.js")
    if os.path.exists(bridge_path):
        return FileResponse(bridge_path, media_type="application/javascript")
    return {"error": "frontend_bridge.js not found"}


@app.get("/health", tags=["Health"])
async def health_check():
    """Service health check endpoint."""
    return {
        "status": "healthy",
        "env": settings.APP_ENV,
        "model": settings.GEMINI_MODEL,
    }

