"""
SentinelEdge deployment server for Hugging Face Spaces.
Adds static file serving to the demo backend app.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# The demo backend app already has /api/* and /ws/* routes
from demo.backend.main import app
from hub.server import app as hub_app

# Mount hub server at /hub
app.mount("/hub", hub_app)

# Serve built React frontend as static files
static_dir = Path(__file__).parent / "static"

if static_dir.exists():
    @app.get("/")
    async def serve_index() -> FileResponse:
        return FileResponse(str(static_dir / "index.html"))

    # Catch-all for SPA client-side routing (must be after API/WS routes)
    @app.get("/{path:path}")
    async def serve_spa(path: str) -> FileResponse:
        file_path = static_dir / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(static_dir / "index.html"))


port = int(os.environ.get("PORT", 7860))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)
