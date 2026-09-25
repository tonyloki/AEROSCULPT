"""
FastAPI Server Entrypoint for AeroSculpt System.
Mounts REST API routes and serves interactive frontend UI.
"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router as api_router

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="AeroSculpt System",
    description="Single-Pass UAV Video to Accurate Georeferenced 3D Model Generation System",
    version="1.0.0"
)

# Enable CORS for external GIS integrations & client requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include REST API
app.include_router(api_router)

# Mount frontend web dashboard static assets
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    # Server launcher configuration
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
