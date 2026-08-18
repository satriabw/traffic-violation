from contextlib import asynccontextmanager

from fastapi import FastAPI
from shared.config import API_V1_PREFIX

from site_service.db import get_db, init_app_db
from site_service.routers.calibration import router as calibration_router
from site_service.routers.site import router as site_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_app_db()
    yield


app = FastAPI(title="site-service", lifespan=lifespan)
app.include_router(site_router, prefix=API_V1_PREFIX)
app.include_router(calibration_router, prefix=API_V1_PREFIX)


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}

__all__ = ["app", "get_db"]
