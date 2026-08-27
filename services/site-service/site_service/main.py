from contextlib import asynccontextmanager

from fastapi import FastAPI
from shared import config
from shared.config import API_V1_PREFIX

from site_service.db import get_db, init_app_db
from site_service.routers.calibration import router as calibration_router
from site_service.routers.configuration import router as configuration_router
from site_service.routers.detection import router as detection_router
from site_service.routers.file import router as file_router
from site_service.routers.site import router as site_router
from site_service.routers.source import router as source_router
from site_service.routers.violation import router as violation_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Checked before anything else, and before init_app_db creates files on disk.
    # Without this an unloaded environment only surfaces on the first upload, as an
    # "Invalid endpoint:" ValueError from deep inside boto3.
    missing = config.missing_s3_settings()
    if missing:
        raise RuntimeError(
            f"Object storage is not configured: {', '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} empty. Load the environment "
            "before starting, e.g. `set -a; source .env; set +a`."
        )
    init_app_db()
    yield


app = FastAPI(title="site-service", lifespan=lifespan)
app.include_router(site_router, prefix=API_V1_PREFIX)
app.include_router(calibration_router, prefix=API_V1_PREFIX)
app.include_router(configuration_router, prefix=API_V1_PREFIX)
app.include_router(file_router, prefix=API_V1_PREFIX)
app.include_router(source_router, prefix=API_V1_PREFIX)
app.include_router(detection_router, prefix=API_V1_PREFIX)
app.include_router(violation_router, prefix=API_V1_PREFIX)


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}

__all__ = ["app", "get_db"]
