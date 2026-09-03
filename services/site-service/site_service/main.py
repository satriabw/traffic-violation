import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from shared import config
from shared.config import API_V1_PREFIX
from shared.db.violations import fail_pending_explanations
from shared.logging import configure_logging

from site_service.actor import start_actor, stop_actor
from site_service.db import get_db, init_app_db
from site_service.routers.calibration import router as calibration_router
from site_service.routers.configuration import router as configuration_router
from site_service.routers.detection import router as detection_router
from site_service.routers.file import router as file_router
from site_service.routers.site import router as site_router
from site_service.routers.source import router as source_router
from site_service.routers.violation import router as violation_router

# The name this service is known by, in the logs and in its own OpenAPI document.
SERVICE_NAME = "site-service"

configure_logging(SERVICE_NAME, config.LOG_LEVEL)
logger = logging.getLogger(__name__)


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
    # Anything left 'pending' belongs to a process that is gone: the actor's mailbox
    # lives in memory, so a violation accepted but unfinished when this service last
    # stopped has nothing to finish it. The row would otherwise promise an answer
    # forever to whoever is polling it.
    stranded = fail_pending_explanations(get_db())
    if stranded:
        logger.warning(
            "marked %d violation(s) failed: they were still awaiting an explanation "
            "when this service last stopped",
            stranded,
        )
    # Started here rather than lazily on the first request, so the thread exists before
    # anything can send to it and its lifetime is the app's — and so nothing can bring
    # one into existence as a side effect of serving a request.
    start_actor()
    yield
    # Drains what it can rather than dropping it; whatever it cannot is what the next
    # startup marks failed.
    stop_actor()


app = FastAPI(title=SERVICE_NAME, lifespan=lifespan)
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
