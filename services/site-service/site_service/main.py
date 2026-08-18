from contextlib import asynccontextmanager

from fastapi import FastAPI

from site_service.db import get_db, init_app_db
from site_service.routers.site import router as site_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_app_db()
    yield


app = FastAPI(title="site-service", lifespan=lifespan)
app.include_router(site_router)

__all__ = ["app", "get_db"]
