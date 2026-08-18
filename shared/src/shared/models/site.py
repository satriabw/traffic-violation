from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class SiteMode(str, Enum):
    VIDEO = "video"
    STREAM = "stream"


class SiteStatus(str, Enum):
    CREATED = "created"
    ACTIVE = "active"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEGRADED = "degraded"


class SiteMetadata(BaseModel):
    total_frames: int | None = None
    fps: float | None = None
    duration_seconds: float | None = None
    resolution: dict | None = None


class SiteCreate(BaseModel):
    name: str
    url: str
    mode: SiteMode


class SiteResponse(BaseModel):
    id: str
    name: str
    url: str
    mode: SiteMode
    status: SiteStatus
    created_at: datetime
    updated_at: datetime
    metadata: SiteMetadata | None = None


class SiteListResponse(BaseModel):
    items: list[SiteResponse]
    total: int
    limit: int
    offset: int
