from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, model_validator


class SourceKind(str, Enum):
    VIDEO = "video"
    STREAM = "stream"


class SourceStatus(str, Enum):
    CREATED = "created"
    # Stream states — they describe a live feed, which is why they belong to a source
    # rather than to the durable site above it.
    ACTIVE = "active"
    DEGRADED = "degraded"
    # Video states.
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SourceMetadata(BaseModel):
    total_frames: int | None = None
    fps: float | None = None
    duration_seconds: float | None = None
    resolution: dict | None = None


class SourceCreate(BaseModel):
    # Strict, so a body carrying site_id is rejected rather than quietly ignored —
    # site_id is owned by the path.
    model_config = ConfigDict(extra="forbid")

    kind: SourceKind
    # Exactly one of these, decided by kind. Mirrors the CHECK on site_sources;
    # validating here turns a bad body into a 422 instead of a 500 from the database.
    stream_url: str | None = None
    file_id: str | None = None

    @model_validator(mode="after")
    def _exactly_one_source_for_the_kind(self):
        if self.kind is SourceKind.VIDEO and (self.file_id is None or self.stream_url is not None):
            raise ValueError("a video source requires file_id and must not carry stream_url")
        if self.kind is SourceKind.STREAM and (
            self.stream_url is None or self.file_id is not None
        ):
            raise ValueError("a stream source requires stream_url and must not carry file_id")
        return self


class SourceResponse(BaseModel):
    id: str
    site_id: str
    version: int
    kind: SourceKind
    stream_url: str | None = None
    file_id: str | None = None
    status: SourceStatus
    metadata: SourceMetadata | None = None
    created_at: datetime
    updated_at: datetime
