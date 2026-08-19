from datetime import datetime

from pydantic import BaseModel, ConfigDict

from shared.models.source import SourceCreate, SourceResponse


class SiteCreate(BaseModel):
    # Strict: a client still sending the old top-level `mode`/`url` gets a 422 rather
    # than a silently source-less site.
    model_config = ConfigDict(extra="forbid")

    name: str
    # Optional: a site can exist before anything is pointed at it, and the source can
    # be added or changed later. Supplying one here creates version 1 in the same
    # request — sugar over POST /sites/{id}/sources, not a second code path.
    source: SourceCreate | None = None


class SiteResponse(BaseModel):
    """A durable camera location. Nothing per-run lives here — see SourceResponse."""

    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    # The active (highest-version) source, embedded so the common read is one request.
    # None until something is attached.
    source: SourceResponse | None = None


class SiteListResponse(BaseModel):
    items: list[SiteResponse]
    total: int
    limit: int
    offset: int
