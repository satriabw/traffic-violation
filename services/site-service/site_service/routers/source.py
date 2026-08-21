import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from shared.models.file import FileType
from shared.models.source import (
    SourceCreate,
    SourceKind,
    SourceMetadata,
    SourceResponse,
)
from shared.video.probe import ProbeUnavailable, VideoUnreadable

from site_service import service
from site_service.db import get_db
from site_service.file_reference import raise_for_unusable_file
from site_service.storage import Storage
from site_service.video import Probe

# Nested under a site: a source has no meaning without one, and site_id is never taken
# from the request body.
router = APIRouter(prefix="/sites/{site_id}/sources", tags=["sources"])
DbConnection = Annotated[sqlite3.Connection, Depends(get_db)]


def require_site(con: sqlite3.Connection, site_id: str) -> None:
    if service.get_site(con, site_id) is None:
        raise HTTPException(status_code=404, detail="Site not found")


def validate_source(
    con: sqlite3.Connection, storage, probe, data: SourceCreate
) -> SourceMetadata | None:
    """Check the source is usable and, for a video, read its shape once.

    Both entry points that create a source go through here, so the inline form on
    POST /sites cannot drift from POST /sites/{id}/sources.

    A stream is neither checked nor probed: its address is a URL this service never
    resolves, it carries no index to read, and connecting to a live feed here would
    block the request on a camera that may not be up yet. Its metadata is filled in
    by whatever consumes the feed.
    """
    if data.kind is not SourceKind.VIDEO:
        return None

    raise_for_unusable_file(con, data.file_id, FileType.VIDEO)
    file = service.get_file(con, storage, data.file_id)
    try:
        # Reads the container header over range requests — a couple of megabytes
        # whatever the file's size — rather than downloading the object.
        return probe(file.download_url)
    except VideoUnreadable as exc:
        # The body named a file that is not a video we can read. Rejecting now beats
        # storing a source whose detection jobs would each fail later.
        raise HTTPException(
            status_code=422, detail="Referenced video could not be decoded"
        ) from exc
    except ProbeUnavailable as exc:
        # The video may be perfectly fine; we could not reach it. Not the client's
        # fault, and the same request may well succeed on a retry.
        raise HTTPException(
            status_code=502, detail="Could not read video metadata, try again"
        ) from exc


@router.post("", response_model=SourceResponse, status_code=201)
def create_source(
    site_id: str, data: SourceCreate, con: DbConnection, storage: Storage, probe: Probe
):
    """Point the site at something new. Appends a version rather than replacing, so
    what the site used to be pointed at stays on record."""
    require_site(con, site_id)
    metadata = validate_source(con, storage, probe, data)
    return service.create_source(con, site_id, data, metadata)


@router.get("", response_model=SourceResponse)
def get_active_source(site_id: str, con: DbConnection):
    require_site(con, site_id)
    source = service.get_active_source(con, site_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return source


@router.get("/{source_id}", response_model=SourceResponse)
def get_source(site_id: str, source_id: str, con: DbConnection):
    require_site(con, site_id)
    source = service.get_source(con, site_id, source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return source
