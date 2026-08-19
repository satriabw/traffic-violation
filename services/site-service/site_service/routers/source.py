from typing import Annotated

import duckdb
from fastapi import APIRouter, Depends, HTTPException
from shared.models.file import FileType
from shared.models.source import SourceCreate, SourceKind, SourceResponse

from site_service import service
from site_service.db import get_db
from site_service.file_reference import raise_for_unusable_file

# Nested under a site: a source has no meaning without one, and site_id is never taken
# from the request body.
router = APIRouter(prefix="/sites/{site_id}/sources", tags=["sources"])
DbConnection = Annotated[duckdb.DuckDBPyConnection, Depends(get_db)]


def require_site(con: duckdb.DuckDBPyConnection, site_id: str) -> None:
    if service.get_site(con, site_id) is None:
        raise HTTPException(status_code=404, detail="Site not found")


def validate_source(con: duckdb.DuckDBPyConnection, data: SourceCreate) -> None:
    """A video source must point at a confirmed upload. A stream has no file to check —
    its address is a URL this service never resolves."""
    if data.kind is SourceKind.VIDEO:
        raise_for_unusable_file(con, data.file_id, FileType.VIDEO)


@router.post("", response_model=SourceResponse, status_code=201)
def create_source(site_id: str, data: SourceCreate, con: DbConnection):
    """Point the site at something new. Appends a version rather than replacing, so
    what the site used to be pointed at stays on record."""
    require_site(con, site_id)
    validate_source(con, data)
    return service.create_source(con, site_id, data)


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
