from typing import Annotated

import duckdb
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from shared.models.site import SiteCreate, SiteListResponse, SiteResponse
from shared.models.source import SourceKind, SourceStatus

from site_service import service
from site_service.db import get_db
from site_service.routers.source import validate_source
from site_service.storage import Storage
from site_service.video import Probe

router = APIRouter(prefix="/sites", tags=["sites"])
DbConnection = Annotated[duckdb.DuckDBPyConnection, Depends(get_db)]


@router.post("", response_model=SiteResponse, status_code=201)
def create_site(data: SiteCreate, con: DbConnection, storage: Storage, probe: Probe):
    """Create a site, optionally pointing it at something in the same request.

    A site with no source is valid — the stream url can be added or changed later.
    """
    metadata = None
    if data.source is not None:
        # Same validation and probing the dedicated source endpoint applies, so the
        # two entry points cannot drift apart.
        metadata = validate_source(con, storage, probe, data.source)
    return service.create_site(con, data, metadata)


@router.get("", response_model=SiteListResponse)
def list_sites(
    con: DbConnection,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    kind: SourceKind | None = None,
    status: SourceStatus | None = None,
):
    """kind and status describe the site's *active* source, not its whole history."""
    return service.list_sites(
        con,
        limit=limit,
        offset=offset,
        kind=kind.value if kind else None,
        status=status.value if status else None,
    )


@router.get("/{site_id}", response_model=SiteResponse)
def get_site(site_id: str, con: DbConnection):
    site = service.get_site(con, site_id)
    if site is None:
        raise HTTPException(status_code=404, detail="Site not found")
    return site


@router.delete("/{site_id}", status_code=204)
def delete_site(site_id: str, con: DbConnection):
    deleted = service.delete_site(con, site_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Site not found")
    return Response(status_code=204)
