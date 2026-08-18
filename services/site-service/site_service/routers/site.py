from typing import Annotated

import duckdb
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from shared.models.site import SiteCreate, SiteListResponse, SiteMode, SiteResponse, SiteStatus

from site_service import service
from site_service.db import get_db

router = APIRouter(prefix="/sites", tags=["sites"])
DbConnection = Annotated[duckdb.DuckDBPyConnection, Depends(get_db)]


@router.post("", response_model=SiteResponse, status_code=201)
def create_site(data: SiteCreate, con: DbConnection):
    return service.create_site(con, data)


@router.get("", response_model=SiteListResponse)
def list_sites(
    con: DbConnection,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    mode: SiteMode | None = None,
    status: SiteStatus | None = None,
):
    return service.list_sites(
        con,
        limit=limit,
        offset=offset,
        mode=mode.value if mode else None,
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
