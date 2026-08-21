"""Shared router factory for calibrations and configurations.

The two resources are the same endpoint shape over the same service functions, so the
routes are built once. Each caller supplies its own response model, so the wire
contract stays per-resource and the OpenAPI schema names the real thing.
"""

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from shared.models.file import FileType

from site_service import service
from site_service.db import get_db
from site_service.file_reference import raise_for_unusable_file

DbConnection = Annotated[sqlite3.Connection, Depends(get_db)]

def build_router(
    *,
    resource: str,
    table: str,
    file_type: FileType,
    create_model: type[BaseModel],
    response_model: type[BaseModel],
) -> APIRouter:
    # Nested under a site: one of these has no meaning without the other, and site_id
    # is never taken from the request body.
    router = APIRouter(prefix=f"/sites/{{site_id}}/{resource}", tags=[resource])
    singular = resource.rstrip("s")

    def _require_site(con: sqlite3.Connection, site_id: str) -> None:
        if service.get_site(con, site_id) is None:
            raise HTTPException(status_code=404, detail="Site not found")

    @router.post("", response_model=response_model, status_code=201)
    def create(site_id: str, data: create_model, con: DbConnection):
        # Site first: an unknown site is a 404 rather than the foreign key blowing up
        # as a 500, and it outranks any problem with the file.
        _require_site(con, site_id)
        raise_for_unusable_file(con, data.file_id, file_type)
        return service.create_versioned_doc(con, table, site_id, data.file_id)

    @router.get("", response_model=response_model)
    def get_active(site_id: str, con: DbConnection):
        f"""Return the site's active {singular} — the highest version."""
        _require_site(con, site_id)
        doc = service.get_active_version(con, table, site_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"{singular.capitalize()} not found")
        return doc

    @router.get("/{doc_id}", response_model=response_model)
    def get_one(site_id: str, doc_id: str, con: DbConnection):
        _require_site(con, site_id)
        doc = service.get_version(con, table, site_id, doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"{singular.capitalize()} not found")
        return doc

    return router
