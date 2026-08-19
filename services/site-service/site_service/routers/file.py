from typing import Annotated

import duckdb
from fastapi import APIRouter, Depends, HTTPException
from shared import config
from shared.models.file import FileCreate, FileResponse, FileUploadResponse

from site_service import service
from site_service.db import get_db
from site_service.storage import Storage

# Files are a top-level resource: a video belongs to a site, but a calibration or an
# evidence frame does not, so nesting them under one would be wrong.
router = APIRouter(prefix="/files", tags=["files"])
DbConnection = Annotated[duckdb.DuckDBPyConnection, Depends(get_db)]


@router.post("", response_model=FileUploadResponse, status_code=201)
def create_file(data: FileCreate, con: DbConnection, storage: Storage):
    """Reserve a key and return a URL the client PUTs the bytes to directly.

    Nothing is uploaded yet — the row stays `pending` until /complete confirms it.
    """
    # Checked before a URL exists: an upload URL is a spending authorisation, so the
    # cap has to be applied at the point it is issued, not at /complete when the
    # bytes have already been paid for.
    if data.size_bytes > config.S3_MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the maximum upload size of {config.S3_MAX_UPLOAD_BYTES} bytes",
        )
    return service.create_file(con, storage, data)


@router.post("/{file_id}/complete", response_model=FileResponse)
def complete_upload(file_id: str, con: DbConnection, storage: Storage):
    # Checked first so an unknown id is a 404 rather than being reported as a
    # missing object, which would send the client looking in the wrong place.
    if service.get_file(con, storage, file_id) is None:
        raise HTTPException(status_code=404, detail="File not found")
    confirmed = service.confirm_upload(con, storage, file_id)
    if confirmed is None:
        # The row exists but the bytes never arrived. 409 rather than 404: the
        # request is fine, the resource just is not in the state it claims.
        raise HTTPException(status_code=409, detail="Upload not found in storage")
    return confirmed


@router.get("/{file_id}", response_model=FileResponse)
def get_file(file_id: str, con: DbConnection, storage: Storage):
    file = service.get_file(con, storage, file_id)
    if file is None:
        raise HTTPException(status_code=404, detail="File not found")
    return file
