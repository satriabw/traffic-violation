"""Turning an unusable file reference into the right HTTP error.

Shared by every endpoint that accepts a file_id — sites, calibrations, configurations
— so a video attached to a site and a calibration attached to a site fail the same way.
"""

import duckdb
from fastapi import HTTPException
from shared.models.file import FileType

from site_service import service

# `pending` is a 409 rather than a 422 because the request is well formed: the file is
# simply not ready yet, and the identical request succeeds once the upload completes.
# The other two are 422 — the body names something unusable.
_FILE_ERRORS = {
    "missing": (422, "Referenced file does not exist"),
    "pending": (409, "Referenced file has not been uploaded yet"),
    "wrong_type": (422, "Referenced file is not a {expected} file"),
}


def raise_for_unusable_file(
    con: duckdb.DuckDBPyConnection, file_id: str, expected_type: FileType
) -> None:
    reason = service.unusable_file_reason(con, file_id, expected_type)
    if reason is None:
        return
    status, detail = _FILE_ERRORS[reason]
    raise HTTPException(status_code=status, detail=detail.format(expected=expected_type.value))
