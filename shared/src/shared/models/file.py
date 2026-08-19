from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class FileType(str, Enum):
    CALIBRATION = "calibration"
    CONFIGURATION = "configuration"
    VIDEO = "video"
    EVIDENCE_FRAME = "evidence_frame"


class FileStatus(str, Enum):
    # A row exists but nothing has been confirmed in storage yet.
    PENDING = "pending"
    # HeadObject saw the object. Only now is the file safe to reference.
    UPLOADED = "uploaded"


class FileCreate(BaseModel):
    name: str
    type: FileType
    # Signed into the upload URL, so the client is bound to the size it declared.
    # Required: without it there is no cap on what a URL holder can push.
    size_bytes: int = Field(gt=0)
    # Signed into the upload URL when given, which means the client must send the
    # identical Content-Type header on its PUT.
    content_type: str | None = None


class FileResponse(BaseModel):
    id: str
    name: str
    # The object key. Deliberately no upload_url here — see FileUploadResponse.
    url: str
    type: FileType
    status: FileStatus
    content_type: str | None = None
    size_bytes: int | None = None
    created_at: datetime
    updated_at: datetime
    # Absent while pending: there is nothing to download yet.
    download_url: str | None = None


class FileUploadResponse(FileResponse):
    """Returned only by file creation. An upload URL grants write access to the
    bucket, so it lives on its own model rather than leaking from every read."""

    upload_url: str
    upload_expires_in: int
