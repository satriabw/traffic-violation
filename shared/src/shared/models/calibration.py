from datetime import datetime

from pydantic import BaseModel


class CalibrationCreate(BaseModel):
    # Only the S3 url the client got back from file-service. site_id comes from the
    # request path, and version is assigned server-side.
    url: str


class CalibrationResponse(BaseModel):
    id: str
    site_id: str
    url: str
    version: int
    created_at: datetime
    updated_at: datetime
