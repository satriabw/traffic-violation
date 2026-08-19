from datetime import datetime

from pydantic import BaseModel


class ConfigurationCreate(BaseModel):
    # The id of an already-uploaded configuration file. site_id comes from the request
    # path and version is assigned server-side, so neither belongs here.
    file_id: str


class ConfigurationResponse(BaseModel):
    id: str
    site_id: str
    file_id: str
    version: int
    created_at: datetime
    updated_at: datetime
