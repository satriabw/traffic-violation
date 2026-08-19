from datetime import datetime

from pydantic import BaseModel


class VersionedDocument(BaseModel):
    """The shared shape behind CalibrationResponse and ConfigurationResponse.

    Both are a versioned pointer from a site to a file, so the service layer returns
    this one model and each router declares its own response_model — the wire contract
    stays per-resource even though the storage logic is written once.
    """

    id: str
    site_id: str
    file_id: str
    version: int
    created_at: datetime
    updated_at: datetime
