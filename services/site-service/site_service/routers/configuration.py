from shared.models.configuration import ConfigurationCreate, ConfigurationResponse
from shared.models.file import FileType

from site_service import service
from site_service.routers._versioned_document import build_router

router = build_router(
    resource="configurations",
    table=service.CONFIGURATIONS,
    file_type=FileType.CONFIGURATION,
    create_model=ConfigurationCreate,
    response_model=ConfigurationResponse,
)
