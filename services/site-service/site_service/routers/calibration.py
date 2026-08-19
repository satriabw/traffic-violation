from shared.models.calibration import CalibrationCreate, CalibrationResponse
from shared.models.file import FileType

from site_service import service
from site_service.routers._versioned_document import build_router

router = build_router(
    resource="calibrations",
    table=service.CALIBRATIONS,
    file_type=FileType.CALIBRATION,
    create_model=CalibrationCreate,
    response_model=CalibrationResponse,
)
