from typing import Annotated

import duckdb
from fastapi import APIRouter, Depends, HTTPException
from shared.models.calibration import CalibrationCreate, CalibrationResponse

from site_service import service
from site_service.db import get_db

# Calibrations are nested under their site: one has no meaning without the other,
# and the site_id is never taken from the request body.
router = APIRouter(prefix="/sites/{site_id}/calibrations", tags=["calibrations"])
DbConnection = Annotated[duckdb.DuckDBPyConnection, Depends(get_db)]


def _require_site(con: duckdb.DuckDBPyConnection, site_id: str) -> None:
    if service.get_site(con, site_id) is None:
        raise HTTPException(status_code=404, detail="Site not found")


@router.post("", response_model=CalibrationResponse, status_code=201)
def create_calibration(site_id: str, data: CalibrationCreate, con: DbConnection):
    # Checked up front so an unknown site is a 404 rather than the foreign key
    # blowing up as a 500.
    _require_site(con, site_id)
    return service.create_calibration(con, site_id, data)


@router.get("", response_model=CalibrationResponse)
def get_active_calibration(site_id: str, con: DbConnection):
    """Return the site's active calibration — the highest version."""
    _require_site(con, site_id)
    calibration = service.get_active_calibration(con, site_id)
    if calibration is None:
        raise HTTPException(status_code=404, detail="Calibration not found")
    return calibration


@router.get("/{calibration_id}", response_model=CalibrationResponse)
def get_calibration(site_id: str, calibration_id: str, con: DbConnection):
    _require_site(con, site_id)
    calibration = service.get_calibration(con, site_id, calibration_id)
    if calibration is None:
        raise HTTPException(status_code=404, detail="Calibration not found")
    return calibration
