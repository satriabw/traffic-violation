import pytest
from pydantic import ValidationError

from shared.models.calibration import CalibrationCreate, CalibrationResponse


def test_calibration_create_accepts_a_file_id():
    assert CalibrationCreate(file_id="file-1").file_id == "file-1"


def test_calibration_create_requires_a_file_id():
    with pytest.raises(ValidationError):
        CalibrationCreate()


def test_calibration_create_no_longer_accepts_a_url():
    # A url was an unverifiable claim; a file_id points at a row whose upload the
    # service has confirmed.
    assert "url" not in CalibrationCreate.model_fields


def test_calibration_create_ignores_site_id_from_body():
    # site_id is owned by the path, never the payload — a client that sends one
    # must not be able to attach a calibration to a different site.
    calibration = CalibrationCreate(file_id="file-1", site_id="other-site")
    assert not hasattr(calibration, "site_id")


def test_calibration_response_carries_version_and_site():
    response = CalibrationResponse(
        id="cal-1",
        site_id="site-1",
        file_id="file-1",
        version=2,
        created_at="2026-08-18T00:00:00",
        updated_at="2026-08-18T00:00:00",
    )

    assert response.version == 2
    assert response.site_id == "site-1"
    assert response.file_id == "file-1"
