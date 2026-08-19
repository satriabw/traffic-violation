import pytest
from fastapi import HTTPException
from shared.models.file import FileType

from site_service.file_reference import raise_for_unusable_file


class _Con:
    """Minimal stand-in: raise_for_unusable_file only forwards to the service."""

    def __init__(self, reason):
        self._reason = reason


@pytest.fixture(autouse=True)
def stub_reason(monkeypatch):
    import site_service.file_reference as module

    monkeypatch.setattr(module.service, "unusable_file_reason", lambda con, fid, t: con._reason)


def test_accepts_a_usable_file():
    raise_for_unusable_file(_Con(None), "f1", FileType.VIDEO)  # does not raise


@pytest.mark.parametrize(
    "reason,status",
    [
        ("missing", 422),
        # 409 not 422: the request is well formed, the file is simply not ready. The
        # same request succeeds once the upload completes.
        ("pending", 409),
        ("wrong_type", 422),
    ],
)
def test_maps_each_reason_to_its_status(reason, status):
    with pytest.raises(HTTPException) as exc:
        raise_for_unusable_file(_Con(reason), "f1", FileType.VIDEO)

    assert exc.value.status_code == status


def test_wrong_type_message_names_the_expected_type():
    with pytest.raises(HTTPException) as exc:
        raise_for_unusable_file(_Con("wrong_type"), "f1", FileType.CALIBRATION)

    assert "calibration" in exc.value.detail
