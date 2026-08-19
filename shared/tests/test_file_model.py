import pytest
from pydantic import ValidationError

from shared.models.file import FileCreate, FileResponse, FileStatus, FileType, FileUploadResponse


def test_file_create_accepts_a_known_type():
    data = FileCreate(name="clip.mp4", type="video", size_bytes=10, content_type="video/mp4")

    assert data.type is FileType.VIDEO


def test_file_create_rejects_an_unknown_type():
    with pytest.raises(ValidationError):
        FileCreate(name="clip.mp4", type="spreadsheet", size_bytes=10)


def test_file_create_requires_a_name():
    with pytest.raises(ValidationError):
        FileCreate(type="video", size_bytes=10)


def test_file_create_treats_content_type_as_optional():
    assert FileCreate(name="a.json", type="calibration", size_bytes=10).content_type is None


def _response_fields(**overrides):
    fields = {
        "id": "f1",
        "name": "clip.mp4",
        "url": "video/f1/clip.mp4",
        "type": "video",
        "status": "pending",
        "created_at": "2026-08-19T00:00:00",
        "updated_at": "2026-08-19T00:00:00",
    }
    return {**fields, **overrides}


def test_file_response_defaults_optional_fields_to_none():
    response = FileResponse(**_response_fields())

    assert response.status is FileStatus.PENDING
    assert response.size_bytes is None
    assert response.download_url is None


def test_file_response_cannot_carry_an_upload_url():
    # An upload URL grants write access, so only the endpoint that mints one may
    # return it. Keeping the field off FileResponse makes that structural.
    assert "upload_url" not in FileResponse.model_fields


def test_file_upload_response_carries_the_upload_url():
    response = FileUploadResponse(
        **_response_fields(), upload_url="https://r2/put", upload_expires_in=3600
    )

    assert response.upload_url == "https://r2/put"
    assert response.upload_expires_in == 3600


def test_file_create_requires_a_declared_size():
    # Required because the size is signed into the upload URL; without it there is
    # nothing binding the client to the size it claimed.
    with pytest.raises(ValidationError):
        FileCreate(name="clip.mp4", type="video")


def test_file_create_rejects_a_non_positive_size():
    with pytest.raises(ValidationError):
        FileCreate(name="clip.mp4", type="video", size_bytes=0)


def test_file_create_accepts_a_declared_size():
    assert FileCreate(name="clip.mp4", type="video", size_bytes=2048).size_bytes == 2048
