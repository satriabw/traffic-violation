import pytest
from shared.models.file import FileCreate, FileStatus

from site_service import service


class FakeStorage:
    """Stands in for shared.s3.client. Records what it was asked to sign so the
    tests can assert on the storage contract without touching the network."""

    def __init__(self, existing_size: int | None = None):
        self.put_calls: list[tuple[str, str | None, int | None]] = []
        # None means "the object is not in the bucket".
        self.existing_size = existing_size

    def presigned_put(self, key, content_type=None, content_length=None):
        self.put_calls.append((key, content_type, content_length))
        return f"https://r2.test/put/{key}"

    def download_url(self, key):
        return f"https://r2.test/get/{key}"

    def head(self, key):
        if self.existing_size is None:
            return None
        return {"ContentLength": self.existing_size}


@pytest.fixture
def storage():
    return FakeStorage()


def _create(con, storage, **overrides):
    data = FileCreate(**{"name": "clip.mp4", "type": "video", "size_bytes": 4096, **overrides})
    return service.create_file(con, storage, data)


def test_create_file_persists_a_pending_row(con, storage):
    created = _create(con, storage)

    row = con.execute("SELECT status, name, type FROM files WHERE id = ?", [created.id]).fetchone()
    assert row == ("pending", "clip.mp4", "video")


def test_create_file_assigns_a_key_under_the_type_and_id_prefix(con, storage):
    created = _create(con, storage)

    assert created.url == f"video/{created.id}/clip.mp4"


def test_create_file_returns_the_presigned_upload_url(con, storage):
    created = _create(con, storage)

    assert created.upload_url == f"https://r2.test/put/{created.url}"
    assert created.upload_expires_in > 0


def test_create_file_forwards_content_type_to_the_presigner(con, storage):
    _create(con, storage, content_type="video/mp4")

    assert storage.put_calls[0][1] == "video/mp4"


def test_create_file_has_no_download_url_yet(con, storage):
    assert _create(con, storage).download_url is None


def test_get_file_returns_none_for_an_unknown_id(con, storage):
    assert service.get_file(con, storage, "no-such-file") is None


def test_get_file_omits_the_download_url_while_pending(con, storage):
    created = _create(con, storage)

    assert service.get_file(con, storage, created.id).download_url is None


def test_get_file_includes_the_download_url_once_uploaded(con):
    storage = FakeStorage(existing_size=1234)
    created = _create(con, storage)
    service.confirm_upload(con, storage, created.id)

    fetched = service.get_file(con, storage, created.id)
    assert fetched.download_url == f"https://r2.test/get/{created.url}"


def test_confirm_upload_returns_none_when_the_object_is_absent(con, storage):
    created = _create(con, storage)

    assert service.confirm_upload(con, storage, created.id) is None


def test_confirm_upload_leaves_the_row_pending_when_the_object_is_absent(con, storage):
    created = _create(con, storage)
    service.confirm_upload(con, storage, created.id)

    assert service.get_file(con, storage, created.id).status is FileStatus.PENDING


def test_confirm_upload_marks_the_row_uploaded_and_records_the_size(con):
    storage = FakeStorage(existing_size=4096)
    created = _create(con, storage)

    confirmed = service.confirm_upload(con, storage, created.id)

    assert confirmed.status is FileStatus.UPLOADED
    assert confirmed.size_bytes == 4096


def test_confirm_upload_is_idempotent(con):
    storage = FakeStorage(existing_size=4096)
    created = _create(con, storage)

    first = service.confirm_upload(con, storage, created.id)
    second = service.confirm_upload(con, storage, created.id)

    assert second.status is FileStatus.UPLOADED
    assert second.size_bytes == first.size_bytes


def test_create_file_records_the_declared_size(con, storage):
    created = _create(con, storage, size_bytes=2048)

    assert created.size_bytes == 2048


def test_create_file_signs_the_declared_size_into_the_upload_url(con):
    class SizeRecordingStorage(FakeStorage):
        def presigned_put(self, key, content_type=None, content_length=None):
            self.signed_length = content_length
            return "https://r2.test/put"

    storage = SizeRecordingStorage()
    _create(con, storage, size_bytes=2048)

    assert storage.signed_length == 2048


def test_confirm_upload_overwrites_the_declared_size_with_the_actual_one(con):
    # The declared size is a claim; HeadObject is the fact.
    storage = FakeStorage(existing_size=1500)
    created = _create(con, storage, size_bytes=2048)

    assert service.confirm_upload(con, storage, created.id).size_bytes == 1500
