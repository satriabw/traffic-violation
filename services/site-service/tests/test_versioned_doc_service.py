import pytest
from shared.models.file import FileCreate, FileType

from site_service import service
from tests.test_file_service import FakeStorage

# Every rule is proven against both tables, so the two resources cannot drift apart.
TABLES = (
    (service.CALIBRATIONS, FileType.CALIBRATION),
    (service.CONFIGURATIONS, FileType.CONFIGURATION),
)
TABLE_IDS = [table for table, _ in TABLES]


@pytest.fixture
def site(con):
    from shared.models.site import SiteCreate

    return service.create_site(con, SiteCreate(name="Main St"))


def _uploaded_file(con, file_type: FileType) -> str:
    """A file whose bytes are confirmed present — the only kind a document may point at."""
    storage = FakeStorage(existing_size=128)
    created = service.create_file(
        con, storage, FileCreate(name="a.json", type=file_type, size_bytes=128)
    )
    service.confirm_upload(con, storage, created.id)
    return created.id


def _pending_file(con, file_type: FileType) -> str:
    storage = FakeStorage()
    return service.create_file(
        con, storage, FileCreate(name="a.json", type=file_type, size_bytes=128)
    ).id


@pytest.mark.parametrize("table,file_type", TABLES, ids=TABLE_IDS)
def test_first_document_is_version_one(con, site, table, file_type):
    created = service.create_versioned_doc(
        con, table, site.id, _uploaded_file(con, file_type)
    )

    assert created.version == 1


@pytest.mark.parametrize("table,file_type", TABLES, ids=TABLE_IDS)
def test_each_post_appends_a_new_version(con, site, table, file_type):
    service.create_versioned_doc(con, table, site.id, _uploaded_file(con, file_type))
    second = service.create_versioned_doc(con, table, site.id, _uploaded_file(con, file_type))

    assert second.version == 2


@pytest.mark.parametrize("table,file_type", TABLES, ids=TABLE_IDS)
def test_active_version_is_the_highest(con, site, table, file_type):
    service.create_versioned_doc(con, table, site.id, _uploaded_file(con, file_type))
    latest = service.create_versioned_doc(con, table, site.id, _uploaded_file(con, file_type))

    assert service.get_active_version(con, table, site.id).id == latest.id


@pytest.mark.parametrize("table,file_type", TABLES, ids=TABLE_IDS)
def test_active_version_is_none_when_the_site_has_no_documents(con, site, table, file_type):
    assert service.get_active_version(con, table, site.id) is None


@pytest.mark.parametrize("table,file_type", TABLES, ids=TABLE_IDS)
def test_get_version_is_scoped_to_its_site(con, site, table, file_type):
    from shared.models.site import SiteCreate

    other = service.create_site(con, SiteCreate(name="Other"))
    created = service.create_versioned_doc(con, table, site.id, _uploaded_file(con, file_type))

    # One site must never be able to read another's documents by guessing an id.
    assert service.get_version(con, table, other.id, created.id) is None
    assert service.get_version(con, table, site.id, created.id).id == created.id


@pytest.mark.parametrize("table,file_type", TABLES, ids=TABLE_IDS)
def test_unusable_file_reason_is_missing_for_an_unknown_id(con, table, file_type):
    assert service.unusable_file_reason(con, "no-such-file", file_type) == "missing"


@pytest.mark.parametrize("table,file_type", TABLES, ids=TABLE_IDS)
def test_unusable_file_reason_is_pending_when_bytes_never_landed(con, table, file_type):
    # This is what the two-phase upload bought: a reserved slot is not a file.
    file_id = _pending_file(con, file_type)

    assert service.unusable_file_reason(con, file_id, file_type) == "pending"


@pytest.mark.parametrize("table,file_type", TABLES, ids=TABLE_IDS)
def test_unusable_file_reason_is_wrong_type_for_a_mismatched_file(con, table, file_type):
    video_id = _uploaded_file(con, FileType.VIDEO)

    assert service.unusable_file_reason(con, video_id, file_type) == "wrong_type"


@pytest.mark.parametrize("table,file_type", TABLES, ids=TABLE_IDS)
def test_unusable_file_reason_is_none_for_an_uploaded_file_of_the_right_type(
    con, table, file_type
):
    assert service.unusable_file_reason(con, _uploaded_file(con, file_type), file_type) is None


def test_delete_site_removes_both_calibrations_and_configurations(con, site):
    service.create_versioned_doc(
        con, service.CALIBRATIONS, site.id, _uploaded_file(con, FileType.CALIBRATION)
    )
    service.create_versioned_doc(
        con, service.CONFIGURATIONS, site.id, _uploaded_file(con, FileType.CONFIGURATION)
    )

    assert service.delete_site(con, site.id) is True
    for table in (service.CALIBRATIONS, service.CONFIGURATIONS):
        assert con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
