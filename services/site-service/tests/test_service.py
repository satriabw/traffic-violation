from shared.models.calibration import CalibrationCreate
from shared.models.site import SiteCreate, SiteStatus

from site_service.service import (
    create_calibration,
    create_site,
    delete_site,
    get_active_calibration,
    get_calibration,
    get_site,
    list_sites,
)


def test_create_site_persists_and_returns_created_status(con):
    site = create_site(con, SiteCreate(name="Main St", url="s3://bucket/a.mp4", mode="video"))

    assert site.name == "Main St"
    assert site.status == SiteStatus.CREATED
    assert site.id  # server-generated UUID
    assert site.created_at == site.updated_at


def test_get_site_returns_existing_site(con):
    created = create_site(con, SiteCreate(name="Main St", url="s3://bucket/a.mp4", mode="video"))

    fetched = get_site(con, created.id)

    assert fetched is not None
    assert fetched.id == created.id


def test_get_site_returns_none_when_missing(con):
    assert get_site(con, "does-not-exist") is None


def test_list_sites_returns_all_with_default_pagination(con):
    create_site(con, SiteCreate(name="A", url="s3://a", mode="video"))
    create_site(con, SiteCreate(name="B", url="s3://b", mode="stream"))

    result = list_sites(con, limit=20, offset=0)

    assert result.total == 2
    assert len(result.items) == 2


def test_list_sites_respects_limit_and_offset(con):
    for i in range(3):
        create_site(con, SiteCreate(name=f"Site {i}", url=f"s3://{i}", mode="video"))

    page = list_sites(con, limit=1, offset=1)

    assert page.total == 3
    assert len(page.items) == 1


def test_list_sites_filters_by_mode(con):
    create_site(con, SiteCreate(name="A", url="s3://a", mode="video"))
    create_site(con, SiteCreate(name="B", url="s3://b", mode="stream"))

    result = list_sites(con, limit=20, offset=0, mode="stream")

    assert result.total == 1
    assert result.items[0].name == "B"


def test_delete_site_removes_existing_site(con):
    created = create_site(con, SiteCreate(name="A", url="s3://a", mode="video"))

    deleted = delete_site(con, created.id)

    assert deleted is True
    assert get_site(con, created.id) is None


def test_delete_site_returns_false_when_missing(con):
    assert delete_site(con, "does-not-exist") is False


def _site(con, name="A"):
    return create_site(con, SiteCreate(name=name, url="s3://v", mode="video"))


def test_create_calibration_starts_at_version_one(con):
    site = _site(con)

    calibration = create_calibration(con, site.id, CalibrationCreate(url="s3://cal/v1.json"))

    assert calibration.version == 1
    assert calibration.site_id == site.id
    assert calibration.url == "s3://cal/v1.json"
    assert calibration.id


def test_create_calibration_increments_version_for_same_site(con):
    site = _site(con)

    create_calibration(con, site.id, CalibrationCreate(url="s3://cal/v1.json"))
    second = create_calibration(con, site.id, CalibrationCreate(url="s3://cal/v2.json"))

    assert second.version == 2


def test_create_calibration_versions_are_independent_per_site(con):
    first_site = _site(con, "A")
    second_site = _site(con, "B")
    create_calibration(con, first_site.id, CalibrationCreate(url="s3://a/v1.json"))

    other = create_calibration(con, second_site.id, CalibrationCreate(url="s3://b/v1.json"))

    assert other.version == 1


def test_get_active_calibration_returns_highest_version(con):
    site = _site(con)
    create_calibration(con, site.id, CalibrationCreate(url="s3://cal/v1.json"))
    create_calibration(con, site.id, CalibrationCreate(url="s3://cal/v2.json"))

    active = get_active_calibration(con, site.id)

    assert active is not None
    assert active.version == 2
    assert active.url == "s3://cal/v2.json"


def test_get_active_calibration_returns_none_when_site_has_none(con):
    site = _site(con)

    assert get_active_calibration(con, site.id) is None


def test_get_calibration_returns_older_version_by_id(con):
    site = _site(con)
    first = create_calibration(con, site.id, CalibrationCreate(url="s3://cal/v1.json"))
    create_calibration(con, site.id, CalibrationCreate(url="s3://cal/v2.json"))

    fetched = get_calibration(con, site.id, first.id)

    assert fetched is not None
    assert fetched.version == 1


def test_get_calibration_is_scoped_to_its_site(con):
    owner = _site(con, "owner")
    stranger = _site(con, "stranger")
    calibration = create_calibration(con, owner.id, CalibrationCreate(url="s3://cal/v1.json"))

    assert get_calibration(con, stranger.id, calibration.id) is None


def test_get_calibration_returns_none_when_missing(con):
    site = _site(con)

    assert get_calibration(con, site.id, "does-not-exist") is None


def test_delete_site_also_deletes_its_calibrations(con):
    site = _site(con)
    create_calibration(con, site.id, CalibrationCreate(url="s3://cal/v1.json"))

    assert delete_site(con, site.id) is True
    assert get_site(con, site.id) is None
    assert con.execute(
        "SELECT COUNT(*) FROM camera_calibrations WHERE site_id = ?", [site.id]
    ).fetchone()[0] == 0
