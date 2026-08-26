import pytest
from shared.models.site import SiteCreate


from site_service.service import (
    SiteHasViolations,
    create_site,
    delete_site,
    get_site,
    list_sites,
)


def test_create_site_persists_and_returns_identity(con):
    site = create_site(con, SiteCreate(name="Main St"))

    assert site.name == "Main St"
    assert site.id  # server-generated UUID
    assert site.created_at == site.updated_at


def test_get_site_returns_existing_site(con):
    created = create_site(con, SiteCreate(name="Main St"))

    fetched = get_site(con, created.id)

    assert fetched is not None
    assert fetched.id == created.id


def test_get_site_returns_none_when_missing(con):
    assert get_site(con, "does-not-exist") is None


def test_list_sites_returns_all_with_default_pagination(con):
    create_site(con, SiteCreate(name="A"))
    create_site(con, SiteCreate(name="B"))

    result = list_sites(con, limit=20, offset=0)

    assert result.total == 2
    assert len(result.items) == 2


def test_list_sites_respects_limit_and_offset(con):
    for i in range(3):
        create_site(con, SiteCreate(name=f"Site {i}"))

    page = list_sites(con, limit=1, offset=1)

    assert page.total == 3
    assert len(page.items) == 1


def test_delete_site_removes_existing_site(con):
    created = create_site(con, SiteCreate(name="A"))

    deleted = delete_site(con, created.id)

    assert deleted is True
    assert get_site(con, created.id) is None


def test_delete_site_returns_false_when_missing(con):
    assert delete_site(con, "does-not-exist") is False


def test_delete_site_refuses_a_site_that_has_violations(con):
    """Configuration is disposable; a record of something that happened is not.

    Everything else hanging off a site cascades away with it. Violations do not, so
    the delete is refused rather than quietly taking them along.
    """
    site = create_site(con, SiteCreate(name="Busy Junction"))
    # A violation pins the source it was found in, so one has to exist to point at.
    con.execute(
        "INSERT INTO site_sources (id, site_id, version, kind, stream_url)"
        " VALUES ('src-busy', ?, 1, 'stream', 'rtsp://camera')",
        [site.id],
    )
    con.execute(
        "INSERT INTO traffic_violations"
        " (id, site_id, source_id, frame_index, type, detected_at)"
        " VALUES ('v1', ?, 'src-busy', 912, 'red_light_running', '2026-08-21 10:00:00')",
        [site.id],
    )

    with pytest.raises(SiteHasViolations):
        delete_site(con, site.id)

    assert get_site(con, site.id) is not None


def _site(con, name="A"):
    return create_site(con, SiteCreate(name=name, url="s3://v", mode="video"))
