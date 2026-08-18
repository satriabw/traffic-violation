from shared.models.site import SiteCreate, SiteStatus

from site_service.service import create_site, delete_site, get_site, list_sites


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
