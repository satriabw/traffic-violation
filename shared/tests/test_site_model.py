import pytest
from pydantic import ValidationError

from shared.models.site import SiteCreate, SiteResponse


def test_site_create_needs_only_a_name():
    site = SiteCreate(name="Junction 5")

    assert site.name == "Junction 5"
    # A site with no source yet is a valid state — the stream url can come later.
    assert site.source is None


def test_site_create_accepts_an_inline_source():
    site = SiteCreate(name="Junction 5", source={"kind": "stream", "stream_url": "rtsp://x"})

    assert site.source.stream_url == "rtsp://x"


def test_site_create_rejects_an_invalid_inline_source():
    with pytest.raises(ValidationError):
        SiteCreate(name="Junction 5", source={"kind": "video", "stream_url": "rtsp://x"})


def test_site_create_requires_a_name():
    with pytest.raises(ValidationError):
        SiteCreate()


@pytest.mark.parametrize("field", ["mode", "url", "stream_url", "file_id", "status", "metadata"])
def test_site_no_longer_carries_per_run_fields(field):
    # These all describe one processing run, not a durable location.
    assert field not in SiteCreate.model_fields
    assert field not in SiteResponse.model_fields


def test_site_response_embeds_the_active_source():
    response = SiteResponse(
        id="s1",
        name="Junction 5",
        created_at="2026-08-19T00:00:00",
        updated_at="2026-08-19T00:00:00",
        source={
            "id": "src1",
            "site_id": "s1",
            "version": 1,
            "kind": "stream",
            "stream_url": "rtsp://x",
            "status": "created",
            "created_at": "2026-08-19T00:00:00",
            "updated_at": "2026-08-19T00:00:00",
        },
    )

    assert response.source.version == 1


def test_site_response_source_is_optional():
    response = SiteResponse(
        id="s1",
        name="Junction 5",
        created_at="2026-08-19T00:00:00",
        updated_at="2026-08-19T00:00:00",
    )

    assert response.source is None
