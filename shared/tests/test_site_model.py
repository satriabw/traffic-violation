import pytest
from pydantic import ValidationError

from shared.models.site import SiteCreate, SiteMode, SiteStatus


def test_site_create_accepts_valid_mode():
    site = SiteCreate(name="Intersection A", url="s3://bucket/video.mp4", mode="video")
    assert site.mode == SiteMode.VIDEO


def test_site_create_rejects_invalid_mode():
    with pytest.raises(ValidationError):
        SiteCreate(name="Intersection A", url="s3://bucket/video.mp4", mode="bogus")


def test_site_status_enum_has_expected_values():
    assert {s.value for s in SiteStatus} == {
        "created", "active", "processing", "completed", "failed", "degraded",
    }
