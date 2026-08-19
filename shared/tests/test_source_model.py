import pytest
from pydantic import ValidationError

from shared.models.source import SourceCreate, SourceKind, SourceStatus


def test_source_create_accepts_a_video_with_a_file():
    source = SourceCreate(kind="video", file_id="file-1")

    assert source.kind is SourceKind.VIDEO
    assert source.file_id == "file-1"
    assert source.stream_url is None


def test_source_create_accepts_a_stream_with_an_address():
    source = SourceCreate(kind="stream", stream_url="rtsp://10.0.0.5/s")

    assert source.kind is SourceKind.STREAM
    assert source.stream_url == "rtsp://10.0.0.5/s"
    assert source.file_id is None


def test_source_create_rejects_an_unknown_kind():
    with pytest.raises(ValidationError):
        SourceCreate(kind="carrier-pigeon", stream_url="rtsp://x")


@pytest.mark.parametrize(
    "label,kwargs",
    [
        ("video with an address", {"kind": "video", "stream_url": "rtsp://x"}),
        ("video with nothing", {"kind": "video"}),
        ("video with both", {"kind": "video", "file_id": "f", "stream_url": "rtsp://x"}),
        ("stream with a file", {"kind": "stream", "file_id": "f"}),
        ("stream with nothing", {"kind": "stream"}),
        ("stream with both", {"kind": "stream", "file_id": "f", "stream_url": "rtsp://x"}),
    ],
)
def test_source_create_rejects_a_source_that_does_not_match_its_kind(label, kwargs):
    # Caught in the model so a bad body is a 422, not a 500 from the table CHECK.
    with pytest.raises(ValidationError):
        SourceCreate(**kwargs)


def test_source_create_rejects_a_site_id_in_the_body():
    # site_id is owned by the path. Rejecting beats ignoring: a client that thinks it
    # is choosing the site finds out immediately.
    with pytest.raises(ValidationError):
        SourceCreate(kind="stream", stream_url="rtsp://x", site_id="other")


def test_source_status_enum_has_expected_values():
    # Unchanged from the old SiteStatus; 'active' and 'degraded' are stream states,
    # which is why they belong here rather than on a site.
    assert {s.value for s in SourceStatus} == {
        "created", "active", "processing", "completed", "failed", "degraded",
    }
