import json

import pytest
from shared.models.file import FileCreate, FileType
from shared.models.site import SiteCreate
from shared.models.source import SourceCreate, SourceKind, SourceMetadata, SourceStatus

from site_service import service
from tests.test_file_service import FakeStorage


@pytest.fixture
def site(con):
    return service.create_site(con, SiteCreate(name="Junction 5"))


def _video_file(con) -> str:
    storage = FakeStorage(existing_size=64)
    created = service.create_file(
        con, storage, FileCreate(name="a.mp4", type=FileType.VIDEO, size_bytes=64)
    )
    service.confirm_upload(con, storage, created.id)
    return created.id


def _stream(url="rtsp://cam") -> SourceCreate:
    return SourceCreate(kind="stream", stream_url=url)


def test_create_site_without_a_source_leaves_it_unset(con):
    site = service.create_site(con, SiteCreate(name="Junction 5"))

    assert site.source is None


def test_create_site_with_an_inline_source_creates_version_one(con):
    site = service.create_site(con, SiteCreate(name="Junction 5", source=_stream()))

    assert site.source.version == 1
    assert site.source.stream_url == "rtsp://cam"


def test_first_source_is_version_one(con, site):
    assert service.create_source(con, site.id, _stream()).version == 1


def test_each_source_appends_a_version(con, site):
    service.create_source(con, site.id, _stream("rtsp://a"))
    second = service.create_source(con, site.id, _stream("rtsp://b"))

    assert second.version == 2


def test_active_source_is_the_highest_version(con, site):
    service.create_source(con, site.id, _stream("rtsp://a"))
    latest = service.create_source(con, site.id, _stream("rtsp://b"))

    assert service.get_active_source(con, site.id).id == latest.id


def test_active_source_is_none_for_a_site_with_none(con, site):
    assert service.get_active_source(con, site.id) is None


def test_a_site_may_hold_both_kinds_over_time(con, site):
    # A camera you stream and also upload recordings from is one location.
    service.create_source(con, site.id, _stream())
    video = service.create_source(
        con, site.id, SourceCreate(kind="video", file_id=_video_file(con))
    )

    assert service.get_active_source(con, site.id).kind is SourceKind.VIDEO
    assert service.get_active_source(con, site.id).id == video.id


def test_superseded_versions_stay_readable(con, site):
    first = service.create_source(con, site.id, _stream("rtsp://a"))
    service.create_source(con, site.id, _stream("rtsp://b"))

    assert service.get_source(con, site.id, first.id).stream_url == "rtsp://a"


def test_get_source_is_scoped_to_its_site(con, site):
    other = service.create_site(con, SiteCreate(name="Other"))
    created = service.create_source(con, site.id, _stream())

    assert service.get_source(con, other.id, created.id) is None


def test_a_new_source_starts_in_created_status(con, site):
    assert service.create_source(con, site.id, _stream()).status is SourceStatus.CREATED


def test_video_source_records_its_file(con, site):
    file_id = _video_file(con)

    source = service.create_source(con, site.id, SourceCreate(kind="video", file_id=file_id))

    assert source.file_id == file_id
    assert source.stream_url is None


def test_get_site_embeds_the_active_source(con, site):
    service.create_source(con, site.id, _stream("rtsp://a"))
    service.create_source(con, site.id, _stream("rtsp://b"))

    assert service.get_site(con, site.id).source.stream_url == "rtsp://b"


def test_list_sites_embeds_each_active_source(con):
    a = service.create_site(con, SiteCreate(name="A", source=_stream("rtsp://a")))
    service.create_site(con, SiteCreate(name="B"))

    items = {s.id: s for s in service.list_sites(con, limit=20, offset=0).items}

    assert items[a.id].source.stream_url == "rtsp://a"
    assert [s for s in items.values() if s.name == "B"][0].source is None


def test_list_sites_filters_by_the_active_sources_kind(con):
    video_site = service.create_site(con, SiteCreate(name="Video"))
    service.create_source(
        con, video_site.id, SourceCreate(kind="video", file_id=_video_file(con))
    )
    service.create_site(con, SiteCreate(name="Stream", source=_stream()))

    result = service.list_sites(con, limit=20, offset=0, kind="video")

    assert result.total == 1
    assert result.items[0].name == "Video"


def test_kind_filter_follows_the_active_source_not_the_history(con):
    # A site that used to be a video and is now a stream is a stream site.
    site = service.create_site(con, SiteCreate(name="Switched"))
    service.create_source(con, site.id, SourceCreate(kind="video", file_id=_video_file(con)))
    service.create_source(con, site.id, _stream())

    assert service.list_sites(con, limit=20, offset=0, kind="video").total == 0
    assert service.list_sites(con, limit=20, offset=0, kind="stream").total == 1


def test_list_sites_filters_by_the_active_sources_status(con):
    service.create_site(con, SiteCreate(name="A", source=_stream()))

    assert service.list_sites(con, limit=20, offset=0, status="created").total == 1
    assert service.list_sites(con, limit=20, offset=0, status="completed").total == 0


def test_delete_site_removes_its_sources(con, site):
    service.create_source(con, site.id, _stream())

    assert service.delete_site(con, site.id) is True
    assert con.execute("SELECT COUNT(*) FROM site_sources").fetchone()[0] == 0


def test_metadata_is_decoded_back_into_a_model(con, site):
    # The metadata column is TEXT, and Pydantic will not coerce a string
    # into a nested model — without an explicit decode this is a 500 on every read of
    # a source that has metadata.
    source = service.create_source(con, site.id, _stream())
    con.execute(
        "UPDATE site_sources SET metadata = ? WHERE id = ?",
        [json.dumps({"fps": 30.0, "nominal_fps": 30.0, "total_frames": 900}), source.id],
    )

    fetched = service.get_source(con, site.id, source.id)

    assert fetched.metadata.fps == 30.0
    assert fetched.metadata.total_frames == 900


def test_a_source_without_metadata_reads_back_as_none(con, site):
    source = service.create_source(con, site.id, _stream())

    assert service.get_source(con, site.id, source.id).metadata is None


def test_create_source_persists_the_metadata_it_is_given(con, site):
    meta = SourceMetadata(fps=29.97, nominal_fps=30.0, total_frames=1000)

    created = service.create_source(con, site.id, _stream(), metadata=meta)

    assert created.metadata.fps == 29.97
    assert created.metadata.nominal_fps == 30.0
    assert service.get_source(con, site.id, created.id).metadata.total_frames == 1000
