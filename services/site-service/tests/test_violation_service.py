"""The service layer's half of the violation list: which setup it filters on, and
the signed links it mints. The wire path is tested through the client in
test_violation_routes.py — these go straight at the connection so the filtering can
be exercised without a request in the way."""

from datetime import datetime, timezone

import pytest
from shared.db.connection import get_connection
from shared.db.init import init_db
from shared.db.violations import record, set_evidence
from shared.models.detection import ViolationType
from shared.models.violation import EvidenceStatus, ViolationCreate

from site_service import service
from tests.test_file_service import FakeStorage

DETECTED_AT = datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc)


@pytest.fixture
def con():
    con = get_connection(":memory:")
    init_db(con)
    con.execute("INSERT INTO sites (id, name) VALUES ('s1', 'Junction 5')")
    con.execute(
        "INSERT INTO files (id, name, url, type, status)"
        " VALUES ('f1', 'a.mp4', 'video/f1/a.mp4', 'video', 'uploaded')"
    )
    con.execute(
        "INSERT INTO site_sources (id, site_id, version, kind, file_id)"
        " VALUES ('src1', 's1', 1, 'video', 'f1')"
    )
    yield con
    con.close()


@pytest.fixture
def storage():
    return FakeStorage(existing_size=64)


def _calibration(con) -> str:
    return service.create_versioned_doc(con, service.CALIBRATIONS, "s1", "f1").id


def _configuration(con) -> str:
    return service.create_versioned_doc(con, service.CONFIGURATIONS, "s1", "f1").id


def _violation(con, calibration_id=None, configuration_id=None) -> str:
    return record(
        con,
        ViolationCreate(
            site_id="s1",
            source_id="src1",
            frame_index=912,
            calibration_id=calibration_id,
            configuration_id=configuration_id,
            type=ViolationType.PEDESTRIAN_RIGHT_OF_WAY,
            detected_at=DETECTED_AT,
        ),
    )


def _list(con, storage, limit=20, offset=0):
    return service.list_violations(con, storage, "s1", limit=limit, offset=offset)


# --- which setup the list means -----------------------------------------------


def test_the_list_is_the_setup_the_site_runs_now(con, storage):
    cal, cfg = _calibration(con), _configuration(con)
    current = _violation(con, calibration_id=cal, configuration_id=cfg)

    page = _list(con, storage)

    assert [item.id for item in page.items] == [current]
    assert page.total == 1


def test_a_violation_judged_under_a_superseded_calibration_drops_out(con, storage):
    # The reason the ids are on the row at all. Re-calibrating does not invalidate what
    # was recorded, but it does mean the old violations no longer describe the setup
    # this site is running — and drawing them with today's polygons would put the
    # vehicle outside the box it was convicted in.
    old_cal, cfg = _calibration(con), _configuration(con)
    under_old = _violation(con, calibration_id=old_cal, configuration_id=cfg)

    assert [item.id for item in _list(con, storage).items] == [under_old]

    new_cal = _calibration(con)
    under_new = _violation(con, calibration_id=new_cal, configuration_id=cfg)

    page = _list(con, storage)
    assert [item.id for item in page.items] == [under_new]
    assert page.total == 1


def test_a_site_with_no_calibration_still_lists_its_violations(con, storage):
    # The null-safe half of the filter, reached the way it is in production: the site
    # simply has no calibration, so the active version resolves to None. A site with a
    # video and no camera model is ordinary, and detection runs against it.
    cfg = _configuration(con)
    without = _violation(con, calibration_id=None, configuration_id=cfg)

    page = _list(con, storage)

    assert [item.id for item in page.items] == [without]


def test_a_site_with_no_documents_at_all_lists_the_violations_that_had_none(con, storage):
    recorded = _violation(con)

    page = _list(con, storage)

    assert [item.id for item in page.items] == [recorded]


# --- the links a reviewer opens -----------------------------------------------


def test_a_finished_cut_carries_a_signed_link_for_each_object(con, storage):
    violation_id = _violation(con)
    set_evidence(
        con,
        violation_id,
        EvidenceStatus.READY,
        thumbnail_key="evidence/v/thumbnail.jpg",
        clip_key="evidence/v/clip.mp4",
    )

    (item,) = _list(con, storage).items

    assert item.thumbnail_url == "https://r2.test/get/evidence/v/thumbnail.jpg"
    assert item.clip_url == "https://r2.test/get/evidence/v/clip.mp4"
    # The keys ride along beside them, the way FileResponse keeps url and download_url.
    assert item.thumbnail_key == "evidence/v/thumbnail.jpg"


@pytest.mark.parametrize("status", [EvidenceStatus.PENDING, EvidenceStatus.FAILED])
def test_a_cut_that_is_not_finished_has_nothing_to_link_to(con, storage, status):
    violation_id = _violation(con)
    set_evidence(con, violation_id, status)

    (item,) = _list(con, storage).items

    assert (item.thumbnail_url, item.clip_url) == (None, None)
    assert item.evidence_status is status


def test_a_violation_that_predates_evidence_worker_links_to_nothing(con, storage):
    # NULL evidence_status, which a reader has to tell apart from 'pending': nothing
    # was ever queued for this one, so waiting on it would wait for good.
    _violation(con)

    (item,) = _list(con, storage).items

    assert (item.thumbnail_url, item.clip_url, item.evidence_status) == (None, None, None)


# --- the shape of the page ----------------------------------------------------


def test_the_page_never_carries_the_metadata_blob(con, storage):
    _violation(con)

    (item,) = _list(con, storage).items

    assert item.metadata is None


def test_the_page_reports_the_window_it_is(con, storage):
    for _ in range(3):
        _violation(con)

    page = _list(con, storage, limit=2, offset=1)

    assert (len(page.items), page.total, page.limit, page.offset) == (2, 3, 2, 1)


def test_a_site_with_no_violations_is_an_empty_page_not_an_error(con, storage):
    page = _list(con, storage)

    assert page.items == []
    assert page.total == 0


# --- the detail read applies no setup filter ----------------------------------


def test_a_violation_the_list_hides_is_still_readable_by_id(con, storage):
    # The list answers "what holds under the setup this site runs now" and drops a
    # violation judged under a superseded calibration. The detail read answers "what
    # is this violation", and a reader holding an id is entitled to it either way —
    # the row carries the ids it was judged under, so they can see for themselves.
    superseded = _calibration(con)
    violation_id = _violation(con, calibration_id=superseded)
    _calibration(con)  # a second version, which is now the active one

    assert _list(con, storage).items == []

    violation = service.get_violation(con, storage, "s1", violation_id)
    assert violation is not None
    assert violation.calibration_id == superseded


def test_the_detail_read_refuses_another_sites_violation(con, storage):
    con.execute("INSERT INTO sites (id, name) VALUES ('s2', 'Junction 6')")
    violation_id = _violation(con)

    assert service.get_violation(con, storage, "s2", violation_id) is None


def test_the_detail_read_of_a_missing_violation_is_none(con, storage):
    assert service.get_violation(con, storage, "s1", "nope") is None
