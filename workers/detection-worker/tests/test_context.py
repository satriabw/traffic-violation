from datetime import timezone

import pytest
from shared.db.connection import get_connection
from shared.db.init import init_db
from shared.models.detection import DetectionJob, FrameRange, JobSource, ViolationType

from violation_detector import ConfigurationInvalid

from detection_worker.context import (
    CALIBRATIONS,
    DEFAULT_EVIDENCE_SECONDS,
    ContextMissing,
    JobContext,
    load,
)


@pytest.fixture
def con():
    connection = get_connection(":memory:")
    init_db(connection)
    connection.execute("INSERT INTO sites (id, name) VALUES ('s1', 'Junction 5')")
    # The source every job here reads. Its created_at is what a violation's
    # detected_at is measured from, so resolving context needs it to exist.
    connection.execute(
        "INSERT INTO site_sources (id, site_id, version, kind, stream_url)"
        " VALUES ('src1', 's1', 1, 'stream', 'rtsp://camera')"
    )
    return connection


def _file(con, file_id: str, key: str, file_type: str = "calibration") -> None:
    con.execute(
        "INSERT INTO files (id, name, url, type, status) VALUES (?, ?, ?, ?, 'uploaded')",
        [file_id, key.rsplit("/", 1)[-1], key, file_type],
    )


def _doc(con, table: str, doc_id: str, file_id: str, version: int) -> None:
    con.execute(
        f"INSERT INTO {table} (id, site_id, file_id, version) VALUES (?, 's1', ?, ?)",
        [doc_id, file_id, version],
    )


def _job(**overrides) -> DetectionJob:
    return DetectionJob(
        **{
            "id": "job-1",
            "site_id": "s1",
            "source": JobSource(source_id="src1", version=1, key="video/f9/a.mp4"),
            "frame_range": FrameRange(start=0, end=10),
            "types": [ViolationType.RED_LIGHT_RUNNING],
            **overrides,
        }
    )


def _fetcher(by_key: dict):
    def fetch(key: str):
        return by_key[key]

    return fetch


def _fetchers(by_key: dict) -> dict:
    """Both fetchers over one map of objects.

    Calibrations come back as raw bytes and configurations as parsed JSON — the
    asymmetry is the point, so the fake keeps it rather than smoothing it over.
    """
    return {"fetch_json": _fetcher(by_key), "fetch_bytes": _fetcher(by_key)}


def test_a_job_with_no_versions_resolves_to_no_context(con):
    # Normal today: there is no rule engine, so a video source with no calibration is
    # a perfectly ordinary site.
    def fetch(key):
        raise AssertionError(f"nothing should have been fetched, got {key}")

    resolved = load(con, _job(), fetch_json=fetch, fetch_bytes=fetch)

    assert (resolved.calibration, resolved.configuration) == (None, None)


def test_the_version_in_the_job_is_the_one_resolved_not_the_active_one(con):
    """The reason the version travels in the message at all.

    A job enqueued against v1 and consumed after someone uploaded v2 must still be
    evaluated against v1. Resolving "the site's active calibration" instead would use
    v2 here — and the run would look completely normal.
    """
    _file(con, "f1", "calibration/f1/v1.yml")
    _file(con, "f2", "calibration/f2/v2.yml")
    _doc(con, CALIBRATIONS, "c1", "f1", version=1)
    _doc(con, CALIBRATIONS, "c2", "f2", version=2)  # uploaded after the job was queued

    resolved = load(
        con,
        _job(calibration_version=1),
        **_fetchers(
            {"calibration/f1/v1.yml": b"which: v1", "calibration/f2/v2.yml": b"which: v2"}
        ),
    )

    assert resolved.calibration == b"which: v1"


def test_calibration_and_configuration_resolve_independently(con):
    _file(con, "f1", "calibration/f1/a.yml", "calibration")
    _file(con, "f2", "configuration/f2/b.json", "configuration")
    _doc(con, CALIBRATIONS, "c1", "f1", version=1)
    _doc(con, "configurations", "g1", "f2", version=3)

    resolved = load(
        con,
        _job(calibration_version=1, configuration_version=3),
        **_fetchers(
            {"calibration/f1/a.yml": b"%YAML:1.0", "configuration/f2/b.json": {"roi": [2]}}
        ),
    )

    # A calibration arrives as the document itself, unparsed — what is inside it is the
    # trajectory package's business, not this module's. A configuration is ours, so it
    # arrives parsed.
    assert (resolved.calibration, resolved.configuration) == (b"%YAML:1.0", {"roi": [2]})


def test_the_context_carries_the_id_of_each_row_it_resolved(con):
    """What a violation records to say what it was judged against. The id rather than
    the version the job named: it is the primary key of one version's row, so it pins
    that version by itself."""
    _file(con, "f1", "calibration/f1/a.yml", "calibration")
    _file(con, "f2", "configuration/f2/b.json", "configuration")
    _doc(con, CALIBRATIONS, "c1", "f1", version=1)
    _doc(con, "configurations", "g1", "f2", version=3)

    resolved = load(
        con,
        _job(calibration_version=1, configuration_version=3),
        **_fetchers({"calibration/f1/a.yml": b"x", "configuration/f2/b.json": {}}),
    )

    assert (resolved.calibration_id, resolved.configuration_id) == ("c1", "g1")


def test_the_id_resolved_belongs_to_the_version_the_job_named(con):
    """The same guarantee the document itself gets. A job pinned to v1 records v1's
    row, even though v2 is what the site is on now."""
    _file(con, "f1", "calibration/f1/v1.yml")
    _file(con, "f2", "calibration/f2/v2.yml")
    _doc(con, CALIBRATIONS, "c1", "f1", version=1)
    _doc(con, CALIBRATIONS, "c2", "f2", version=2)

    resolved = load(
        con,
        _job(calibration_version=1),
        **_fetchers({"calibration/f1/v1.yml": b"v1", "calibration/f2/v2.yml": b"v2"}),
    )

    assert resolved.calibration_id == "c1"


def test_a_job_with_no_versions_resolves_to_no_ids(con):
    # None means the site had none, which is ordinary — the same absence the documents
    # themselves report, arrived at without anyone deciding anything.
    resolved = load(con, _job(), **_fetchers({}))

    assert (resolved.calibration_id, resolved.configuration_id) == (None, None)


def test_only_one_of_the_two_being_present_is_fine(con):
    _file(con, "f2", "configuration/f2/b.json", "configuration")
    _doc(con, "configurations", "g1", "f2", version=1)

    resolved = load(
        con,
        _job(configuration_version=1),
        **_fetchers({"configuration/f2/b.json": {"roi": []}}),
    )

    assert resolved.calibration is None
    assert resolved.configuration == {"roi": []}


def test_a_version_that_is_not_in_the_database_is_an_error(con):
    """Not a fallback to the active version.

    The message and the database disagreeing means something is wrong. Quietly using
    whatever is active would reintroduce exactly the drift the version pin prevents,
    and the run would look fine.
    """
    _file(con, "f1", "calibration/f1/a.yml")
    _doc(con, CALIBRATIONS, "c1", "f1", version=1)

    with pytest.raises(ContextMissing, match="camera_calibrations v7"):
        load(con, _job(calibration_version=7), **_fetchers({}))


def test_one_site_cannot_resolve_another_sites_calibration(con):
    con.execute("INSERT INTO sites (id, name) VALUES ('s2', 'Elsewhere')")
    _file(con, "f1", "calibration/f1/a.yml")
    con.execute(
        f"INSERT INTO {CALIBRATIONS} (id, site_id, file_id, version) VALUES ('c1', 's2', 'f1', 1)"
    )

    with pytest.raises(ContextMissing):
        load(con, _job(calibration_version=1), **_fetchers({}))


# --- how much lead-up the site asks for -------------------------------------


def _with_configuration(con, document: dict):
    """A site whose configuration document is exactly this."""
    _file(con, "f-cfg", "configuration/f-cfg/c.json", "configuration")
    _doc(con, "configurations", "cfg-1", "f-cfg", 1)
    return load(
        con,
        _job(configuration_version=1),
        fetch_json=_fetcher({"configuration/f-cfg/c.json": document}),
    )


def _document(**overrides) -> dict:
    return {"version": 1, "violations": ["rlr_violation"], "regions": {}, **overrides}


def test_a_site_with_no_configuration_gets_the_default_window(con):
    assert load(con, _job()).evidence_seconds == DEFAULT_EVIDENCE_SECONDS


def test_a_site_whose_document_says_nothing_gets_the_default_window(con):
    assert _with_configuration(con, _document()).evidence_seconds == DEFAULT_EVIDENCE_SECONDS


def test_a_site_can_ask_for_a_longer_lead_up(con):
    # A junction is the thing that knows better. An approach with a long sight line
    # wants more; a tight one-way needs less.
    assert _with_configuration(con, _document(evidence_seconds=12)).evidence_seconds == 12.0


def test_a_window_written_as_a_string_is_read_as_a_number(con):
    assert _with_configuration(con, _document(evidence_seconds="7.5")).evidence_seconds == 7.5


@pytest.mark.parametrize("declared", [0, -1, "0"])
def test_a_window_that_is_not_positive_stops_the_job(con, declared):
    # While the context resolves, before a frame is decoded. A site whose records were
    # silently empty would look exactly like a junction where nothing ever happened.
    with pytest.raises(ConfigurationInvalid, match="evidence_seconds must be positive"):
        _with_configuration(con, _document(evidence_seconds=declared))


@pytest.mark.parametrize("declared", ["five", None, [5]])
def test_a_window_that_is_not_a_number_stops_the_job(con, declared):
    with pytest.raises(ConfigurationInvalid, match="evidence_seconds must be a number"):
        _with_configuration(con, _document(evidence_seconds=declared))
def test_the_context_carries_the_anchor_a_detected_at_is_measured_from(con):
    """The frame index gives the offset into the footage; this turns it into a moment.

    It is the upload time, not the recording time — nothing in the system knows when
    footage was shot — so it is late by a constant per source, which leaves violations
    correctly ordered and spaced within a video.
    """
    anchor = load(con, _job()).source_created_at

    assert anchor == con.execute(
        "SELECT created_at FROM site_sources WHERE id = 'src1'"
    ).fetchone()[0].replace(tzinfo=timezone.utc)
    # Aware, because detected_at is written aware and one column holding both kinds is
    # a comparison that raises the first time anyone sorts it.
    assert anchor.tzinfo is not None


def test_a_job_naming_a_source_that_is_not_in_the_database_is_an_error(con):
    """The message and the database disagree. A job that cannot say when its
    violations happened should stop rather than guess."""
    with pytest.raises(ContextMissing, match="source no-such-source does not exist"):
        load(con, _job(source=JobSource(source_id="no-such-source", version=1, key="k")))
