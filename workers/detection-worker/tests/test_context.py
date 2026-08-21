import pytest
from shared.db.connection import get_connection
from shared.db.init import init_db
from shared.models.detection import DetectionJob, FrameRange, JobSource, ViolationType

from detection_worker.context import CALIBRATIONS, ContextMissing, JobContext, load


@pytest.fixture
def con():
    connection = get_connection(":memory:")
    init_db(connection)
    connection.execute("INSERT INTO sites (id, name) VALUES ('s1', 'Junction 5')")
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
    def fetch(key: str) -> dict:
        return by_key[key]

    return fetch


def test_a_job_with_no_versions_resolves_to_no_context(con):
    # Normal today: there is no rule engine, so a video source with no calibration is
    # a perfectly ordinary site.
    def fetch(key):
        raise AssertionError(f"nothing should have been fetched, got {key}")

    assert load(con, _job(), fetch=fetch) == JobContext()


def test_the_version_in_the_job_is_the_one_resolved_not_the_active_one(con):
    """The reason the version travels in the message at all.

    A job enqueued against v1 and consumed after someone uploaded v2 must still be
    evaluated against v1. Resolving "the site's active calibration" instead would use
    v2 here — and the run would look completely normal.
    """
    _file(con, "f1", "calibration/f1/v1.json")
    _file(con, "f2", "calibration/f2/v2.json")
    _doc(con, CALIBRATIONS, "c1", "f1", version=1)
    _doc(con, CALIBRATIONS, "c2", "f2", version=2)  # uploaded after the job was queued

    resolved = load(
        con,
        _job(calibration_version=1),
        fetch=_fetcher(
            {"calibration/f1/v1.json": {"which": "v1"}, "calibration/f2/v2.json": {"which": "v2"}}
        ),
    )

    assert resolved.calibration == {"which": "v1"}


def test_calibration_and_configuration_resolve_independently(con):
    _file(con, "f1", "calibration/f1/a.json", "calibration")
    _file(con, "f2", "configuration/f2/b.json", "configuration")
    _doc(con, CALIBRATIONS, "c1", "f1", version=1)
    _doc(con, "configurations", "g1", "f2", version=3)

    resolved = load(
        con,
        _job(calibration_version=1, configuration_version=3),
        fetch=_fetcher(
            {"calibration/f1/a.json": {"homography": [1]}, "configuration/f2/b.json": {"roi": [2]}}
        ),
    )

    assert resolved == JobContext(calibration={"homography": [1]}, configuration={"roi": [2]})


def test_only_one_of_the_two_being_present_is_fine(con):
    _file(con, "f2", "configuration/f2/b.json", "configuration")
    _doc(con, "configurations", "g1", "f2", version=1)

    resolved = load(
        con,
        _job(configuration_version=1),
        fetch=_fetcher({"configuration/f2/b.json": {"roi": []}}),
    )

    assert resolved.calibration is None
    assert resolved.configuration == {"roi": []}


def test_a_version_that_is_not_in_the_database_is_an_error(con):
    """Not a fallback to the active version.

    The message and the database disagreeing means something is wrong. Quietly using
    whatever is active would reintroduce exactly the drift the version pin prevents,
    and the run would look fine.
    """
    _file(con, "f1", "calibration/f1/a.json")
    _doc(con, CALIBRATIONS, "c1", "f1", version=1)

    with pytest.raises(ContextMissing, match="camera_calibrations v7"):
        load(con, _job(calibration_version=7), fetch=_fetcher({}))


def test_one_site_cannot_resolve_another_sites_calibration(con):
    con.execute("INSERT INTO sites (id, name) VALUES ('s2', 'Elsewhere')")
    _file(con, "f1", "calibration/f1/a.json")
    con.execute(
        f"INSERT INTO {CALIBRATIONS} (id, site_id, file_id, version) VALUES ('c1', 's2', 'f1', 1)"
    )

    with pytest.raises(ContextMissing):
        load(con, _job(calibration_version=1), fetch=_fetcher({}))
