import sqlite3

import pytest

from shared.db import init
from shared.db.connection import get_connection
from shared.db.init import init_db


def _fresh():
    """A schema-ready in-memory database.

    get_connection rather than sqlite3.connect: foreign keys are off by default in
    SQLite and enabled per connection, so a raw connect would make every FK
    assertion below pass vacuously.
    """
    con = get_connection(":memory:")
    init_db(con)
    return con


def _columns(con, table: str) -> set[str]:
    # PRAGMA table_info puts the column name at index 1; index 0 is its position.
    return {row[1] for row in con.execute(f"PRAGMA table_info({table})").fetchall()}


def test_init_db_creates_sites_table():
    con = _fresh()

    columns = _columns(con, "sites")
    # Identity only. Everything per-run lives on site_sources, because a site
    # outlives any one video.
    assert columns == {"id", "name", "created_at", "updated_at"}


def test_init_db_creates_site_sources_table():
    con = _fresh()

    columns = _columns(con, "site_sources")
    assert columns == {
        "id", "site_id", "version", "kind", "file_id", "stream_url",
        "status", "metadata", "created_at", "updated_at",
    }


def _with_site_and_file():
    con = _fresh()
    con.execute("INSERT INTO sites (id, name) VALUES ('s1', 'Junction 5')")
    con.execute(
        "INSERT INTO files (id, name, url, type, status)"
        " VALUES ('f1', 'a.mp4', 'video/f1/a.mp4', 'video', 'uploaded')"
    )
    return con


def _insert_source(con, kind, stream_url=None, file_id=None, version=1, source_id="src1"):
    con.execute(
        "INSERT INTO site_sources (id, site_id, version, kind, stream_url, file_id)"
        " VALUES (?, 's1', ?, ?, ?, ?)",
        [source_id, version, kind, stream_url, file_id],
    )


# kind decides which column carries the source; the other must be empty.
@pytest.mark.parametrize(
    "label,kind,stream_url,file_id",
    [
        ("video source with a file", "video", None, "f1"),
        ("stream source with an address", "stream", "rtsp://cam", None),
    ],
)
def test_site_sources_accepts_a_valid_source_for_its_kind(label, kind, stream_url, file_id):
    con = _with_site_and_file()

    _insert_source(con, kind, stream_url, file_id)

    assert con.execute("SELECT COUNT(*) FROM site_sources").fetchone()[0] == 1


@pytest.mark.parametrize(
    "label,kind,stream_url,file_id",
    [
        ("video source with an address", "video", "rtsp://cam", None),
        ("video source with nothing", "video", None, None),
        ("video source with both", "video", "rtsp://cam", "f1"),
        ("stream source with a file", "stream", None, "f1"),
        ("stream source with nothing", "stream", None, None),
        ("stream source with both", "stream", "rtsp://cam", "f1"),
    ],
)
def test_site_sources_rejects_a_source_that_does_not_match_its_kind(
    label, kind, stream_url, file_id
):
    con = _with_site_and_file()

    with pytest.raises(sqlite3.IntegrityError):
        _insert_source(con, kind, stream_url, file_id)


def test_site_sources_rejects_an_unknown_kind():
    con = _with_site_and_file()

    with pytest.raises(sqlite3.IntegrityError):
        _insert_source(con, "carrier-pigeon", stream_url="rtsp://cam")


def test_site_sources_rejects_an_unknown_site_id():
    con = _with_site_and_file()

    with pytest.raises(sqlite3.IntegrityError):
        con.execute(
            "INSERT INTO site_sources (id, site_id, version, kind, stream_url)"
            " VALUES ('src1', 'nope', 1, 'stream', 'rtsp://cam')"
        )


def test_site_sources_rejects_an_unknown_file_id():
    con = _with_site_and_file()

    with pytest.raises(sqlite3.IntegrityError):
        _insert_source(con, "video", file_id="no-such-file")


def test_site_sources_rejects_a_duplicate_version_for_a_site():
    con = _with_site_and_file()
    _insert_source(con, "stream", stream_url="rtsp://a", version=1, source_id="src1")

    with pytest.raises(sqlite3.IntegrityError):
        _insert_source(con, "stream", stream_url="rtsp://b", version=1, source_id="src2")


def test_site_sources_defaults_to_created_status():
    con = _with_site_and_file()
    _insert_source(con, "stream", stream_url="rtsp://cam")

    assert con.execute("SELECT status FROM site_sources").fetchone()[0] == "created"


def test_site_sources_rejects_an_unknown_status():
    con = _with_site_and_file()

    with pytest.raises(sqlite3.IntegrityError):
        con.execute(
            "INSERT INTO site_sources (id, site_id, version, kind, stream_url, status)"
            " VALUES ('src1', 's1', 1, 'stream', 'rtsp://cam', 'vibing')"
        )


def test_init_db_is_idempotent():
    con = _fresh()
    init_db(con)  # should not raise


# camera_calibrations and configurations are the same shape, so every structural rule
# is asserted against both rather than trusting them to stay in step.
VERSIONED_DOC_TABLES = ("camera_calibrations", "configurations")


def _seeded(table_and_rows=True):
    con = _fresh()
    con.execute("INSERT INTO sites (id, name) VALUES ('s1', 'A')")
    con.execute(
        "INSERT INTO files (id, name, url, type, status)"
        " VALUES ('f1', 'a.json', 'calibration/f1/a.json', 'calibration', 'uploaded')"
    )
    return con


@pytest.mark.parametrize("table", VERSIONED_DOC_TABLES)
def test_versioned_doc_table_has_the_expected_columns(table):
    con = _fresh()

    columns = _columns(con, table)
    assert columns == {
        "id", "site_id", "file_id", "version", "created_at", "updated_at",
    }


@pytest.mark.parametrize("table", VERSIONED_DOC_TABLES)
def test_versioned_doc_rejects_unknown_site_id(table):
    con = _seeded()

    with pytest.raises(sqlite3.IntegrityError):
        con.execute(
            f"INSERT INTO {table} (id, site_id, file_id, version)"
            " VALUES ('c1', 'no-such-site', 'f1', 1)"
        )


@pytest.mark.parametrize("table", VERSIONED_DOC_TABLES)
def test_versioned_doc_rejects_unknown_file_id(table):
    # The whole point of the file_id switch: a document cannot reference a file that
    # does not exist.
    con = _seeded()

    with pytest.raises(sqlite3.IntegrityError):
        con.execute(
            f"INSERT INTO {table} (id, site_id, file_id, version)"
            " VALUES ('c1', 's1', 'no-such-file', 1)"
        )


@pytest.mark.parametrize("table", VERSIONED_DOC_TABLES)
def test_versioned_doc_rejects_duplicate_version_for_a_site(table):
    con = _seeded()
    con.execute(f"INSERT INTO {table} (id, site_id, file_id, version) VALUES ('c1', 's1', 'f1', 1)")

    with pytest.raises(sqlite3.IntegrityError):
        con.execute(
            f"INSERT INTO {table} (id, site_id, file_id, version) VALUES ('c2', 's1', 'f1', 1)"
        )


def test_init_db_creates_files_table():
    con = _fresh()

    columns = _columns(con, "files")
    assert columns == {
        "id", "name", "url", "type", "status",
        "content_type", "size_bytes", "created_at", "updated_at",
    }


def test_files_defaults_to_pending_status():
    con = _fresh()
    con.execute(
        "INSERT INTO files (id, name, url, type) VALUES ('f1', 'a.mp4', 'video/f1/a.mp4', 'video')"
    )

    assert con.execute("SELECT status FROM files WHERE id = 'f1'").fetchone()[0] == "pending"


def test_files_rejects_unknown_type():
    con = _fresh()

    with pytest.raises(sqlite3.IntegrityError):
        con.execute(
            "INSERT INTO files (id, name, url, type) VALUES ('f1', 'a', 'k', 'not-a-type')"
        )


def test_files_rejects_unknown_status():
    con = _fresh()

    with pytest.raises(sqlite3.IntegrityError):
        con.execute(
            "INSERT INTO files (id, name, url, type, status)"
            " VALUES ('f1', 'a', 'k', 'video', 'halfway')"
        )


def test_site_sources_metadata_survives_a_numeric_looking_document():
    """The reason the column is TEXT and not JSON.

    SQLite gives a JSON-declared column NUMERIC affinity, so a document that happens to
    look like a number is coerced on the way in and comes back as an int — valid JSON
    turning into something Pydantic then refuses. Declaring TEXT is the whole fix, and
    this is what keeps it from being quietly reverted.
    """
    con = _with_site_and_file()
    _insert_source(con, "stream", stream_url="rtsp://cam")
    con.execute("UPDATE site_sources SET metadata = '123' WHERE id = 'src1'")

    value, kind = con.execute(
        "SELECT metadata, typeof(metadata) FROM site_sources WHERE id = 'src1'"
    ).fetchone()
    assert (value, kind) == ("123", "text")


def test_deleting_a_site_cascades_to_everything_that_hangs_off_it():
    """Schema-level now, not application-level.

    delete_site used to issue four DELETEs in a particular order because DuckDB had no
    ON DELETE CASCADE. SQLite does, so the rule moved into init.py — and it belongs in
    a schema test rather than a service test now that the schema is what enforces it.
    """
    con = _with_site_and_file()
    _insert_source(con, "video", file_id="f1")
    for table in VERSIONED_DOC_TABLES:
        con.execute(
            f"INSERT INTO {table} (id, site_id, file_id, version)"
            f" VALUES ('{table[:3]}1', 's1', 'f1', 1)"
        )

    con.execute("DELETE FROM sites WHERE id = 's1'")

    for table in ("site_sources", *VERSIONED_DOC_TABLES):
        assert con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    # The file itself outlives the site that referenced it: nothing cascades to files,
    # which is what keeps a calibration reusable across sites.
    assert con.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 1


VIOLATION = (
    "INSERT INTO traffic_violations"
    " (id, site_id, source_id, frame_index, type, detected_at)"
    " VALUES (?, 's1', 'src1', 912, ?, '2026-08-21 10:00:00')"
)

# Derived from init.ADDED_COLUMNS rather than restated, so a column added there is
# covered by the migration tests below without anyone remembering to widen them. What
# stops the schema and the migration list drifting apart is the literal column set in
# test_init_db_creates_the_violation_tables — that one has to be edited by hand, which
# is the point of it.
ADDED_COLUMNS = {
    column for table, column, _ in init.ADDED_COLUMNS if table == "traffic_violations"
}


def _with_source():
    """A site, a file and the video version a violation can be pinned to."""
    con = _with_site_and_file()
    _insert_source(con, "video", file_id="f1")
    return con


def test_init_db_creates_the_violation_tables():
    con = _fresh()

    assert _columns(con, "traffic_violations") == {
        "id", "site_id", "source_id", "frame_index",
        "calibration_id", "configuration_id",
        "type", "status", "detected_at",
        "explanation", "severity", "created_at", "updated_at",
    }
    assert _columns(con, "violation_metadata") == {
        "id", "traffic_violation_id", "json_blob",
    }


def test_traffic_violations_defaults_to_detected():
    con = _with_source()
    con.execute(VIOLATION, ["v1", "red_light_running"])

    assert con.execute("SELECT status FROM traffic_violations").fetchone()[0] == "detected"


def test_traffic_violations_rejects_an_unknown_type():
    con = _with_source()

    with pytest.raises(sqlite3.IntegrityError):
        con.execute(VIOLATION, ["v1", "jaywalking_backwards"])


def test_a_violation_recorded_before_the_source_columns_existed_keeps_its_row():
    # They are nullable because they arrived after rows did, and a row that predates
    # them genuinely does not know which source it came from. NULL says that; throwing
    # the row away to get a NOT NULL constraint says nothing and loses the record.
    con = _with_source()

    con.execute(
        "INSERT INTO traffic_violations (id, site_id, type, detected_at)"
        " VALUES ('v-old', 's1', 'red_light_running', '2026-08-21 10:00:00')"
    )

    assert con.execute(
        "SELECT source_id, frame_index FROM traffic_violations WHERE id = 'v-old'"
    ).fetchone() == (None, None)


def test_an_existing_database_grows_the_new_columns_without_losing_its_rows():
    # The whole point of _add_missing_columns. CREATE TABLE IF NOT EXISTS does nothing
    # to a table that already exists, and recreating it would throw away what it held.
    con = _fresh()
    con.execute("INSERT INTO sites (id, name) VALUES ('s1', 'Junction 5')")
    con.execute(
        "INSERT INTO traffic_violations (id, site_id, type, detected_at)"
        " VALUES ('v-old', 's1', 'red_light_running', '2026-08-21 10:00:00')"
    )
    for column in ADDED_COLUMNS:
        con.execute(f"ALTER TABLE traffic_violations DROP COLUMN {column}")

    init_db(con)

    assert ADDED_COLUMNS <= _columns(con, "traffic_violations")
    assert con.execute(
        "SELECT COUNT(*) FROM traffic_violations WHERE id = 'v-old'"
    ).fetchone()[0] == 1


def test_bringing_a_database_up_to_date_twice_changes_nothing():
    con = _fresh()

    init_db(con)
    init_db(con)

    assert ADDED_COLUMNS <= _columns(con, "traffic_violations")


def _with_documents():
    """A site with a video, a calibration and a configuration to pin against."""
    con = _with_source()
    for table, doc_id in (("camera_calibrations", "cal1"), ("configurations", "cfg1")):
        con.execute(
            f"INSERT INTO {table} (id, site_id, file_id, version) VALUES (?, 's1', 'f1', 1)",
            [doc_id],
        )
    return con


def test_a_violation_pins_the_calibration_and_configuration_it_was_judged_against():
    con = _with_documents()

    con.execute(
        "INSERT INTO traffic_violations"
        " (id, site_id, source_id, frame_index, calibration_id, configuration_id,"
        "  type, detected_at)"
        " VALUES ('v1', 's1', 'src1', 912, 'cal1', 'cfg1', 'red_light_running',"
        " '2026-08-21 10:00:00')"
    )

    assert con.execute(
        "SELECT calibration_id, configuration_id FROM traffic_violations WHERE id = 'v1'"
    ).fetchone() == ("cal1", "cfg1")


@pytest.mark.parametrize("column", ["calibration_id", "configuration_id"])
def test_traffic_violations_rejects_a_document_that_does_not_exist(column):
    con = _with_documents()

    with pytest.raises(sqlite3.IntegrityError):
        con.execute(
            "INSERT INTO traffic_violations"
            f" (id, site_id, source_id, frame_index, {column}, type, detected_at)"
            " VALUES ('v1', 's1', 'src1', 912, 'no-such-document',"
            " 'red_light_running', '2026-08-21 10:00:00')"
        )


def test_a_violation_from_an_uncalibrated_site_records_no_documents():
    # NULL here is a live state, not a row predating the columns. A site with a video
    # and no calibration is ordinary — DetectionJob carries calibration_version as
    # int | None and detection runs without one — so this has to be insertable.
    con = _with_source()

    con.execute(VIOLATION, ["v1", "red_light_running"])

    assert con.execute(
        "SELECT calibration_id, configuration_id FROM traffic_violations WHERE id = 'v1'"
    ).fetchone() == (None, None)


def test_traffic_violations_rejects_a_source_that_does_not_exist():
    con = _with_source()

    with pytest.raises(sqlite3.IntegrityError):
        con.execute(
            "INSERT INTO traffic_violations"
            " (id, site_id, source_id, frame_index, type, detected_at)"
            " VALUES ('v1', 's1', 'no-such-source', 912, 'red_light_running',"
            " '2026-08-21 10:00:00')"
        )


def test_deleting_a_source_a_violation_points_at_is_refused():
    # A source is configuration; a violation is a record of something that happened.
    # The same reasoning that keeps sites(id) restricting rather than cascading.
    con = _with_source()
    con.execute(VIOLATION, ["v1", "red_light_running"])

    with pytest.raises(sqlite3.IntegrityError):
        con.execute("DELETE FROM site_sources WHERE id = 'src1'")


def test_traffic_violations_rejects_an_unknown_status():
    con = _with_source()

    with pytest.raises(sqlite3.IntegrityError):
        con.execute(
            "INSERT INTO traffic_violations (id, site_id, type, detected_at, status)"
            " VALUES ('v1', 's1', 'red_light_running', '2026-08-21 10:00:00', 'probably')"
        )


def test_traffic_violations_rejects_an_unknown_site():
    con = _with_site_and_file()

    with pytest.raises(sqlite3.IntegrityError):
        con.execute(
            "INSERT INTO traffic_violations (id, site_id, type, detected_at)"
            " VALUES ('v1', 'no-such-site', 'red_light_running', '2026-08-21 10:00:00')"
        )


def test_a_site_with_violations_cannot_be_deleted():
    """The one child of sites that does not cascade.

    Sources and calibrations describe how a site is configured and are meaningless
    without it. A violation is a record of something that happened, and deleting a
    site should not take it along silently.
    """
    con = _with_source()
    con.execute(VIOLATION, ["v1", "red_light_running"])

    with pytest.raises(sqlite3.IntegrityError):
        con.execute("DELETE FROM sites WHERE id = 's1'")


def test_violation_metadata_goes_with_its_violation():
    con = _with_source()
    con.execute(VIOLATION, ["v1", "red_light_running"])
    con.execute(
        "INSERT INTO violation_metadata (id, traffic_violation_id, json_blob)"
        " VALUES ('m1', 'v1', '{}')"
    )

    con.execute("DELETE FROM traffic_violations WHERE id = 'v1'")

    assert con.execute("SELECT COUNT(*) FROM violation_metadata").fetchone()[0] == 0


def test_a_violation_cannot_have_two_metadata_blobs():
    # One-to-one in the LLD. A second blob would be a bug, not an addition, and
    # whichever one a reader picked up would be arbitrary.
    con = _with_source()
    con.execute(VIOLATION, ["v1", "red_light_running"])
    con.execute(
        "INSERT INTO violation_metadata (id, traffic_violation_id, json_blob)"
        " VALUES ('m1', 'v1', '{}')"
    )

    with pytest.raises(sqlite3.IntegrityError):
        con.execute(
            "INSERT INTO violation_metadata (id, traffic_violation_id, json_blob)"
            " VALUES ('m2', 'v1', '{}')"
        )
