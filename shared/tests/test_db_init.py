import duckdb
import pytest

from shared.db.init import init_db


def test_init_db_creates_sites_table():
    con = duckdb.connect(":memory:")
    init_db(con)

    columns = {row[0] for row in con.execute("DESCRIBE sites").fetchall()}
    # Identity only. Everything per-run lives on site_sources, because a site
    # outlives any one video.
    assert columns == {"id", "name", "created_at", "updated_at"}


def test_init_db_creates_site_sources_table():
    con = duckdb.connect(":memory:")
    init_db(con)

    columns = {row[0] for row in con.execute("DESCRIBE site_sources").fetchall()}
    assert columns == {
        "id", "site_id", "version", "kind", "file_id", "stream_url",
        "status", "metadata", "created_at", "updated_at",
    }


def _with_site_and_file():
    con = duckdb.connect(":memory:")
    init_db(con)
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

    with pytest.raises(duckdb.ConstraintException):
        _insert_source(con, kind, stream_url, file_id)


def test_site_sources_rejects_an_unknown_kind():
    con = _with_site_and_file()

    with pytest.raises(duckdb.ConstraintException):
        _insert_source(con, "carrier-pigeon", stream_url="rtsp://cam")


def test_site_sources_rejects_an_unknown_site_id():
    con = _with_site_and_file()

    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "INSERT INTO site_sources (id, site_id, version, kind, stream_url)"
            " VALUES ('src1', 'nope', 1, 'stream', 'rtsp://cam')"
        )


def test_site_sources_rejects_an_unknown_file_id():
    con = _with_site_and_file()

    with pytest.raises(duckdb.ConstraintException):
        _insert_source(con, "video", file_id="no-such-file")


def test_site_sources_rejects_a_duplicate_version_for_a_site():
    con = _with_site_and_file()
    _insert_source(con, "stream", stream_url="rtsp://a", version=1, source_id="src1")

    with pytest.raises(duckdb.ConstraintException):
        _insert_source(con, "stream", stream_url="rtsp://b", version=1, source_id="src2")


def test_site_sources_defaults_to_created_status():
    con = _with_site_and_file()
    _insert_source(con, "stream", stream_url="rtsp://cam")

    assert con.execute("SELECT status FROM site_sources").fetchone()[0] == "created"


def test_site_sources_rejects_an_unknown_status():
    con = _with_site_and_file()

    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "INSERT INTO site_sources (id, site_id, version, kind, stream_url, status)"
            " VALUES ('src1', 's1', 1, 'stream', 'rtsp://cam', 'vibing')"
        )


def test_init_db_is_idempotent():
    con = duckdb.connect(":memory:")
    init_db(con)
    init_db(con)  # should not raise


# camera_calibrations and configurations are the same shape, so every structural rule
# is asserted against both rather than trusting them to stay in step.
VERSIONED_DOC_TABLES = ("camera_calibrations", "configurations")


def _seeded(table_and_rows=True):
    con = duckdb.connect(":memory:")
    init_db(con)
    con.execute("INSERT INTO sites (id, name) VALUES ('s1', 'A')")
    con.execute(
        "INSERT INTO files (id, name, url, type, status)"
        " VALUES ('f1', 'a.json', 'calibration/f1/a.json', 'calibration', 'uploaded')"
    )
    return con


@pytest.mark.parametrize("table", VERSIONED_DOC_TABLES)
def test_versioned_doc_table_has_the_expected_columns(table):
    con = duckdb.connect(":memory:")
    init_db(con)

    columns = {row[0] for row in con.execute(f"DESCRIBE {table}").fetchall()}
    assert columns == {
        "id", "site_id", "file_id", "version", "created_at", "updated_at",
    }


@pytest.mark.parametrize("table", VERSIONED_DOC_TABLES)
def test_versioned_doc_rejects_unknown_site_id(table):
    con = _seeded()

    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            f"INSERT INTO {table} (id, site_id, file_id, version)"
            " VALUES ('c1', 'no-such-site', 'f1', 1)"
        )


@pytest.mark.parametrize("table", VERSIONED_DOC_TABLES)
def test_versioned_doc_rejects_unknown_file_id(table):
    # The whole point of the file_id switch: a document cannot reference a file that
    # does not exist.
    con = _seeded()

    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            f"INSERT INTO {table} (id, site_id, file_id, version)"
            " VALUES ('c1', 's1', 'no-such-file', 1)"
        )


@pytest.mark.parametrize("table", VERSIONED_DOC_TABLES)
def test_versioned_doc_rejects_duplicate_version_for_a_site(table):
    con = _seeded()
    con.execute(f"INSERT INTO {table} (id, site_id, file_id, version) VALUES ('c1', 's1', 'f1', 1)")

    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            f"INSERT INTO {table} (id, site_id, file_id, version) VALUES ('c2', 's1', 'f1', 1)"
        )


def test_init_db_creates_files_table():
    con = duckdb.connect(":memory:")
    init_db(con)

    columns = {row[0] for row in con.execute("DESCRIBE files").fetchall()}
    assert columns == {
        "id", "name", "url", "type", "status",
        "content_type", "size_bytes", "created_at", "updated_at",
    }


def test_files_defaults_to_pending_status():
    con = duckdb.connect(":memory:")
    init_db(con)
    con.execute(
        "INSERT INTO files (id, name, url, type) VALUES ('f1', 'a.mp4', 'video/f1/a.mp4', 'video')"
    )

    assert con.execute("SELECT status FROM files WHERE id = 'f1'").fetchone()[0] == "pending"


def test_files_rejects_unknown_type():
    con = duckdb.connect(":memory:")
    init_db(con)

    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "INSERT INTO files (id, name, url, type) VALUES ('f1', 'a', 'k', 'not-a-type')"
        )


def test_files_rejects_unknown_status():
    con = duckdb.connect(":memory:")
    init_db(con)

    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "INSERT INTO files (id, name, url, type, status)"
            " VALUES ('f1', 'a', 'k', 'video', 'halfway')"
        )
