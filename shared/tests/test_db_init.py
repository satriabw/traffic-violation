import duckdb
import pytest

from shared.db.init import init_db


def test_init_db_creates_sites_table():
    con = duckdb.connect(":memory:")
    init_db(con)

    columns = {row[0] for row in con.execute("DESCRIBE sites").fetchall()}
    assert columns == {
        "id", "name", "url", "mode", "status",
        "metadata", "created_at", "updated_at",
    }


def test_init_db_is_idempotent():
    con = duckdb.connect(":memory:")
    init_db(con)
    init_db(con)  # should not raise


def test_init_db_creates_camera_calibrations_table():
    con = duckdb.connect(":memory:")
    init_db(con)

    columns = {row[0] for row in con.execute("DESCRIBE camera_calibrations").fetchall()}
    assert columns == {
        "id", "site_id", "url", "version", "created_at", "updated_at",
    }


def test_camera_calibrations_rejects_unknown_site_id():
    con = duckdb.connect(":memory:")
    init_db(con)

    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "INSERT INTO camera_calibrations (id, site_id, url, version)"
            " VALUES ('c1', 'no-such-site', 's3://a', 1)"
        )


def test_camera_calibrations_rejects_duplicate_version_for_a_site():
    con = duckdb.connect(":memory:")
    init_db(con)
    con.execute("INSERT INTO sites (id, name, url, mode) VALUES ('s1', 'A', 's3://v', 'video')")
    con.execute(
        "INSERT INTO camera_calibrations (id, site_id, url, version)"
        " VALUES ('c1', 's1', 's3://a', 1)"
    )

    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "INSERT INTO camera_calibrations (id, site_id, url, version)"
            " VALUES ('c2', 's1', 's3://b', 1)"
        )
