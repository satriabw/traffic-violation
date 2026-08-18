import duckdb

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
