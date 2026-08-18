import os

from shared.db.connection import get_connection


def test_get_connection_creates_parent_dir_and_connects(tmp_path):
    db_path = tmp_path / "nested" / "site.duckdb"
    con = get_connection(str(db_path))

    assert os.path.exists(db_path)
    con.execute("SELECT 1").fetchone()
    con.close()
