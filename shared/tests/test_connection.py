import os

from shared.db.connection import get_connection


def test_get_connection_creates_parent_dir_and_connects(tmp_path):
    db_path = tmp_path / "nested" / "traffic.sqlite"
    con = get_connection(str(db_path))

    assert os.path.exists(db_path)
    con.execute("SELECT 1").fetchone()
    con.close()


def test_get_connection_enables_wal_and_foreign_keys(tmp_path):
    """The two pragmas the multi-process story rests on.

    WAL is what lets detection-worker read while site-service writes; foreign keys are
    off by default in SQLite and per connection, so without that pragma every
    REFERENCES clause in init.py is decorative.
    """
    con = get_connection(str(tmp_path / "traffic.sqlite"))

    assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert con.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    con.close()


def test_timestamp_columns_come_back_as_datetimes(tmp_path):
    """Guards the converter registered in shared.db.connection.

    Python 3.12 deprecated the built-in timestamp converter, so this module registers
    its own. If that registration is dropped, this returns a str and every model
    parsing a created_at starts doing the work itself.
    """
    from datetime import datetime

    con = get_connection(str(tmp_path / "traffic.sqlite"))
    con.execute("CREATE TABLE t (id TEXT PRIMARY KEY, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    con.execute("INSERT INTO t (id) VALUES ('a')")

    assert isinstance(con.execute("SELECT created_at FROM t").fetchone()[0], datetime)
    con.close()


def test_a_second_connection_reads_while_the_first_holds_a_write_transaction(tmp_path):
    """The property DuckDB could not give us, and the reason for this migration.

    Under DuckDB a second process could not open the file at all while site-service
    held it. Two connections here stand in for site-service and detection-worker: the
    reader is never blocked by the writer, and sees the row once it commits.
    """
    path = str(tmp_path / "traffic.sqlite")
    writer = get_connection(path)
    writer.execute("CREATE TABLE v (id TEXT PRIMARY KEY)")
    reader = get_connection(path)

    writer.execute("BEGIN IMMEDIATE")
    writer.execute("INSERT INTO v VALUES ('a')")
    # Not blocked, and not seeing uncommitted work either.
    assert reader.execute("SELECT COUNT(*) FROM v").fetchone()[0] == 0
    writer.execute("COMMIT")

    assert reader.execute("SELECT COUNT(*) FROM v").fetchone()[0] == 1
    writer.close()
    reader.close()
