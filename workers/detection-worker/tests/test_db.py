import sqlite3

import pytest
from shared.db.connection import get_connection
from shared.db.init import init_db

from detection_worker.db import SchemaMissing, connect


def test_connect_opens_a_database_that_has_the_schema(tmp_path):
    path = str(tmp_path / "traffic.sqlite")
    init_db(get_connection(path))  # site-service does this at startup

    con = connect(path)

    assert con.execute("SELECT COUNT(*) FROM traffic_violations").fetchone()[0] == 0


def test_connect_refuses_a_database_with_no_schema(tmp_path):
    """The likely misconfiguration, caught loudly.

    A worker pointed at the wrong path would otherwise create an empty database, find
    no calibration for any site and record nothing — indistinguishable from quiet
    footage. Refusing to run the DDL here is what turns that into an error.
    """
    path = str(tmp_path / "wrong-path.sqlite")

    with pytest.raises(SchemaMissing) as exc:
        connect(path)

    assert "TRAFFIC_DB_PATH" in str(exc.value)


def test_connect_names_what_was_missing(tmp_path):
    # A database that is a database, but not this one — the message should say which
    # tables it looked for rather than just "no schema".
    path = str(tmp_path / "someone-elses.sqlite")
    get_connection(path).execute("CREATE TABLE unrelated (id TEXT)")

    with pytest.raises(SchemaMissing, match="traffic_violations"):
        connect(path)
