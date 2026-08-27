import pytest
from shared.db.connection import get_connection
from shared.db.init import init_db

from evidence_worker.db import SchemaMissing, connect


def test_connect_opens_a_database_that_has_the_schema(tmp_path):
    path = str(tmp_path / "traffic.sqlite")
    init_db(get_connection(path))  # site-service does this at startup

    con = connect(path)

    assert con.execute("SELECT COUNT(*) FROM traffic_violations").fetchone()[0] == 0


def test_connect_refuses_a_database_with_no_schema(tmp_path):
    """The likely misconfiguration, caught loudly.

    A worker pointed at the wrong path would otherwise create an empty database, find
    no violation for any job, and mark every one of them failed — which reads as broken
    footage rather than a broken path.
    """
    path = str(tmp_path / "wrong-path.sqlite")

    with pytest.raises(SchemaMissing) as exc:
        connect(path)

    assert "TRAFFIC_DB_PATH" in str(exc.value)


def test_connect_looks_for_the_tables_a_cut_actually_joins(tmp_path):
    # Not just traffic_violations: locating a violation's footage joins through
    # site_sources to files, so a database with only the first is still unusable here.
    path = str(tmp_path / "half-a-schema.sqlite")
    con = get_connection(path)
    con.execute("CREATE TABLE traffic_violations (id TEXT)")

    with pytest.raises(SchemaMissing, match="site_sources, files"):
        connect(path)
