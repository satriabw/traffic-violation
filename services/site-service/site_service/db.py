import duckdb
from shared.config import DB_PATH
from shared.db.connection import get_connection
from shared.db.init import init_db

_connection: duckdb.DuckDBPyConnection | None = None


def init_app_db() -> None:
    global _connection
    _connection = get_connection(DB_PATH)
    init_db(_connection)


def get_db() -> duckdb.DuckDBPyConnection:
    return _connection
