import os

import duckdb


def get_connection(db_path: str) -> duckdb.DuckDBPyConnection:
    if db_path != ":memory:":
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return duckdb.connect(db_path)
