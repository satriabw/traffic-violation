import duckdb

SITES_TABLE = """
CREATE TABLE IF NOT EXISTS sites (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    url VARCHAR NOT NULL,
    mode VARCHAR NOT NULL CHECK (mode IN ('video', 'stream')),
    status VARCHAR NOT NULL DEFAULT 'created' CHECK (
        status IN ('created', 'active', 'processing', 'completed', 'failed', 'degraded')
    ),
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SITES_TABLE)
