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

# References sites(id), so it must be created after SITES_TABLE.
CAMERA_CALIBRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS camera_calibrations (
    id VARCHAR PRIMARY KEY,
    site_id VARCHAR NOT NULL REFERENCES sites(id),
    url VARCHAR NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (site_id, version)
);
"""


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SITES_TABLE)
    con.execute(CAMERA_CALIBRATIONS_TABLE)
