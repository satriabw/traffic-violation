import uuid

import duckdb
from shared.models.calibration import CalibrationCreate, CalibrationResponse
from shared.models.site import SiteCreate, SiteListResponse, SiteResponse

_COLUMNS = ("id", "name", "url", "mode", "status", "metadata", "created_at", "updated_at")


def _row_to_site(row: tuple) -> SiteResponse:
    data = dict(zip(_COLUMNS, row))
    return SiteResponse(**data)


def create_site(con: duckdb.DuckDBPyConnection, data: SiteCreate) -> SiteResponse:
    site_id = str(uuid.uuid4())
    con.execute(
        """
        INSERT INTO sites (id, name, url, mode)
        VALUES (?, ?, ?, ?)
        """,
        [site_id, data.name, data.url, data.mode.value],
    )
    return get_site(con, site_id)


def get_site(con: duckdb.DuckDBPyConnection, site_id: str) -> SiteResponse | None:
    row = con.execute(
        f"SELECT {', '.join(_COLUMNS)} FROM sites WHERE id = ?", [site_id]
    ).fetchone()
    return _row_to_site(row) if row else None


def list_sites(
    con: duckdb.DuckDBPyConnection,
    limit: int,
    offset: int,
    mode: str | None = None,
    status: str | None = None,
) -> SiteListResponse:
    where_clauses = []
    params: list = []
    if mode is not None:
        where_clauses.append("mode = ?")
        params.append(mode)
    if status is not None:
        where_clauses.append("status = ?")
        params.append(status)
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    total = con.execute(f"SELECT COUNT(*) FROM sites {where_sql}", params).fetchone()[0]
    rows = con.execute(
        f"""
        SELECT {', '.join(_COLUMNS)} FROM sites {where_sql}
        ORDER BY created_at
        LIMIT ? OFFSET ?
        """,
        [*params, limit, offset],
    ).fetchall()

    return SiteListResponse(
        items=[_row_to_site(row) for row in rows], total=total, limit=limit, offset=offset
    )


def delete_site(con: duckdb.DuckDBPyConnection, site_id: str) -> bool:
    if get_site(con, site_id) is None:
        return False
    # DuckDB enforces the camera_calibrations -> sites foreign key but has no
    # ON DELETE CASCADE, so the children have to go first. Do NOT wrap these two
    # statements in a transaction: DuckDB's FK check does not see uncommitted child
    # deletes, so the site delete then fails with a ConstraintException. Autocommit
    # is what makes this work.
    con.execute("DELETE FROM camera_calibrations WHERE site_id = ?", [site_id])
    con.execute("DELETE FROM sites WHERE id = ?", [site_id])
    return True


_CALIBRATION_COLUMNS = ("id", "site_id", "url", "version", "created_at", "updated_at")


def _row_to_calibration(row: tuple) -> CalibrationResponse:
    return CalibrationResponse(**dict(zip(_CALIBRATION_COLUMNS, row)))


def create_calibration(
    con: duckdb.DuckDBPyConnection, site_id: str, data: CalibrationCreate
) -> CalibrationResponse:
    calibration_id = str(uuid.uuid4())
    # site-service is the only writer of this DuckDB file, so read-then-insert is
    # safe here and no sequence is needed.
    version = con.execute(
        "SELECT COALESCE(MAX(version), 0) + 1 FROM camera_calibrations WHERE site_id = ?",
        [site_id],
    ).fetchone()[0]
    con.execute(
        """
        INSERT INTO camera_calibrations (id, site_id, url, version)
        VALUES (?, ?, ?, ?)
        """,
        [calibration_id, site_id, data.url, version],
    )
    return get_calibration(con, site_id, calibration_id)


def get_active_calibration(
    con: duckdb.DuckDBPyConnection, site_id: str
) -> CalibrationResponse | None:
    row = con.execute(
        f"""
        SELECT {', '.join(_CALIBRATION_COLUMNS)} FROM camera_calibrations
        WHERE site_id = ?
        ORDER BY version DESC
        LIMIT 1
        """,
        [site_id],
    ).fetchone()
    return _row_to_calibration(row) if row else None


def get_calibration(
    con: duckdb.DuckDBPyConnection, site_id: str, calibration_id: str
) -> CalibrationResponse | None:
    # Scoped by both ids so one site can never read another site's calibration.
    row = con.execute(
        f"""
        SELECT {', '.join(_CALIBRATION_COLUMNS)} FROM camera_calibrations
        WHERE site_id = ? AND id = ?
        """,
        [site_id, calibration_id],
    ).fetchone()
    return _row_to_calibration(row) if row else None
