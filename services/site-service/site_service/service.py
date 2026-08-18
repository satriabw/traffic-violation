import uuid

import duckdb
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
    con.execute("DELETE FROM sites WHERE id = ?", [site_id])
    return True
