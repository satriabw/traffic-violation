import uuid

import duckdb
from shared import config
from shared.models.file import (
    FileCreate,
    FileResponse,
    FileStatus,
    FileType,
    FileUploadResponse,
)
from shared.models.versioned_document import VersionedDocument
from shared.models.site import SiteCreate, SiteListResponse, SiteResponse
from shared.models.source import SourceCreate, SourceResponse
from shared.s3.keys import build_key

_COLUMNS = ("id", "name", "created_at", "updated_at")


def _row_to_site(con: duckdb.DuckDBPyConnection, row: tuple) -> SiteResponse:
    data = dict(zip(_COLUMNS, row))
    # The active source is embedded so the common read is one request.
    return SiteResponse(**data, source=get_active_source(con, data["id"]))


def create_site(con: duckdb.DuckDBPyConnection, data: SiteCreate) -> SiteResponse:
    site_id = str(uuid.uuid4())
    con.execute("INSERT INTO sites (id, name) VALUES (?, ?)", [site_id, data.name])
    if data.source is not None:
        # Sugar for the caller, not a second code path: the inline source goes through
        # the same function POST /sites/{id}/sources uses.
        create_source(con, site_id, data.source)
    return get_site(con, site_id)


def get_site(con: duckdb.DuckDBPyConnection, site_id: str) -> SiteResponse | None:
    row = con.execute(
        f"SELECT {', '.join(_COLUMNS)} FROM sites WHERE id = ?", [site_id]
    ).fetchone()
    return _row_to_site(con, row) if row else None


def list_sites(
    con: duckdb.DuckDBPyConnection,
    limit: int,
    offset: int,
    kind: str | None = None,
    status: str | None = None,
) -> SiteListResponse:
    where_clauses = []
    params: list = []
    # kind and status describe the *active* source, so both filter against the highest
    # version rather than any row in the history: a site that used to be a video and is
    # now a stream is a stream site.
    for column, value in (("kind", kind), ("status", status)):
        if value is None:
            continue
        where_clauses.append(
            f"""
            EXISTS (
                SELECT 1 FROM site_sources src
                WHERE src.site_id = sites.id
                  AND src.{column} = ?
                  AND src.version = (
                      SELECT MAX(version) FROM site_sources WHERE site_id = sites.id
                  )
            )
            """
        )
        params.append(value)
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
        items=[_row_to_site(con, row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


def delete_site(con: duckdb.DuckDBPyConnection, site_id: str) -> bool:
    if get_site(con, site_id) is None:
        return False
    # DuckDB enforces the child -> sites foreign keys but has no
    # ON DELETE CASCADE, so the children have to go first. Do NOT wrap these two
    # statements in a transaction: DuckDB's FK check does not see uncommitted child
    # deletes, so the site delete then fails with a ConstraintException. Autocommit
    # is what makes this work.
    con.execute("DELETE FROM site_sources WHERE site_id = ?", [site_id])
    con.execute(f"DELETE FROM {CALIBRATIONS} WHERE site_id = ?", [site_id])
    con.execute(f"DELETE FROM {CONFIGURATIONS} WHERE site_id = ?", [site_id])
    con.execute("DELETE FROM sites WHERE id = ?", [site_id])
    return True


def _next_version(con: duckdb.DuckDBPyConnection, table: str, site_id: str) -> int:
    """Next version number for a site in a versioned table.

    The one piece genuinely shared by sources, calibrations, and configurations —
    their column sets differ too much to share more. site-service is the only writer
    of this DuckDB file, so read-then-insert is safe and no sequence is needed.
    """
    return con.execute(
        f"SELECT COALESCE(MAX(version), 0) + 1 FROM {table} WHERE site_id = ?", [site_id]
    ).fetchone()[0]


# The two versioned-document tables. Module constants, never request data — they are
# interpolated into SQL below the same way _COLUMNS already is.
CALIBRATIONS = "camera_calibrations"
CONFIGURATIONS = "configurations"

_VERSIONED_DOC_COLUMNS = ("id", "site_id", "file_id", "version", "created_at", "updated_at")


def _row_to_versioned_doc(row: tuple) -> VersionedDocument:
    return VersionedDocument(**dict(zip(_VERSIONED_DOC_COLUMNS, row)))


def unusable_file_reason(
    con: duckdb.DuckDBPyConnection, file_id: str, expected_type: FileType
) -> str | None:
    """Why this file may not be attached to a site, or None if it may.

    Returns a reason rather than raising so the caller decides the status code — the
    service layer has no business knowing about HTTP.
    """
    row = con.execute("SELECT type, status FROM files WHERE id = ?", [file_id]).fetchone()
    if row is None:
        return "missing"
    file_type, status = row
    # A pending row is a reserved slot, not a file. Attaching one would recreate
    # exactly the unverifiable state the old url column allowed.
    if status != FileStatus.UPLOADED.value:
        return "pending"
    if file_type != expected_type.value:
        return "wrong_type"
    return None


def create_versioned_doc(
    con: duckdb.DuckDBPyConnection, table: str, site_id: str, file_id: str
) -> VersionedDocument:
    """Append a new version. Callers validate the file first — see unusable_file_reason."""
    doc_id = str(uuid.uuid4())
    # site-service is the only writer of this DuckDB file, so read-then-insert is
    # safe here and no sequence is needed.
    version = _next_version(con, table, site_id)
    con.execute(
        f"""
        INSERT INTO {table} (id, site_id, file_id, version)
        VALUES (?, ?, ?, ?)
        """,
        [doc_id, site_id, file_id, version],
    )
    return get_version(con, table, site_id, doc_id)


def get_active_version(
    con: duckdb.DuckDBPyConnection, table: str, site_id: str
) -> VersionedDocument | None:
    """The one valid document for a site — the highest version."""
    row = con.execute(
        f"""
        SELECT {', '.join(_VERSIONED_DOC_COLUMNS)} FROM {table}
        WHERE site_id = ?
        ORDER BY version DESC
        LIMIT 1
        """,
        [site_id],
    ).fetchone()
    return _row_to_versioned_doc(row) if row else None


def get_version(
    con: duckdb.DuckDBPyConnection, table: str, site_id: str, doc_id: str
) -> VersionedDocument | None:
    # Scoped by both ids so one site can never read another site's documents.
    row = con.execute(
        f"""
        SELECT {', '.join(_VERSIONED_DOC_COLUMNS)} FROM {table}
        WHERE site_id = ? AND id = ?
        """,
        [site_id, doc_id],
    ).fetchone()
    return _row_to_versioned_doc(row) if row else None


_FILE_COLUMNS = (
    "id", "name", "url", "type", "status",
    "content_type", "size_bytes", "created_at", "updated_at",
)


def _row_to_file(row: tuple, storage, download: bool) -> FileResponse:
    data = dict(zip(_FILE_COLUMNS, row))
    # Presigned URLs expire, so the download link is minted per read rather than
    # stored, and only once there is actually an object behind it.
    download_url = storage.download_url(data["url"]) if download else None
    return FileResponse(**data, download_url=download_url)


def create_file(con: duckdb.DuckDBPyConnection, storage, data: FileCreate) -> FileUploadResponse:
    """Reserve a key and hand back a URL the client can PUT to directly.

    The row starts `pending`: nothing has been uploaded yet, and only confirm_upload
    can say otherwise. size_bytes here is the client's *declared* size, replaced with
    the measured one at confirm time.
    """
    file_id = str(uuid.uuid4())
    # The key is derived server-side from a sanitised name — the client never
    # supplies a path.
    key = build_key(data.type.value, file_id, data.name)
    con.execute(
        """
        INSERT INTO files (id, name, url, type, content_type, size_bytes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [file_id, data.name, key, data.type.value, data.content_type, data.size_bytes],
    )
    row = con.execute(
        f"SELECT {', '.join(_FILE_COLUMNS)} FROM files WHERE id = ?", [file_id]
    ).fetchone()
    return FileUploadResponse(
        **_row_to_file(row, storage, download=False).model_dump(),
        upload_url=storage.presigned_put(
            key, content_type=data.content_type, content_length=data.size_bytes
        ),
        upload_expires_in=config.S3_PRESIGN_EXPIRY_SECONDS,
    )


def get_file(con: duckdb.DuckDBPyConnection, storage, file_id: str) -> FileResponse | None:
    row = con.execute(
        f"SELECT {', '.join(_FILE_COLUMNS)} FROM files WHERE id = ?", [file_id]
    ).fetchone()
    if row is None:
        return None
    status = dict(zip(_FILE_COLUMNS, row))["status"]
    return _row_to_file(row, storage, download=status == FileStatus.UPLOADED.value)


def confirm_upload(
    con: duckdb.DuckDBPyConnection, storage, file_id: str
) -> FileResponse | None:
    """Verify the client's upload actually landed, then mark the row uploaded.

    Returns None when the object is *not in storage*. Callers are expected to have
    already established that the row exists, so None here is never "unknown file".
    Safe to call twice: HeadObject still succeeds and the write is idempotent.
    """
    current = get_file(con, storage, file_id)
    if current is None:
        return None
    head = storage.head(current.url)
    if head is None:
        return None
    con.execute(
        """
        UPDATE files
        SET status = ?, size_bytes = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        [FileStatus.UPLOADED.value, head.get("ContentLength"), file_id],
    )
    return get_file(con, storage, file_id)


_SOURCE_COLUMNS = (
    "id", "site_id", "version", "kind", "stream_url", "file_id",
    "status", "metadata", "created_at", "updated_at",
)


def _row_to_source(row: tuple) -> SourceResponse:
    return SourceResponse(**dict(zip(_SOURCE_COLUMNS, row)))


def create_source(
    con: duckdb.DuckDBPyConnection, site_id: str, data: SourceCreate
) -> SourceResponse:
    """Append a new source version. Callers validate a video's file first — see
    site_service.file_reference."""
    source_id = str(uuid.uuid4())
    version = _next_version(con, "site_sources", site_id)
    # Exactly one of stream_url / file_id is set: SourceCreate's validator and the
    # table CHECK both guarantee it, so no branching is needed here.
    con.execute(
        """
        INSERT INTO site_sources (id, site_id, version, kind, stream_url, file_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [source_id, site_id, version, data.kind.value, data.stream_url, data.file_id],
    )
    return get_source(con, site_id, source_id)


def get_active_source(
    con: duckdb.DuckDBPyConnection, site_id: str
) -> SourceResponse | None:
    """What the site is currently pointed at — the highest version."""
    row = con.execute(
        f"""
        SELECT {', '.join(_SOURCE_COLUMNS)} FROM site_sources
        WHERE site_id = ?
        ORDER BY version DESC
        LIMIT 1
        """,
        [site_id],
    ).fetchone()
    return _row_to_source(row) if row else None


def get_source(
    con: duckdb.DuckDBPyConnection, site_id: str, source_id: str
) -> SourceResponse | None:
    # Scoped by both ids so one site can never read another site's sources.
    row = con.execute(
        f"""
        SELECT {', '.join(_SOURCE_COLUMNS)} FROM site_sources
        WHERE site_id = ? AND id = ?
        """,
        [site_id, source_id],
    ).fetchone()
    return _row_to_source(row) if row else None
