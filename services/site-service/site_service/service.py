import json
import sqlite3
import uuid

from shared import config
from shared.db.violations import get_with_metadata, list_for_setup
from shared.models.file import (
    FileCreate,
    FileResponse,
    FileStatus,
    FileType,
    FileUploadResponse,
)
from shared.models.versioned_document import VersionedDocument
from shared.models.site import SiteCreate, SiteListResponse, SiteResponse
from shared.models.source import SourceCreate, SourceMetadata, SourceResponse
from shared.models.violation import (
    ViolationExplanation,
    ViolationListResponse,
    ViolationResponse,
)
from shared.s3.keys import build_key

_COLUMNS = ("id", "name", "created_at", "updated_at")


def _row_to_site(con: sqlite3.Connection, row: tuple) -> SiteResponse:
    data = dict(zip(_COLUMNS, row))
    # The active source is embedded so the common read is one request.
    return SiteResponse(**data, source=get_active_source(con, data["id"]))


def create_site(
    con: sqlite3.Connection,
    data: SiteCreate,
    source_metadata: SourceMetadata | None = None,
) -> SiteResponse:
    site_id = str(uuid.uuid4())
    con.execute("INSERT INTO sites (id, name) VALUES (?, ?)", [site_id, data.name])
    if data.source is not None:
        # Sugar for the caller, not a second code path: the inline source goes through
        # the same function POST /sites/{id}/sources uses, metadata included.
        create_source(con, site_id, data.source, source_metadata)
    return get_site(con, site_id)


def get_site(con: sqlite3.Connection, site_id: str) -> SiteResponse | None:
    row = con.execute(
        f"SELECT {', '.join(_COLUMNS)} FROM sites WHERE id = ?", [site_id]
    ).fetchone()
    return _row_to_site(con, row) if row else None


def list_sites(
    con: sqlite3.Connection,
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


def delete_site(con: sqlite3.Connection, site_id: str) -> bool:
    if get_site(con, site_id) is None:
        return False
    # Sources, calibrations and configurations go with it. That is ON DELETE CASCADE
    # in the schema now rather than four statements here — see init.py. The file rows
    # they point at are deliberately not cascaded: a file outlives the site that
    # referenced it, and nothing deletes files yet.
    #
    # Violations are the exception and do not cascade, so this raises rather than
    # quietly destroying them. Checked up front rather than caught, because the
    # IntegrityError alone cannot say *which* reference stood in the way.
    if has_violations(con, site_id):
        raise SiteHasViolations(site_id)
    con.execute("DELETE FROM sites WHERE id = ?", [site_id])
    return True


class SiteHasViolations(Exception):
    """A site cannot be deleted while it has violations recorded against it.

    Configuration is disposable; a record of something that happened is not. Deleting
    the site would take the violations with it, so the caller has to deal with them
    first rather than have that happen as a side effect.
    """

    def __init__(self, site_id: str):
        super().__init__(f"site {site_id} has recorded violations")
        self.site_id = site_id


def has_violations(con: sqlite3.Connection, site_id: str) -> bool:
    return (
        con.execute(
            "SELECT EXISTS (SELECT 1 FROM traffic_violations WHERE site_id = ?)",
            [site_id],
        ).fetchone()[0]
        == 1
    )


def _next_version(con: sqlite3.Connection, table: str, site_id: str) -> int:
    """Next version number for a site in a versioned table.

    The one piece genuinely shared by sources, calibrations, and configurations —
    their column sets differ too much to share more.

    Read-then-insert rather than a sequence. What makes that safe is not that there is
    one writer — there no longer is — but UNIQUE (site_id, version): two racing
    appends cannot both land, and the loser gets an IntegrityError rather than a
    duplicate version.
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
    con: sqlite3.Connection, file_id: str, expected_type: FileType
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
    con: sqlite3.Connection, table: str, site_id: str, file_id: str
) -> VersionedDocument:
    """Append a new version. Callers validate the file first — see unusable_file_reason."""
    doc_id = str(uuid.uuid4())
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
    con: sqlite3.Connection, table: str, site_id: str
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
    con: sqlite3.Connection, table: str, site_id: str, doc_id: str
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


def create_file(con: sqlite3.Connection, storage, data: FileCreate) -> FileUploadResponse:
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


def get_file(con: sqlite3.Connection, storage, file_id: str) -> FileResponse | None:
    row = con.execute(
        f"SELECT {', '.join(_FILE_COLUMNS)} FROM files WHERE id = ?", [file_id]
    ).fetchone()
    if row is None:
        return None
    status = dict(zip(_FILE_COLUMNS, row))["status"]
    return _row_to_file(row, storage, download=status == FileStatus.UPLOADED.value)


def confirm_upload(
    con: sqlite3.Connection, storage, file_id: str
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
    data = dict(zip(_SOURCE_COLUMNS, row))
    # The column is TEXT, and Pydantic will not coerce a string into a nested model,
    # so the decode is explicit rather than implied.
    if isinstance(data.get("metadata"), str):
        data["metadata"] = json.loads(data["metadata"])
    return SourceResponse(**data)


def create_source(
    con: sqlite3.Connection,
    site_id: str,
    data: SourceCreate,
    metadata: SourceMetadata | None = None,
) -> SourceResponse:
    """Append a new source version.

    Callers validate a video's file and read its metadata first — see
    site_service.routers.source.validate_source. Both arrive here as plain data so
    this layer stays pure database work with no network of its own.
    """
    source_id = str(uuid.uuid4())
    version = _next_version(con, "site_sources", site_id)
    # Exactly one of stream_url / file_id is set: SourceCreate's validator and the
    # table CHECK both guarantee it, so no branching is needed here.
    con.execute(
        """
        INSERT INTO site_sources (id, site_id, version, kind, stream_url, file_id, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            source_id, site_id, version, data.kind.value, data.stream_url, data.file_id,
            # NULL means "never probed", which is why unset fields are dropped rather
            # than written as nulls inside the document.
            json.dumps(metadata.model_dump(exclude_none=True)) if metadata else None,
        ],
    )
    return get_source(con, site_id, source_id)


def get_active_source(
    con: sqlite3.Connection, site_id: str
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
    con: sqlite3.Connection, site_id: str, source_id: str
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


def _evidence_url(storage, key: str | None) -> str | None:
    """A signed link for one evidence object, or None when there is no object.

    The same shape as _row_to_file's download link and for the same reason: a presigned
    URL expires, so it is minted at the moment of reading rather than stored beside the
    key. A falsy key is every state except a finished cut — see ViolationResponse.
    """
    return storage.download_url(key) if key else None


def _row_to_violation(row: dict, storage) -> ViolationResponse:
    # The dict's keys are already the model's field names, so there is no column list
    # here to fall out of step with the one shared.db.violations selects.
    return ViolationResponse(
        **row,
        thumbnail_url=_evidence_url(storage, row["thumbnail_key"]),
        clip_url=_evidence_url(storage, row["clip_key"]),
    )


def get_violation(
    con: sqlite3.Connection, storage, site_id: str, violation_id: str
) -> ViolationResponse | None:
    """One violation, with everything the list leaves out.

    SCOPED TO THE SITE IN THE PATH, and the check is not a formality. The violation id
    is a uuid and the route is nested, so a caller holding an id from one site could
    otherwise read it through another site's URL — and every reader that trusts the
    path to mean what it says would be wrong. A mismatch is None here and 404 at the
    router, deliberately the same answer as an id that does not exist: telling the two
    apart would confirm the violation is real and belongs to somebody else.

    NO SETUP FILTER, unlike the list. That one answers "what holds under the setup this
    site runs now" and hides violations judged under a superseded calibration; this
    answers "what is this violation", and a reader who has an id in their hand is
    entitled to it whether or not it still describes the current setup. The row carries
    the ids it was judged under, so a reader can see for themselves.

    The metadata blob comes back here and nowhere else. `explanation_json` is parsed
    into `explanation_detail` on the way out rather than handed over as text — the
    column is storage, the model is the contract.
    """
    row = get_with_metadata(con, violation_id)
    if row is None or row["site_id"] != site_id:
        return None
    row = dict(row)
    explanation = row.pop("explanation_json")
    return ViolationResponse(
        **row,
        explanation_detail=(
            ViolationExplanation.model_validate_json(explanation) if explanation else None
        ),
        thumbnail_url=_evidence_url(storage, row["thumbnail_key"]),
        clip_url=_evidence_url(storage, row["clip_key"]),
    )


def list_violations(
    con: sqlite3.Connection, storage, site_id: str, limit: int, offset: int
) -> ViolationListResponse:
    """A page of the violations that hold under the setup this site runs now.

    THE SETUP IS IMPLIED, not asked for. The calibration and configuration are resolved
    here to the site's active versions — the same get_active_version the detection
    endpoint calls to pin a job's versions — so the list filters on the same notion of
    "active" that detection runs under rather than a second one free to disagree with
    it. A caller cannot ask for a superseded setup, which is deliberate: the question
    this answers is "what holds now", and every other question wants the detail view.

    A SITE WITH NO CALIBRATION MATCHES THE VIOLATIONS THAT HAD NONE, because None is
    what its active version resolves to and the filter is null-safe. That also sweeps in
    anything recorded before those columns existed — the row cannot tell the two causes
    apart, and this takes the permissive reading. See shared.db.init.

    `metadata` stays None on every item. The blob is a separate table so that a page of
    violations does not carry every track's trajectory, and nothing here joins it.
    """
    calibration = get_active_version(con, CALIBRATIONS, site_id)
    configuration = get_active_version(con, CONFIGURATIONS, site_id)
    rows, total = list_for_setup(
        con,
        site_id,
        calibration.id if calibration else None,
        configuration.id if configuration else None,
        limit=limit,
        offset=offset,
    )
    return ViolationListResponse(
        items=[_row_to_violation(row, storage) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )
