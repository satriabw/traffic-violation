import sqlite3

# Identity only. A site is a durable camera location: it outlives any one video, so
# nothing per-run (source, status, metadata) belongs here — see site_sources.
SITES_TABLE = """
CREATE TABLE IF NOT EXISTS sites (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


# What a site is currently pointed at, versioned like calibrations: each POST appends,
# the highest version is active, superseded ones stay readable so a past violation can
# still be traced to the stream or file that produced it.
# References both sites(id) and files(id).
SITE_SOURCES_TABLE = """
CREATE TABLE IF NOT EXISTS site_sources (
    id VARCHAR PRIMARY KEY,
    site_id VARCHAR NOT NULL REFERENCES sites(id) ON DELETE CASCADE,
    version INTEGER NOT NULL DEFAULT 1,
    kind VARCHAR NOT NULL CHECK (kind IN ('video', 'stream')),
    -- A discriminated union keyed on kind. A stream is a user-typed address this
    -- service never resolves; a video is a file whose upload was confirmed. One
    -- generic column for both is what let a video claim a key nobody had uploaded.
    stream_url VARCHAR,
    file_id VARCHAR REFERENCES files(id),
    -- Per-source, not per-site: 'processing' describes one video, and 'active' /
    -- 'degraded' only ever described a stream.
    status VARCHAR NOT NULL DEFAULT 'created' CHECK (
        status IN ('created', 'active', 'processing', 'completed', 'failed', 'degraded')
    ),
    -- TEXT, not JSON: SQLite gives a JSON-declared column NUMERIC affinity, so a
    -- document like '123' would come back as the integer 123. The value is a JSON
    -- string either way — only the affinity differs.
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (site_id, version),
    CHECK (
        (kind = 'video'  AND file_id IS NOT NULL AND stream_url IS NULL) OR
        (kind = 'stream' AND stream_url IS NOT NULL AND file_id IS NULL)
    )
);
"""


# No foreign keys of its own, and created first: sites, camera_calibrations, and
# configurations all reference it.
FILES_TABLE = """
CREATE TABLE IF NOT EXISTS files (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    -- The object key, not a URL, despite the column name inherited from the LLD
    -- (which sites.url and camera_calibrations.url share). Presigned URLs expire,
    -- so they are computed per request and never stored.
    url VARCHAR NOT NULL,
    type VARCHAR NOT NULL CHECK (
        type IN ('calibration', 'configuration', 'video', 'evidence_frame')
    ),
    -- Bytes go client-direct to S3, so a row only becomes a fact once HeadObject
    -- confirms the object landed. Until then it is a claim.
    status VARCHAR NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'uploaded')),
    content_type VARCHAR,
    size_bytes BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


# camera_calibrations and configurations are the same shape — a versioned pointer from
# a site to a file. They stay separate tables because the LLD models them as distinct
# resources; the service layer is written once and parameterised by table name.
_VERSIONED_DOC_TABLE = """
CREATE TABLE IF NOT EXISTS {table} (
    id VARCHAR PRIMARY KEY,
    site_id VARCHAR NOT NULL REFERENCES sites(id) ON DELETE CASCADE,
    file_id VARCHAR NOT NULL REFERENCES files(id),
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (site_id, version)
);
"""

# References both sites(id) and files(id), so both must be created first.
CAMERA_CALIBRATIONS_TABLE = _VERSIONED_DOC_TABLE.format(table="camera_calibrations")
CONFIGURATIONS_TABLE = _VERSIONED_DOC_TABLE.format(table="configurations")


# The product's actual output. References sites(id), so sites must exist first.
#
# Deliberately NOT ON DELETE CASCADE, unlike every other child of sites. Sources,
# calibrations and configurations are configuration — they describe how a site is set
# up, and they are meaningless once it is gone. A violation is a record of something
# that happened. Deleting a site should not silently destroy it, so this FK restricts
# and site-service turns the resulting IntegrityError into a 409.
TRAFFIC_VIOLATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS traffic_violations (
    id VARCHAR PRIMARY KEY,
    site_id VARCHAR NOT NULL REFERENCES sites(id),
    -- WHICH VIDEO, AND WHERE IN IT. What evidence-worker seeks to when it cuts the
    -- thumbnail and the clip below, and the only thing that could ever cut them
    -- again. A row that cannot locate its own footage is a detection nobody can
    -- review, and one nobody can re-derive either.
    --
    -- site_sources appends a row per version and its id is the primary key, so this
    -- pins the exact version on its own — a separate version column would be a second
    -- copy of the same fact, free to disagree with it.
    --
    -- Restricting, like sites(id) above and for the same reason: a source is
    -- configuration, a violation is a record of something that happened, and deleting
    -- the first should not silently destroy the second.
    --
    -- NULLABLE, and that is a deliberate concession rather than a looser rule. These
    -- arrived after rows already existed, and NOT NULL cannot be added to a table with
    -- rows in it without either a default nobody means or throwing the rows away. A
    -- violation recorded before this existed genuinely does not know which source it
    -- came from, and NULL says exactly that. Everything written from now on has both,
    -- because ViolationCreate requires them — the guarantee lives there instead.
    source_id VARCHAR REFERENCES site_sources(id),
    -- Absolute, in the source's own frames. Frames rather than an offset in seconds
    -- because variable-rate footage makes a time offset ambiguous — the same reason
    -- FrameRange is in frames and SourceMetadata keeps fps and nominal_fps apart.
    frame_index INTEGER,
    -- THE BAKED EVIDENCE, and the state of the job that bakes it. Written by
    -- evidence-worker once the violation row exists, never by the detector — cutting a
    -- clip is ffmpeg and a network round trip, and doing it on the GPU box would spend
    -- the one resource that cannot be scaled sideways.
    --
    -- ON THE ROW RATHER THAN IN violation_metadata, which is where the S3 keys of
    -- evidence otherwise live. The list endpoint never joins that table — that is the
    -- entire reason it is a separate one — and a thumbnail the list cannot reach
    -- without a second query per violation is a thumbnail the list cannot render.
    --
    -- Object keys, not URLs, for the reason ViolationMetadata.frames already gives:
    -- presigned links expire, so they are minted per read.
    thumbnail_key VARCHAR,
    clip_key VARCHAR,
    -- 'pending' from the moment the job is queued, one of the other two once the
    -- worker has finished with it.
    --
    -- SEPARATE FROM `status` BELOW, which is about the LLM explanation. The two move
    -- independently — a violation can be explained while its clip is still cutting, or
    -- explained after the cut failed — and one enum covering both would make those
    -- states unrepresentable.
    --
    -- NULL is a fourth state rather than an oversight: every row written before this
    -- column existed has no evidence and no queued job that will ever produce it. A
    -- reader that showed a spinner for those would spin for good, which is why this is
    -- worth distinguishing from 'pending' rather than defaulting.
    evidence_status VARCHAR CHECK (
        evidence_status IN ('pending', 'ready', 'failed')
    ),
    -- WHICH CAMERA MODEL AND WHICH ANNOTATION THIS WAS JUDGED AGAINST. The job message
    -- pins both, so the run is reproducible; without them on the row the record is not.
    -- Two things need them. A reader asking "which violations hold under the setup this
    -- site has now" cannot tell, because nothing says what any of them was judged under.
    -- And evidence drawn with the *current* polygons over a violation found under older
    -- ones shows a vehicle sitting outside the box it was convicted in — evidence that
    -- looks falsified rather than merely stale.
    --
    -- Ids, not version numbers, for the reason source_id gives above: each is the
    -- primary key of one version's row, so it pins that version on its own, and a
    -- separate version column would be a second copy of the same fact free to disagree
    -- with it.
    --
    -- NULLABLE, and here that is not the concession source_id makes. A site with a
    -- video and no calibration is an ordinary site today — DetectionJob already carries
    -- calibration_version as int | None, and detection runs without one. So NULL means
    -- "there was none", a live state rather than a row predating the column, and a
    -- reader filtering on these has to decide what it wants that to mean.
    calibration_id VARCHAR REFERENCES camera_calibrations(id),
    configuration_id VARCHAR REFERENCES configurations(id),
    type VARCHAR NOT NULL CHECK (
        type IN ('red_light_running', 'pedestrian_right_of_way')
    ),
    -- HOW FAR THE EXPLANATION HAS GOT, and the only thing a client polls for it.
    -- 'detected' is the worker's write, and the only one it ever makes. The other
    -- three belong to site-service: 'pending' the moment a request is accepted and
    -- handed to the explanation actor, then 'explained' or 'failed' once that actor is
    -- done with it.
    --
    -- 'failed' rather than reverting to 'detected' on a failure. A violation somebody
    -- asked about and got no answer for is not in the same state as one nobody has
    -- opened, and a reader unable to tell them apart would show a user no trace of a
    -- request they made and waited on.
    --
    -- SEPARATE FROM `evidence_status` ABOVE, which tracks the cut of the footage. The
    -- two move independently and one enum over both would have to enumerate the
    -- product of the two.
    status VARCHAR NOT NULL DEFAULT 'detected' CHECK (
        status IN ('detected', 'pending', 'explained', 'failed')
    ),
    -- When the violation happened in the footage, not when the row was written. The
    -- two differ by however long the job sat in the queue, and only this one is
    -- meaningful to anyone looking at the video.
    detected_at TIMESTAMP NOT NULL,
    explanation VARCHAR,
    severity VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# GET /traffic_violation filters by site and date, which is exactly this pair.
TRAFFIC_VIOLATIONS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_traffic_violations_site_detected
    ON traffic_violations (site_id, detected_at);
"""


# Trajectories, speeds and bounding boxes for one violation, plus the S3 keys of its
# evidence frames. A separate table rather than a column on traffic_violations because
# it is large and the list endpoint never wants it: keeping it out of that row means
# listing violations does not drag every trajectory along with it.
#
# TEXT for the same affinity reason as site_sources.metadata. UNIQUE because the
# relationship is one-to-one — a second blob for the same violation would be a bug
# rather than an addition.
VIOLATION_METADATA_TABLE = """
CREATE TABLE IF NOT EXISTS violation_metadata (
    id VARCHAR PRIMARY KEY,
    traffic_violation_id VARCHAR NOT NULL
        REFERENCES traffic_violations(id) ON DELETE CASCADE,
    json_blob TEXT NOT NULL,
    UNIQUE (traffic_violation_id)
);
"""


# Columns that arrived after the tables did. CREATE TABLE IF NOT EXISTS does nothing
# to a table that already exists, so a database created before one of these was written
# would never grow it — and the alternative, recreating the table, throws away the rows
# it was holding.
#
# ADD COLUMN is the whole mechanism, and it is enough because it is the only kind of
# change made here: existing rows keep everything they had and get NULL for the new
# column. That is why these are nullable and why none of them has a default. A change
# that needed more than this — narrowing a column, adding NOT NULL to one with rows
# under it — would need a real migration tool, and is worth avoiding for exactly that
# reason.
ADDED_COLUMNS = (
    ("traffic_violations", "source_id", "VARCHAR REFERENCES site_sources(id)"),
    ("traffic_violations", "frame_index", "INTEGER"),
    ("traffic_violations", "calibration_id", "VARCHAR REFERENCES camera_calibrations(id)"),
    ("traffic_violations", "configuration_id", "VARCHAR REFERENCES configurations(id)"),
    ("traffic_violations", "thumbnail_key", "VARCHAR"),
    ("traffic_violations", "clip_key", "VARCHAR"),
    (
        "traffic_violations",
        "evidence_status",
        # The same CHECK the CREATE above carries, so a database that grew the column
        # by migration validates exactly like one that was created with it. SQLite
        # applies it to new rows only, which is what leaves the existing NULLs alone.
        "VARCHAR CHECK (evidence_status IN ('pending', 'ready', 'failed'))",
    ),
    # THE WHOLE EXPLANATION, beside the two flat fields that already carry part of it.
    # `explanation` is prose and `severity` is one word, and both are on the row because
    # the list endpoint renders them without joining anything. What neither can hold is
    # the rest of what an explainer returns — what the severity was grounded in, what it
    # observed, and which evidence it distrusted — and that last one is the reason this
    # column exists rather than the first two being deemed enough: an explainer that
    # noticed its own inputs were unreliable has said something a reviewer needs, and
    # with nowhere to put it that doubt either vanishes or leaks into the verdict.
    #
    # JSON in a TEXT column, like violation_metadata.json_blob, and for the same reason:
    # the shape is the explainer's and it will change as the prompt does, which is not a
    # migration anybody should have to run. NULL on every violation nothing has
    # explained, which is all of them until somebody asks.
    ("traffic_violations", "explanation_json", "TEXT"),
)


def _add_missing_columns(con: sqlite3.Connection) -> None:
    """Bring an existing database up to the schema above, without touching its rows.

    Idempotent, and cheap: PRAGMA table_info on a handful of tables, then nothing on
    every run after the first. A foreign key added this way is enforced on new rows and
    not applied retroactively to old ones, which is the behaviour wanted — the old rows
    reference nothing.
    """
    for table, column, ddl in ADDED_COLUMNS:
        present = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
        if column not in present:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


# What the widened `status` CHECK looks like once it is in place, with whitespace
# collapsed. Matching on the *desired* state rather than the old one is what makes the
# migration below idempotent and makes a freshly created database skip it outright.
_WIDENED_STATUS_CHECK = "status IN ('detected', 'pending', 'explained', 'failed')"


def _status_check_is_widened(con: sqlite3.Connection) -> bool:
    row = con.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'traffic_violations'"
    ).fetchone()
    if row is None or row[0] is None:
        # No table yet, so nothing to migrate — init_db is about to create it with the
        # CHECK already widened.
        return True
    return _WIDENED_STATUS_CHECK in " ".join(row[0].split())


def _widen_status_check(con: sqlite3.Connection) -> None:
    """Let `status` hold 'pending' and 'failed', on a database created before it could.

    THE ONLY THING IN THIS MODULE THAT IS NOT ADD COLUMN, and it is here because SQLite
    cannot alter a CHECK constraint. Explaining a violation stopped being something that
    happens inside the request that asked for it, so the column needs two states it was
    never built to hold, and the constraint that was catching bad writes is also what
    refuses the good ones now.

    SQLite's own documented procedure: build the table beside the old one, copy every row
    across by name, drop, rename. The rules it comes with are followed exactly, and each
    one earns its line —

    `foreign_keys` is toggled OUTSIDE the transaction, because the pragma is a no-op
    inside one. With it on, dropping the old table would either fail or cascade into
    violation_metadata, whose rows reference it; with it off, those rows are untouched
    and end up pointing at the new table once it takes the old one's name.

    Columns are copied BY NAME, never `SELECT *`. A database built by CREATE and one
    grown by ALTER hold the same columns in different orders, and positional copying
    would quietly transpose them — putting a clip key in a severity, on the table this
    system exists to keep.

    ADDED_COLUMNS is re-applied to the new table before anything is copied, because the
    CREATE above does not carry all of them — `explanation_json` exists only as a
    migration. Building from the CREATE alone would produce a table with nowhere to put
    it, and every explanation already written would be dropped on the way across.

    `foreign_key_check` runs before the commit rather than after it, so a migration that
    would orphan a row fails while the transaction can still be rolled back.
    """
    if _status_check_is_widened(con):
        return

    con.execute("PRAGMA foreign_keys = OFF")
    try:
        con.execute("BEGIN")
        try:
            con.execute(
                TRAFFIC_VIOLATIONS_TABLE.replace(
                    "traffic_violations", "traffic_violations_new", 1
                )
            )
            for table, column, ddl in ADDED_COLUMNS:
                if table == "traffic_violations":
                    present = {
                        row[1]
                        for row in con.execute("PRAGMA table_info(traffic_violations_new)")
                    }
                    if column not in present:
                        con.execute(
                            f"ALTER TABLE traffic_violations_new ADD COLUMN {column} {ddl}"
                        )

            old = [row[1] for row in con.execute("PRAGMA table_info(traffic_violations)")]
            new = {
                row[1] for row in con.execute("PRAGMA table_info(traffic_violations_new)")
            }
            # The intersection, in the old table's order. Anything the old table has and
            # the new one does not would be a column deliberately removed from the
            # schema, and dropping it here is the only way this ever loses data — so the
            # set is asserted rather than assumed.
            carried = [column for column in old if column in new]
            missing = [column for column in old if column not in new]
            if missing:
                raise RuntimeError(
                    "refusing to migrate traffic_violations: the new schema has no "
                    f"column for {', '.join(missing)}, and the data would be lost"
                )

            columns = ", ".join(carried)
            con.execute(
                f"INSERT INTO traffic_violations_new ({columns}) "
                f"SELECT {columns} FROM traffic_violations"
            )
            con.execute("DROP TABLE traffic_violations")
            con.execute("ALTER TABLE traffic_violations_new RENAME TO traffic_violations")
            # DROP TABLE took the index with it.
            con.execute(TRAFFIC_VIOLATIONS_INDEX)

            broken = con.execute("PRAGMA foreign_key_check").fetchall()
            if broken:
                raise RuntimeError(
                    f"refusing to migrate traffic_violations: {len(broken)} row(s) "
                    "would be left referencing nothing"
                )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
    finally:
        con.execute("PRAGMA foreign_keys = ON")


def init_db(con: sqlite3.Connection) -> None:
    # files first — every other table references it.
    con.execute(FILES_TABLE)
    con.execute(SITES_TABLE)
    con.execute(SITE_SOURCES_TABLE)
    con.execute(CAMERA_CALIBRATIONS_TABLE)
    con.execute(CONFIGURATIONS_TABLE)
    con.execute(TRAFFIC_VIOLATIONS_TABLE)
    con.execute(TRAFFIC_VIOLATIONS_INDEX)
    con.execute(VIOLATION_METADATA_TABLE)
    # After every CREATE, so a fresh database has the tables to inspect and finds
    # nothing missing. An existing one picks up whatever it was built before.
    _add_missing_columns(con)
    # And after that, never before: the rebuild copies whatever columns the table has,
    # so every migration that adds one has to have run first or its data is left behind.
    _widen_status_check(con)
