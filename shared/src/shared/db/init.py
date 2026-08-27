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
    -- 'explained' is the LLM having filled in explanation and severity, which happens
    -- on demand at read time rather than here. The worker only ever writes 'detected'.
    status VARCHAR NOT NULL DEFAULT 'detected' CHECK (status IN ('detected', 'explained')),
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
