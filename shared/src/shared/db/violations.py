"""Reading and writing the violation tables.

Here rather than in detection-worker because there are two writers now. The detector
records the violation; evidence-worker comes back afterwards and fills in the cut of
the footage that goes with it. Two processes, one pair of tables, and the SQL that
touches them belongs in the place they both already depend on rather than copied into
each — a second copy of an INSERT is a second copy free to fall behind the schema.

There is one reader too, and it is here for the weaker version of the same reason: the
column set it selects has to agree with the one `record` inserts, and site-service
holding its own copy of that list is a copy free to drift. What it does NOT do is
decide anything — the setup it filters on arrives already resolved.

Reads of a site's calibration and configuration are NOT here. Those resolve by version
and are only ever wanted by a job that is about to run, which is what keeps them in
detection_worker.context.
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass

from shared.models.violation import EvidenceStatus, ViolationCreate


# Everything the list read carries, which is every column on the row and none of the
# blob beside it. Names are the model's field names, so a caller builds a
# ViolationResponse straight off the dict without a translation table in between.
_LIST_COLUMNS = (
    "id", "site_id", "source_id", "frame_index",
    "calibration_id", "configuration_id",
    "type", "status", "detected_at", "explanation", "severity",
    "thumbnail_key", "clip_key", "evidence_status",
    "created_at", "updated_at",
)

# `IS` on the two document ids, NOT `=`, and that is the whole correctness of this
# filter. A site with no calibration resolves to None, and `calibration_id = NULL` is
# NULL rather than true for every row — so the ordinary case of a site running without
# a camera model would return an empty page while holding violations. `IS` is SQLite's
# null-safe equality and binds a parameter the same way, so one clause covers both
# "judged under this calibration" and "judged under none".
_SETUP_WHERE = "site_id = ? AND calibration_id IS ? AND configuration_id IS ?"


def list_for_setup(
    con: sqlite3.Connection,
    site_id: str,
    calibration_id: str | None,
    configuration_id: str | None,
    limit: int,
    offset: int,
) -> tuple[list[dict], int]:
    """One page of a site's violations judged under one setup, and how many there are.

    THE IDS ARE PASSED IN, ALREADY RESOLVED. Deciding which calibration is *active* is
    a read of camera_calibrations, and this module deliberately does not do those — see
    the note at the top. It also keeps the meaning of the filter at the caller: a
    reader that wants the setup a site runs now and one that wants the setup a
    particular job ran under ask the same question here with different arguments.

    NO JOIN ON violation_metadata, which is the reason that table exists. A page of
    violations carrying every track's trajectory is ~13.5KB per track per row, and the
    thumbnail a list actually wants is on this row precisely so the list can render
    without reaching for it.

    Two statements rather than a window function: the count is over the whole filter
    and the page is a slice of it, and a caller paginating needs both. They are read
    on one autocommit connection with no writer in between worth serialising against —
    a violation appearing between them costs a total that is low by one, not a page
    that disagrees with itself.

    Dicts rather than rows, keyed by column name. The tuple order is this module's
    business and nobody else's, and a caller in another distribution zipping a column
    list it imported from here would break silently the day a column moves.
    """
    params = [site_id, calibration_id, configuration_id]
    total = con.execute(
        f"SELECT COUNT(*) FROM traffic_violations WHERE {_SETUP_WHERE}", params
    ).fetchone()[0]
    rows = con.execute(
        f"""
        SELECT {', '.join(_LIST_COLUMNS)} FROM traffic_violations
        WHERE {_SETUP_WHERE}
        ORDER BY detected_at DESC, id
        LIMIT ? OFFSET ?
        """,
        # `id` breaks the tie, and it is not decoration: several violations routinely
        # share a detected_at — one frame can fire a rule for more than one track — and
        # an ORDER BY they all tie on lets SQLite return them in any order it likes per
        # statement. Two pages read that way can repeat a violation and skip another.
        [*params, limit, offset],
    ).fetchall()
    return [dict(zip(_LIST_COLUMNS, row)) for row in rows], total


def record(con: sqlite3.Connection, violation: ViolationCreate) -> str:
    """Write the violation and its metadata, and return the new id.

    NO EVIDENCE IS WRITTEN HERE, and the row leaves with `evidence_status` NULL rather
    than 'pending'. What the detector knows is that a rule fired; whether anybody has
    queued a cut of the footage for it is somebody else's fact, and stamping 'pending'
    on the way past would promise a job that may never have been enqueued. The row pins
    the source and the frame index, which is what lets the cut be made later — or made
    again.

    The two inserts share one transaction. Connections here are autocommit, so the
    BEGIN is explicit — without it a crash between the statements leaves a violation
    with no trajectories behind it, which reads as a detection nobody can review.
    BEGIN IMMEDIATE rather than plain BEGIN so the write lock is taken up front and
    two writers contend at the start rather than half way through.
    """
    violation_id = str(uuid.uuid4())
    con.execute("BEGIN IMMEDIATE")
    try:
        con.execute(
            """
            INSERT INTO traffic_violations
                (id, site_id, source_id, frame_index, calibration_id, configuration_id,
                 type, detected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                violation_id,
                violation.site_id,
                violation.source_id,
                violation.frame_index,
                violation.calibration_id,
                violation.configuration_id,
                violation.type.value,
                violation.detected_at,
            ],
        )
        con.execute(
            """
            INSERT INTO violation_metadata (id, traffic_violation_id, json_blob)
            VALUES (?, ?, ?)
            """,
            [str(uuid.uuid4()), violation_id, violation.metadata.model_dump_json()],
        )
    except Exception:
        con.execute("ROLLBACK")
        raise
    con.execute("COMMIT")
    return violation_id


@dataclass(frozen=True)
class EvidenceTarget:
    """Where in which video one violation happened — everything a cut needs.

    Not the violation itself. Evidence-worker never reads the metadata blob: the boxes
    in it are for whoever draws over the clip afterwards, and the cut is made with
    ffmpeg, which has no use for them.
    """

    # The object key, not a URL. `files.url` holds a key despite its name — see
    # shared.db.init — and a presigned link is minted from it at the moment of reading.
    key: str
    frame_index: int
    # None when the probe could not determine one. A frame index cannot be turned into
    # a seek position without it, so a caller that gets None here has no cut to make;
    # it is reported rather than defaulted, because a guessed frame rate seeks to the
    # wrong second of a real video and looks like a detector that fired at nothing.
    fps: float | None


def evidence_target(
    con: sqlite3.Connection, violation_id: str
) -> EvidenceTarget | None:
    """The footage one violation points at, or None if it points at none.

    None covers two causes deliberately. Either there is no such violation, or there is
    one that predates the source columns and genuinely does not know which video it came
    from — and the caller does the same thing in both cases, because neither can ever be
    cut. Which one it was belongs in the log, not in the control flow, so the caller
    that cares can say "violation %s cannot locate its footage" and be right either way.

    fps comes out of the source's probed metadata rather than a second probe: it was
    measured once when the source was created, and re-reading a container header per
    violation would ask the same question of the same immutable object.
    """
    row = con.execute(
        """
        SELECT files.url, traffic_violations.frame_index, site_sources.metadata
        FROM traffic_violations
        JOIN site_sources ON site_sources.id = traffic_violations.source_id
        JOIN files ON files.id = site_sources.file_id
        WHERE traffic_violations.id = ?
        """,
        [violation_id],
    ).fetchone()
    if row is None:
        return None
    key, frame_index, metadata = row
    if frame_index is None:
        # source_id and frame_index arrived together, so in practice a row with one has
        # the other. Checked anyway: the columns are independently nullable, and a
        # frame index of None would otherwise reach ffmpeg as the string "None".
        return None
    return EvidenceTarget(key=key, frame_index=frame_index, fps=_fps(metadata))


def _fps(metadata: str | None) -> float | None:
    """The frame rate out of a source's metadata document, if it has one.

    Tolerant on purpose, and it is the only tolerant thing in this module. The column
    is free-form TEXT written by the probe, this is one optional number inside it, and
    a document that has drifted should cost the caller a cut it can report rather than
    a worker that stops. Everything else here would rather raise.
    """
    if not metadata:
        return None
    try:
        fps = json.loads(metadata).get("fps")
    except (ValueError, AttributeError):
        return None
    return float(fps) if isinstance(fps, (int, float)) else None


def set_evidence(
    con: sqlite3.Connection,
    violation_id: str,
    status: EvidenceStatus,
    thumbnail_key: str | None = None,
    clip_key: str | None = None,
) -> None:
    """Record how far the cut got, and the keys if it got all the way.

    Both keys are written every time, including the Nones a 'pending' or 'failed' write
    carries. That is what makes a retry clean: a second attempt that fails does not
    leave the first one's half-written key behind, pointing at an object that may have
    been overwritten since.

    `updated_at` is bumped by hand — SQLite's DEFAULT CURRENT_TIMESTAMP applies to the
    INSERT and nothing else, so a row that never bumped it would claim it had not been
    touched since the detector wrote it.
    """
    con.execute(
        """
        UPDATE traffic_violations
        SET evidence_status = ?,
            thumbnail_key = ?,
            clip_key = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        [status.value, thumbnail_key, clip_key, violation_id],
    )
