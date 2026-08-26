"""Resolving a job's per-site context.

The job carries version numbers; the documents themselves are small JSON files in
object storage, reached through the database. Two hops, both of which have to be
pinned to the version in the message:

    camera_calibrations WHERE site_id = ? AND version = ?  ->  file_id
    files               WHERE id = ?                       ->  url (the object key)
    S3                                                     ->  the document

Never "the site's active calibration". A job enqueued against v3 and consumed after
someone uploaded v4 must still be evaluated against v3 — otherwise the run silently
uses a different camera model than the one it was created for, and the output stays
plausible enough that nobody notices.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from shared.models.detection import DetectionJob
from shared.s3.client import get_bytes, get_json
from violation_detector import ConfigurationInvalid

# Table names, as module constants. Interpolated into SQL below and never derived from
# a message — the same rule site_service.service follows for the pair.
CALIBRATIONS = "camera_calibrations"
CONFIGURATIONS = "configurations"

# Where a site says how much lead-up its violations should carry. A top-level key in
# the configuration document, alongside `violations` and `regions` — the parser in
# violation-detector leaves unknown top-level keys alone precisely so a document can
# carry something like this, so nothing over there changes.
EVIDENCE_SECONDS_KEY = "evidence_seconds"

# What a site that says nothing gets. Five seconds is what the pipeline this is ported
# from used, and its output bears the choice out: a car that ran a crossing was outside
# the region of interest for every one of the ninety-five frames recorded before the one
# it was convicted on, which is to say the whole window is approach.
#
# A junction is the thing that knows better. An approach with a long sight line wants
# more; a tight one-way needs less. That is why this is only the fallback.
DEFAULT_EVIDENCE_SECONDS = 5.0


class ContextMissing(RuntimeError):
    """A job named a version that is not in the database.

    Not the same thing as a job naming no version at all, which is normal. This means
    the message and the database disagree, and guessing which one is right — by
    falling back to the active version, say — would reintroduce exactly the drift the
    version pin exists to prevent.
    """


@dataclass(frozen=True)
class JobContext:
    """What a rule engine will need. Both None until a site has them.

    The two are fetched differently, and the asymmetry is deliberate. A configuration
    is ours — we define its shape, it is JSON, and parsing it here is the last anyone
    has to think about it. A calibration is not: it is whatever the tool that produced
    the camera model wrote, which in practice is an OpenCV FileStorage `.yml`. So it
    travels as the raw document and `trajectory_collector` decides what it means, which
    is the only place that knows what a camera model is.
    """

    calibration: bytes | None = None
    configuration: dict | None = None
    # How many seconds of lead-up this site's violations should carry. Read out of the
    # configuration document here, where the document is already in hand, and passed on
    # as a number — nothing downstream of this is handed the document to go digging in.
    evidence_seconds: float = DEFAULT_EVIDENCE_SECONDS
    # When the source this job reads was added to the site. The anchor a violation's
    # `detected_at` is measured from — the frame index gives the offset into the
    # footage, and this is what turns that into a moment.
    #
    # IT IS THE UPLOAD, NOT THE RECORDING. Nothing in the system knows when footage was
    # shot: the probe does not read the container's creation_time, and neither
    # SourceMetadata nor JobSource carries one. So every `detected_at` is late by
    # however long the video sat between being filmed and being uploaded — a constant
    # per source, which leaves violations correctly ordered within a video and
    # correctly spaced, and only wrong in absolute terms. When a real recording time
    # lands this is the one line that changes.
    source_created_at: datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _document_key(
    con: sqlite3.Connection, table: str, site_id: str, version: int
) -> str:
    row = con.execute(
        f"""
        SELECT files.url
        FROM {table}
        JOIN files ON files.id = {table}.file_id
        WHERE {table}.site_id = ? AND {table}.version = ?
        """,
        [site_id, version],
    ).fetchone()
    if row is None:
        raise ContextMissing(f"{table} v{version} for site {site_id} does not exist")
    # `url` on a files row is the object key, not a URL — the column name is inherited
    # from the LLD, the same way JobSource.key is.
    return row[0]


def _evidence_seconds(document: dict | None) -> float:
    """How much lead-up this site asks for, or the default if it does not say.

    Checked while the context resolves, so a document with a nonsense value stops the
    job before a frame is decoded — the same bargain the rest of this module strikes.
    A value that is not a number, or not positive, would otherwise surface as a
    TypeError deep inside a ring buffer on the first frame, or as a site whose records
    were silently empty and looked exactly like a junction where nothing ever happened.
    """
    if not document or EVIDENCE_SECONDS_KEY not in document:
        return DEFAULT_EVIDENCE_SECONDS
    declared = document[EVIDENCE_SECONDS_KEY]
    try:
        seconds = float(declared)
    except (TypeError, ValueError):
        raise ConfigurationInvalid(
            f"{EVIDENCE_SECONDS_KEY} must be a number, got {declared!r}"
        ) from None
    if seconds <= 0:
        raise ConfigurationInvalid(
            f"{EVIDENCE_SECONDS_KEY} must be positive, got {declared!r}"
        )
    return seconds
def _source_created_at(con: sqlite3.Connection, source_id: str) -> datetime:
    row = con.execute(
        "SELECT created_at FROM site_sources WHERE id = ?", [source_id]
    ).fetchone()
    if row is None:
        # The same refusal as a missing calibration, for the same reason: the message
        # and the database disagree, and a job that cannot say when its violations
        # happened should stop rather than guess.
        raise ContextMissing(f"source {source_id} does not exist")
    created_at = row[0]
    # SQLite's CURRENT_TIMESTAMP is UTC and carries no offset, so it comes back naive.
    # Stamped rather than left that way: detected_at is written aware, and one column
    # holding both kinds is a comparison that raises the first time anyone sorts it.
    if created_at.tzinfo is None:
        return created_at.replace(tzinfo=timezone.utc)
    return created_at


def load(
    con: sqlite3.Connection,
    job: DetectionJob,
    fetch_json: Callable[[str], dict] = get_json,
    fetch_bytes: Callable[[str], bytes] = get_bytes,
) -> JobContext:
    """The calibration and configuration this job was created against.

    Both fetchers are injectable so tests resolve real keys out of a real database
    without object storage anywhere in reach.
    """
    configuration = (
        None
        if job.configuration_version is None
        else fetch_json(
            _document_key(con, CONFIGURATIONS, job.site_id, job.configuration_version)
        )
    )
    return JobContext(
        configuration=configuration,
        evidence_seconds=_evidence_seconds(configuration),
        source_created_at=_source_created_at(con, job.source.source_id),
        calibration=(
            None
            if job.calibration_version is None
            else fetch_bytes(
                _document_key(con, CALIBRATIONS, job.site_id, job.calibration_version)
            )
        ),
    )
