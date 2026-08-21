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
from typing import Callable

from shared.models.detection import DetectionJob
from shared.s3.client import get_bytes, get_json

# Table names, as module constants. Interpolated into SQL below and never derived from
# a message — the same rule site_service.service follows for the pair.
CALIBRATIONS = "camera_calibrations"
CONFIGURATIONS = "configurations"


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
    return JobContext(
        calibration=(
            None
            if job.calibration_version is None
            else fetch_bytes(
                _document_key(con, CALIBRATIONS, job.site_id, job.calibration_version)
            )
        ),
        configuration=(
            None
            if job.configuration_version is None
            else fetch_json(
                _document_key(con, CONFIGURATIONS, job.site_id, job.configuration_version)
            )
        ),
    )
