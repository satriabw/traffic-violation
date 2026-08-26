"""Recording a violation.

The only thing in this process that writes. Reads — a site's calibration and
configuration — resolve by version and are handled elsewhere.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timedelta

from evidence_collector import TrackWindow
from shared.models.detection import DetectionJob, ViolationType
from shared.models.violation import (
    TrackSummary,
    ViolationCreate,
    ViolationMetadata,
)
from violation_detector import Violation


def summary(window: TrackWindow) -> TrackSummary:
    """One track's window, in the shape the metadata blob keeps.

    A rename and nothing else. The four parallel lists are the same four the window
    already holds, in the same order — the evidence package answers in its own
    vocabulary because it depends on nothing, and this is the whole cost of that.

    Nothing is filled in on the way past. A frame nothing projected keeps its None,
    which is why `TrackSummary` accepts one: a job with no calibration writes a
    trajectory of Nones beside a full set of boxes, and that is an honest record of
    what was known rather than a plausible one of what was not.
    """
    return TrackSummary(
        track_id=window.track_id,
        trajectory=list(window.positions),
        speed=list(window.speeds),
        frame_idxs=list(window.frame_indices),
        bboxes=list(window.bboxes),
    )


def detected_at(anchor: datetime, frame_index: int, fps: float | None) -> datetime:
    """When in the footage this happened, as a moment.

    The frame index is the offset; `anchor` turns it into a time. A source with no
    probed frame rate cannot convert one into the other at all, so it reports the
    anchor itself rather than inventing a rate — every violation in such a job lands on
    the same timestamp, which is visibly wrong rather than quietly wrong.
    """
    if not fps or fps <= 0:
        return anchor
    return anchor + timedelta(seconds=frame_index / fps)


def to_create(
    job: DetectionJob,
    violation: Violation,
    window: TrackWindow | None,
    anchor: datetime,
) -> ViolationCreate:
    """A firing rule and its record, as the row that gets written.

    `violation.frame_index`, never the index of the frame being analysed when this was
    produced. A module working on a clip reports several frames late, and recording the
    loop's position would misdate it by the length of the window.

    ONLY THE VIOLATOR IS SUMMARISED. Both rules that ship report against a vehicle, so
    `vehicles` holds its window and `pedestrians` stays empty — even for the pedestrian
    rule, whose whole subject is somebody the module does not name in what it returns.
    Filling that in means either widening `Violation` or having the analyzer keep every
    window rather than the ones that fired, and neither belongs in the change that
    first makes a row appear.

    `frames` stays empty by design: the pixels come back from the source on demand.
    """
    return ViolationCreate(
        site_id=job.site_id,
        source_id=job.source.source_id,
        frame_index=violation.frame_index,
        type=ViolationType(violation.type),
        detected_at=detected_at(anchor, violation.frame_index, job.source.fps),
        metadata=ViolationMetadata(vehicles=[summary(window)] if window else []),
    )


def record(con: sqlite3.Connection, violation: ViolationCreate) -> str:
    """Write the violation and its metadata, and return the new id.

    No evidence frames are uploaded, here or anywhere. The row pins the source and the
    frame index, and the source is an immutable object in storage that can be seeked
    back into, so the pixels are re-derived when somebody opens the detail view — by
    whatever knows how to draw them then, rather than by this process guessing now.
    `ViolationMetadata.frames` stays empty, and the ordering problem the LLD worried
    about (a row pointing at frames nobody uploaded) cannot arise.

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
                (id, site_id, source_id, frame_index, type, detected_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                violation_id,
                violation.site_id,
                violation.source_id,
                violation.frame_index,
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
