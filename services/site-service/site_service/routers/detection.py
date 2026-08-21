import sqlite3
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from shared.models.detection import (
    DetectionJob,
    DetectionRequest,
    FrameRange,
    JobSource,
    ViolationType,
)
from shared.models.source import SourceKind, SourceResponse

from site_service import service
from site_service.db import get_db
from site_service.queue import Queue
from site_service.routers.source import require_site
from site_service.storage import Storage

# Nested under a site for the same reason sources are: detection runs against a
# location's footage, and site_id is never taken from the request body.
router = APIRouter(prefix="/sites/{site_id}/detect", tags=["detection"])
DbConnection = Annotated[sqlite3.Connection, Depends(get_db)]


def video_for(
    con: sqlite3.Connection, storage, source: SourceResponse | None
) -> tuple[JobSource, FrameRange]:
    """The video to detect over and the frames to cover, or a 409 explaining why
    there is neither.

    All three rejections are 409 rather than 422: the request is well formed, the site
    is simply not in a state where detection means anything, and the identical request
    succeeds once a video source is attached.

    One function rather than two so "is this site detectable" is decided once, and so
    the job's source and its frame range cannot be derived from different reads of the
    site — which is the whole reason the source travels in the message.
    """
    if source is None:
        raise HTTPException(status_code=409, detail="Site has no source")
    if source.kind is not SourceKind.VIDEO:
        # A live feed has no frame count to bound a job with. Streams are the
        # supervisor-worker path in the LLD, spawned rather than enqueued.
        raise HTTPException(
            status_code=409, detail="Detection runs on a video source, not a stream"
        )
    total_frames = source.metadata.total_frames if source.metadata else None
    if not total_frames:
        # The source exists but was never successfully probed, so nobody knows where
        # the video ends. Guessing a range would hand the worker a job it cannot run.
        raise HTTPException(status_code=409, detail="Video metadata is not available")

    # A video source's file_id is non-null by CHECK constraint, and its upload was
    # verified when the source was created — so this is a lookup, not a check.
    file = service.get_file(con, storage, source.file_id)
    return (
        JobSource(
            source_id=source.id,
            version=source.version,
            # `url` on a file row is the object key, not a URL — the column name is
            # inherited from the LLD. The worker signs its own url from it.
            key=file.url,
            fps=source.metadata.fps,
            total_frames=total_frames,
        ),
        FrameRange(start=0, end=total_frames),
    )


@router.post("", response_model=DetectionJob, status_code=202)
def detect(
    site_id: str,
    con: DbConnection,
    queue: Queue,
    storage: Storage,
    data: DetectionRequest | None = None,
):
    """Queue traffic violation detection over the site's active video.

    202 rather than 201: nothing was created here. The work was accepted, and the id
    that comes back is what identifies it downstream.

    One job for the whole video for now. The LLD's 30-second chunks with 10-second
    overlap are what makes boundary violations detectable, and they arrive with the
    pipeline that needs them.
    """
    require_site(con, site_id)
    source, frame_range = video_for(con, storage, service.get_active_source(con, site_id))
    job = DetectionJob(
        id=str(uuid.uuid4()),
        site_id=site_id,
        # The video itself, not just its id: the worker never looks the source up, so
        # the job means the same thing whenever it is consumed. See JobSource.
        source=source,
        frame_range=frame_range,
        # Unset means every type we know about, so the list can grow without every
        # caller having to be updated.
        types=(data.types if data and data.types else list(ViolationType)),
    )
    # Enqueue before responding: a 202 promising work nobody received is worse than a
    # 500 the caller can retry.
    queue.enqueue(job)
    return job
