"""Queue consumer entrypoint.

It takes one violation off the queue, works out where in which video it happened, cuts
a thumbnail and a clip out of that video with ffmpeg, puts both in storage and writes
the keys back onto the row.

WHY THIS IS NOT INSIDE DETECTION-WORKER. That process holds the GPU, and it is the one
resource here that cannot be scaled sideways; cutting a clip is ffmpeg and a network
round trip, which is neither. Running it there — on a thread, or inline — would also
put the work somewhere with no durability: detection-worker deliberately has no restart
policy, so a handler that raises stops it, and a background thread would die with it
silently, leaving a row that says nothing and nothing that knows to try again. On the
queue, the job simply outlives the process.

WHY NOT AT READ TIME, which is where this used to be going. A page of violations is a
seek and a decode each, so list latency would scale with page size — and a clip is not
work an HTTP handler can do at all, at any page size.

A FAILED CUT DOES NOT STOP THE WORKER, which is where this diverges from
detection-worker's contract. There a raising handler stops the process, because losing
a job silently is worse than stopping loudly. Here the failure has somewhere to go: the
violation is marked failed on its own row, where a reader can see it and act. What
still stops the worker is everything that is not about one violation — no database, no
credentials, a bucket that refuses a write — because those are true of the next job too.
"""

import logging
import os
import tempfile
from typing import Callable

from shared.db.violations import EvidenceTarget, evidence_target, set_evidence
from shared.models.evidence import EvidenceJob
from shared.models.violation import EvidenceStatus
from shared.queue.client import evidence_from_config
from shared.s3.client import presigned_get, upload
from shared.s3.keys import build_key

from evidence_worker import cut
from evidence_worker.db import get_db

logger = logging.getLogger(__name__)

# What the two objects are called inside a violation's prefix. Fixed names, so a retry
# overwrites its own previous attempt instead of accumulating one object per try.
THUMBNAIL_NAME = "thumbnail.jpg"
CLIP_NAME = "clip.mp4"


def window(target: EvidenceTarget, evidence_seconds: float) -> tuple[float, float, float]:
    """(the moment, where the clip starts, how long it runs) — all in seconds.

    THE SAME SPAN THE RECORD COVERS, and that is the whole of why `evidence_seconds`
    rides on the job rather than being a setting here. The detector sized its ring
    buffer with this number, so the blob holds boxes for exactly these frames — a clip
    cut to any other length is either missing footage the record describes or showing
    footage it does not, and a reviewer drawing those boxes over it would watch them
    start late or run out early.

    IT ENDS ON THE VIOLATION, like the buffer, and for the buffer's own reason: what a
    reviewer needs is the approach, and what came after is still in the source for
    anyone who wants it. The `+ 1/fps` is FrameBuffer.over's `+ 1` in seconds — the
    frame a rule fires on is part of its window, so the clip has to reach the far side
    of that frame rather than stopping at its leading edge.

    Clamped at zero, which matters more than it looks: a violation in the first few
    seconds of a video would otherwise ask ffmpeg to seek to a negative offset. The
    duration absorbs the clamp, so such a clip is short rather than wrong — which is the
    same truncation the record itself carries, and the same one `short=` counts.

    fps is the source's own, measured once at upload. `target.fps` is None-checked by
    the caller, not here, because "we cannot place this violation in time" is a verdict
    about the violation and belongs where the verdict gets written.
    """
    at = target.frame_index / target.fps
    start = max(0.0, at - evidence_seconds)
    return at, start, (at - start) + (1.0 / target.fps)


def make_handler(
    con,
    sign: Callable[[str], str] = presigned_get,
    put: Callable[..., str] = upload,
    cut_thumbnail: Callable[[str, float, str], None] = cut.thumbnail,
    cut_clip: Callable[[str, float, float, str], None] = cut.clip,
) -> Callable[[EvidenceJob], None]:
    """Build the job handler, with its collaborators injectable.

    A factory for the same reason detection-worker's is: `run` keeps taking a plain
    `Callable[[EvidenceJob], None]`, and tests substitute a fake signer, uploader and
    pair of cutters so the whole path runs with no S3, no ffmpeg and no video.
    """

    def fail(violation_id: str, reason: str) -> None:
        # On the row rather than only in the log. A reader has to be able to tell a
        # violation whose clip will never arrive from one still waiting for it, and the
        # log is not somewhere the detail view can look.
        logger.warning("evidence for violation %s failed: %s", violation_id, reason)
        set_evidence(con, violation_id, EvidenceStatus.FAILED)

    def handle(job: EvidenceJob) -> None:
        target = evidence_target(con, job.violation_id)
        if target is None:
            # Either there is no such violation or it predates the source columns and
            # cannot say which video it came from. Neither can ever be cut — and in the
            # first case the write below matches no row, which is the right amount of
            # nothing to do.
            return fail(job.violation_id, "it cannot locate its own footage")
        if target.fps is None:
            # Rather than assuming 25 or 30. The frame index is only a position in time
            # alongside a frame rate, and a guessed one seeks to the wrong second of a
            # real video — evidence of something that did not happen, which is worse
            # than no evidence at all.
            return fail(job.violation_id, f"source {target.key} has no known frame rate")

        at, start, duration = window(target, job.evidence_seconds)
        # Signed here rather than at enqueue time, for the reason JobSource gives: a
        # presigned url that expired in a backlog fails after the work has started.
        url = sign(target.key)

        try:
            # One directory per job, removed on the way out whatever happens. The two
            # objects are the only artifacts; nothing is kept locally once they land.
            with tempfile.TemporaryDirectory() as workspace:
                thumbnail_path = os.path.join(workspace, THUMBNAIL_NAME)
                clip_path = os.path.join(workspace, CLIP_NAME)

                cut_thumbnail(url, at, thumbnail_path)
                cut_clip(url, start, duration, clip_path)

                # Uploaded inside the try but NOT inside the except below: a bucket
                # that refuses a write is not a fact about this violation, and marking
                # it failed would quietly do the same to every job behind it.
                thumbnail_key = put(
                    build_key("evidence", job.violation_id, THUMBNAIL_NAME),
                    thumbnail_path,
                    "image/jpeg",
                )
                clip_key = put(
                    build_key("evidence", job.violation_id, CLIP_NAME),
                    clip_path,
                    "video/mp4",
                )
        except cut.CutFailed as error:
            return fail(job.violation_id, str(error))

        # Last, and only once both objects are in storage. The row never names a key
        # nobody has uploaded — the ordering problem the LLD worried about, avoided by
        # doing the work first rather than by a transaction.
        set_evidence(
            con,
            job.violation_id,
            EvidenceStatus.READY,
            thumbnail_key=thumbnail_key,
            clip_key=clip_key,
        )
        logger.info(
            "evidence for violation %s ready: frame %d at %.3fs, clip %.3f-%.3fs of %s",
            job.violation_id,
            target.frame_index,
            at,
            start,
            start + duration,
            target.key,
        )

    return handle


def run(
    queue,
    handle: Callable[[EvidenceJob], None],
    max_jobs: int | None = None,
) -> int:
    """Consume until the queue is exhausted, or max_jobs have been handled.

    The same loop detection-worker runs, deliberately copied rather than shared. It is
    eight lines, and the two workers' stopping conditions are not the same promise —
    see the module docstring — so a common `run` would need a flag to say which one it
    was keeping, which is more to read than the loop.

    Whether "exhausted" ever happens is the queue's decision: an InMemoryQueue returns
    None once drained, while RedisQueue.consume blocks on BRPOP and so keeps a deployed
    worker running indefinitely.

    max_jobs exists so tests can bound the loop without signals; nothing else uses it.
    """
    handled = 0
    while max_jobs is None or handled < max_jobs:
        job = queue.consume()
        if job is None:
            break
        handle(job)
        handled += 1
    return handled


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Before the queue is touched, so a database that is missing or has no schema stops
    # the worker while it still has no claim on any job.
    con = get_db()
    logger.info("evidence-worker waiting for jobs")
    run(evidence_from_config(), make_handler(con))


if __name__ == "__main__":
    main()
