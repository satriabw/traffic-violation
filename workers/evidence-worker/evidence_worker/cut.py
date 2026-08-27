"""Cutting a thumbnail and a clip out of a source video.

TWO SUBPROCESS CALLS, AND NO PIXELS IN PYTHON. ffmpeg speaks HTTP and issues range
requests, so against a presigned URL it fetches the bytes around the moment it was
asked for and stops — the same property shared.video.probe relies on, and the reason
this is affordable per violation rather than per job.

NOTHING IS DRAWN ON. No boxes, no region outlines, no highlight on the vehicle that
was convicted. That is not a limitation being worked around: the violation's metadata
blob already holds every box and every frame index, so whoever renders the detail view
draws them over the clip at read time — which means fixing the drawing improves every
violation ever recorded, and the expensive half is still baked exactly once. Annotating
here would freeze one rendering into an object and cost a decode of every frame to do
it.
"""

import os
import subprocess

from shared import config


class CutFailed(RuntimeError):
    """ffmpeg did not produce the file we asked for.

    One exception rather than the transient/permanent pair shared.video.probe draws,
    because nothing here retries: the violation is marked failed and the worker moves
    on either way. The message carries ffmpeg's stderr, which is where the difference
    actually shows.
    """


def _run(command: list[str], path: str) -> None:
    """Run one ffmpeg invocation and insist it left a file behind.

    THE EXIT CODE IS NOT ENOUGH. Asked to seek past the end of a video, ffmpeg exits 0
    and writes a zero-byte file — so a violation whose frame index does not match the
    footage would otherwise upload an empty object and record it as ready evidence.
    """
    try:
        done = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=config.EVIDENCE_CUT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        # There is one worker and it is not threaded, so a stalled read would park it
        # on this violation for good.
        raise CutFailed(
            f"ffmpeg did not finish within {config.EVIDENCE_CUT_TIMEOUT_SECONDS}s"
        ) from None
    except FileNotFoundError:
        raise CutFailed("ffmpeg is not on PATH") from None

    if done.returncode != 0:
        raise CutFailed(done.stderr.strip() or f"ffmpeg exited {done.returncode}")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise CutFailed(
            f"ffmpeg exited 0 but wrote nothing to {os.path.basename(path)} — "
            "the seek position is probably past the end of the video"
        )


def thumbnail(url: str, seconds: float, path: str) -> None:
    """One frame, at `seconds` into the video.

    `-ss` BEFORE `-i` is the whole performance story: as an input option it seeks the
    container and starts decoding there, so the cost is the moment asked for rather
    than everything leading up to it. After `-i` it would decode the video from the
    beginning and throw the result away.
    """
    _run(
        [
            "ffmpeg",
            "-v", "error",
            "-ss", f"{seconds:.3f}",
            "-i", url,
            "-frames:v", "1",
            # 2-5 is the useful range; 3 is a list thumbnail nobody squints at.
            "-q:v", "3",
            "-y", path,
        ],
        path,
    )


def clip(url: str, start: float, duration: float, path: str) -> None:
    """`duration` seconds of the source from `start`, remuxed rather than re-encoded.

    `-c copy` IS WHY THIS IS CHEAP. The frames are copied out of one container into
    another with no decode and no encode, so a clip costs roughly its own bytes off
    the network and nothing on the CPU.

    What it costs instead is precision: a stream copy can only begin on a keyframe, so
    the clip starts at or before the requested moment by up to one GOP — a second or
    two of ordinary footage. That is lead-up either way, and it is the wrong direction
    to be wrong in only if it ever ate the violation itself, which it cannot, because
    the error is always earlier. Re-encoding with `-c:v libx264` is the one-line fix if
    the drift ever matters more than the CPU.
    """
    _run(
        [
            "ffmpeg",
            "-v", "error",
            "-ss", f"{start:.3f}",
            "-i", url,
            "-t", f"{duration:.3f}",
            "-c", "copy",
            # Moves the index to the front of the file so a reviewer's browser can
            # start playing on the first bytes instead of waiting for the whole object.
            "-movflags", "+faststart",
            "-y", path,
        ],
        path,
    )
