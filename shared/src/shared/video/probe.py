"""Reading a video's shape without downloading it.

ffprobe speaks HTTP and issues range requests, so against a presigned URL it fetches
the container index and stops — a couple of megabytes regardless of how large the
object is, because the cost tracks frame count rather than bytes. That is what makes
it cheap enough to run inside a request.

Nothing here decodes picture data. The one exception to "no file bytes pass through a
service" is this bounded header read; see shared.s3.client.
"""

import json
import subprocess

from shared import config
from shared.models.source import SourceMetadata


class ProbeError(Exception):
    """Base for the two outcomes a caller has to tell apart."""


class VideoUnreadable(ProbeError):
    """The object is not a video we can read. Retrying will not change that."""


class ProbeUnavailable(ProbeError):
    """We could not reach the object or ffprobe did not finish. The video may well
    be fine; the same call could succeed later."""


# Substrings ffmpeg emits when the *data* is the problem. Anything not matching is
# treated as transient: a wrong "your file is corrupt" sends someone chasing a
# non-existent problem, while a wrong "try again" costs only a retry.
_UNREADABLE_MARKERS = (
    "moov atom not found",
    "invalid data found",
    "end of file",
    "unknown format",
    "no such file or directory",
    "does not contain any stream",
)

# Only the fields SourceMetadata has room for. Asking for less means ffprobe reads
# less, and it keeps the parser below honest about what it may rely on.
_ENTRIES = "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames:format=duration"


def _command(url: str) -> list[str]:
    return [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        # v:0 — a video's first picture stream. Without it an audio track could be
        # measured instead, and an audio-only file would look like a valid video.
        "-select_streams", "v:0",
        "-show_entries", _ENTRIES,
        url,
    ]


def _classify(returncode: int, stderr: str) -> ProbeError:
    lowered = stderr.lower()
    if any(marker in lowered for marker in _UNREADABLE_MARKERS):
        return VideoUnreadable(stderr.strip() or f"ffprobe exited {returncode}")
    return ProbeUnavailable(stderr.strip() or f"ffprobe exited {returncode}")


def _rational(value) -> float | None:
    """ffprobe writes rates as `num/den` strings. Unknown ones come back as `0/0`,
    which is a division by zero rather than a number."""
    if not value or value == "N/A":
        return None
    text = str(value)
    if "/" in text:
        numerator, _, denominator = text.partition("/")
        try:
            num, den = float(numerator), float(denominator)
        except ValueError:
            return None
        return num / den if den else None
    try:
        return float(text)
    except ValueError:
        return None


def _number(value, cast):
    if value is None or value == "N/A":
        return None
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def _metadata_from_probe(payload: dict) -> SourceMetadata:
    streams = payload.get("streams") or []
    if not streams:
        # -select_streams filters rather than fails, so an empty list is how a file
        # with no picture in it arrives here.
        raise VideoUnreadable("no video stream found")
    stream = streams[0]

    width = _number(stream.get("width"), int)
    height = _number(stream.get("height"), int)

    return SourceMetadata(
        total_frames=_number(stream.get("nb_frames"), int),
        fps=_rational(stream.get("avg_frame_rate")),
        nominal_fps=_rational(stream.get("r_frame_rate")),
        duration_seconds=_number((payload.get("format") or {}).get("duration"), float),
        resolution={"width": width, "height": height}
        if width is not None and height is not None
        else None,
    )


def probe(url: str, timeout: float | None = None) -> SourceMetadata:
    """Read a video's metadata from `url`, which may be local or remote.

    Raises VideoUnreadable when the object is not a readable video, and
    ProbeUnavailable when the attempt itself failed.
    """
    timeout = config.VIDEO_PROBE_TIMEOUT_SECONDS if timeout is None else timeout
    try:
        result = subprocess.run(
            _command(url), capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise ProbeUnavailable(f"ffprobe timed out after {timeout}s") from exc
    except FileNotFoundError as exc:
        # ffprobe missing is a deployment fault, not a bad video.
        raise ProbeUnavailable("ffprobe is not installed") from exc

    if result.returncode != 0:
        raise _classify(result.returncode, result.stderr)

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ProbeUnavailable("ffprobe returned output we could not parse") from exc

    return _metadata_from_probe(payload)
