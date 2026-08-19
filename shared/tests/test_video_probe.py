import shutil
import subprocess

import pytest

from shared.video import probe as probe_module
from shared.video.probe import ProbeUnavailable, VideoUnreadable

# ffprobe's real output shape, trimmed to the fields we ask for. Written out rather
# than generated so a change in what we request is visible as a change in the tests.
def _payload(stream=None, duration="900.000000"):
    base = {
        "width": 1280,
        "height": 720,
        "avg_frame_rate": "30/1",
        "r_frame_rate": "30/1",
        "nb_frames": "27000",
    }
    if stream is not None:
        base = {**base, **stream}
        base = {k: v for k, v in base.items() if v is not None}
    return {"streams": [base], "format": {"duration": duration}}


def test_parses_a_constant_rate_video():
    meta = probe_module._metadata_from_probe(_payload())

    assert meta.resolution == {"width": 1280, "height": 720}
    assert meta.fps == 30.0
    assert meta.nominal_fps == 30.0
    assert meta.duration_seconds == 900.0
    assert meta.total_frames == 27000


def test_variable_rate_video_keeps_the_two_rates_apart():
    # The reason both are stored: here a frame index divided by the nominal rate
    # lands at twice the true timestamp.
    meta = probe_module._metadata_from_probe(
        _payload({"avg_frame_rate": "15/1", "r_frame_rate": "30/1"})
    )

    assert meta.fps == 15.0
    assert meta.nominal_fps == 30.0


def test_missing_frame_count_leaves_total_frames_unset():
    # Matroska and transport streams carry no global sample index, so ffprobe has no
    # frame count to report. Everything else is still usable.
    meta = probe_module._metadata_from_probe(_payload({"nb_frames": None}))

    assert meta.total_frames is None
    assert meta.fps == 30.0
    assert meta.resolution == {"width": 1280, "height": 720}


@pytest.mark.parametrize("value", ["0/0", "N/A", ""])
def test_an_unknown_rate_becomes_none_rather_than_an_error(value):
    # ffprobe writes 0/0 when it cannot determine a rate. Dividing it is a crash.
    meta = probe_module._metadata_from_probe(_payload({"avg_frame_rate": value}))

    assert meta.fps is None


def test_a_file_with_no_video_stream_is_unreadable():
    # -select_streams v:0 returns an empty list rather than failing, so this is the
    # only signal that a file typed as video has no picture in it.
    with pytest.raises(VideoUnreadable):
        probe_module._metadata_from_probe({"streams": [], "format": {}})


@pytest.mark.parametrize(
    "label,stderr,expected",
    [
        ("truncated upload", "moov atom not found", VideoUnreadable),
        ("not a video at all", "Invalid data found when processing input", VideoUnreadable),
        ("storage unreachable", "Connection refused", ProbeUnavailable),
    ],
)
def test_failures_are_classified_by_whether_a_retry_could_help(label, stderr, expected):
    # The distinction decides 422 versus 502. Calling a corrupt file transient would
    # have it retried for ever; calling a network blip permanent rejects a good video.
    assert isinstance(probe_module._classify(1, stderr), expected)


_HAVE_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="requires ffmpeg and ffprobe")
def test_probe_reads_a_real_file(tmp_path):
    """The one test that checks ffprobe's actual output against the parser above.

    Without it the parsing tests only confirm our assumptions about a JSON shape we
    wrote ourselves.
    """
    path = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=size=32x32:rate=10:duration=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )

    meta = probe_module.probe(path.as_uri())

    assert meta.resolution == {"width": 32, "height": 32}
    assert meta.fps == 10.0
    assert meta.total_frames == 10
    assert meta.duration_seconds == pytest.approx(1.0, abs=0.1)


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="requires ffmpeg and ffprobe")
def test_probing_something_that_is_not_a_video_raises_unreadable(tmp_path):
    path = tmp_path / "notavideo.mp4"
    path.write_bytes(b"this is not an mp4" * 100)

    with pytest.raises(VideoUnreadable):
        probe_module.probe(path.as_uri())
