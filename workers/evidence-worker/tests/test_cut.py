import shutil
import subprocess

import pytest

from shared.video.probe import probe

from evidence_worker import cut
from evidence_worker.cut import CutFailed

_HAVE_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")
requires_ffmpeg = pytest.mark.skipif(not _HAVE_FFMPEG, reason="requires ffmpeg and ffprobe")


@pytest.fixture
def video(tmp_path):
    """Ten seconds of test pattern, with a keyframe every second.

    The GOP length is pinned rather than left to the encoder because one test is about
    where a stream copy is allowed to start, and a default GOP of 250 frames would make
    that assertion depend on a version of x264.
    """
    path = tmp_path / "source.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=size=64x64:rate=10:duration=10",
         "-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


@requires_ffmpeg
def test_a_thumbnail_is_one_readable_frame(tmp_path, video):
    out = tmp_path / "thumbnail.jpg"

    cut.thumbnail(str(video), 4.0, str(out))

    assert out.stat().st_size > 0
    # Read back rather than trusted: a jpeg header is what makes this an image and not
    # just some bytes ffmpeg was willing to write.
    assert out.read_bytes()[:2] == b"\xff\xd8"


@requires_ffmpeg
def test_a_clip_holds_the_stretch_it_was_asked_for(tmp_path, video):
    out = tmp_path / "clip.mp4"

    cut.clip(str(video), 3.0, 4.0, str(out))

    # Probed with the same reader the rest of the system uses, so this asserts the file
    # is a video something else can open, not merely that bytes landed.
    assert probe(out.as_uri()).duration_seconds == pytest.approx(4.0, abs=0.6)


@requires_ffmpeg
def test_a_clip_never_starts_after_the_moment_it_was_asked_for(tmp_path, video):
    # The one direction the stream-copy drift is allowed to go. `-c copy` can only begin
    # on a keyframe, so the clip starts at or before the requested second — which is
    # more lead-up, never less, and can therefore never cut off the violation itself.
    out = tmp_path / "clip.mp4"

    cut.clip(str(video), 3.5, 3.0, str(out))

    assert probe(out.as_uri()).duration_seconds >= 3.0


@requires_ffmpeg
def test_seeking_past_the_end_of_the_video_is_a_failure_not_an_empty_file(tmp_path, video):
    # ffmpeg exits 0 here and writes nothing. Without the size check in _run, this
    # violation would upload a zero-byte object and be recorded as ready evidence.
    out = tmp_path / "thumbnail.jpg"

    with pytest.raises(CutFailed, match="past the end"):
        cut.thumbnail(str(video), 60.0, str(out))


@requires_ffmpeg
def test_a_source_that_is_not_there_fails_with_what_ffmpeg_said(tmp_path):
    out = tmp_path / "thumbnail.jpg"

    with pytest.raises(CutFailed) as error:
        cut.thumbnail(str(tmp_path / "absent.mp4"), 1.0, str(out))

    assert "absent.mp4" in str(error.value)


def test_a_cut_that_hangs_is_given_up_on(tmp_path, monkeypatch):
    # There is one worker and it is not threaded, so a stalled read would otherwise park
    # it on this violation for good.
    def hang(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1)

    monkeypatch.setattr(subprocess, "run", hang)

    with pytest.raises(CutFailed, match="did not finish"):
        cut.thumbnail("http://example/v.mp4", 1.0, str(tmp_path / "t.jpg"))


def test_a_missing_ffmpeg_says_so_rather_than_raising_oserror(tmp_path, monkeypatch):
    # The failure a fresh image gives, and one worth naming: FileNotFoundError from
    # subprocess reads as if the *video* were missing.
    def absent(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "ffmpeg")

    monkeypatch.setattr(subprocess, "run", absent)

    with pytest.raises(CutFailed, match="ffmpeg is not on PATH"):
        cut.thumbnail("http://example/v.mp4", 1.0, str(tmp_path / "t.jpg"))


@requires_ffmpeg
def test_the_seek_is_an_input_option(tmp_path, video, monkeypatch):
    # -ss BEFORE -i seeks the container and decodes from there; after -i it decodes the
    # whole video and throws the result away. Nothing about the output distinguishes the
    # two, so the argument order is asserted directly.
    seen = []
    monkeypatch.setattr(cut, "_run", lambda command, path: seen.append(command))

    cut.thumbnail("http://example/v.mp4", 4.0, str(tmp_path / "t.jpg"))

    assert seen[0].index("-ss") < seen[0].index("-i")
