import shutil
import subprocess

import cv2
import numpy as np
import pytest
from shared.models.detection import FrameRange

from detection_worker.reader import VideoUnavailable, read_frames


class FakeCapture:
    """A VideoCapture with the four methods read_frames actually uses.

    Standing in for the real one keeps every test below about the loop — the range it
    covers, when it seeks, when it stops — rather than about ffmpeg. The one test that
    checks those assumptions against real OpenCV is at the bottom of this file.
    """

    def __init__(self, frame_count: int = 10, opened: bool = True):
        self._frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(frame_count)]
        self._opened = opened
        self.position = 0
        self.seeks: list[int] = []
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 — OpenCV's name
        return self._opened

    def set(self, prop: int, value: float) -> bool:  # noqa: A003
        assert prop == cv2.CAP_PROP_POS_FRAMES
        self.seeks.append(int(value))
        self.position = int(value)
        return True

    def read(self):
        if self.position >= len(self._frames):
            # What OpenCV returns past the end of a video.
            return False, None
        frame = self._frames[self.position]
        self.position += 1
        return True, frame

    def release(self) -> None:
        self.released = True


def _capture_of(capture: FakeCapture):
    return lambda url: capture


def test_yields_exactly_the_requested_half_open_range():
    frames = list(read_frames("u", FrameRange(start=2, end=5), _capture_of(FakeCapture())))

    assert [index for index, _ in frames] == [2, 3, 4]


def test_indices_are_absolute_positions_in_the_video():
    # Not 0-based within the run: a violation is recorded against the frame number
    # someone can scrub to in the original footage.
    capture = FakeCapture()

    frames = list(read_frames("u", FrameRange(start=6, end=8), _capture_of(capture)))

    assert [index for index, _ in frames] == [6, 7]
    # The frames really are the ones at those positions, not the first two decoded.
    assert [int(frame[0][0][0]) for _, frame in frames] == [6, 7]


def test_a_range_that_does_not_start_at_zero_seeks_there_first():
    capture = FakeCapture()

    list(read_frames("u", FrameRange(start=4, end=6), _capture_of(capture)))

    assert capture.seeks == [4]


def test_a_range_starting_at_zero_does_not_seek():
    # A seek to where the capture already is costs a backend round trip for nothing.
    capture = FakeCapture()

    list(read_frames("u", FrameRange(start=0, end=3), _capture_of(capture)))

    assert capture.seeks == []


def test_a_video_that_ends_early_stops_without_raising():
    # nb_frames is the container's claim, and a truncated or stream-copied file
    # over-reports it. Running out is the video ending, not a failure.
    capture = FakeCapture(frame_count=4)

    frames = list(read_frames("u", FrameRange(start=0, end=100), _capture_of(capture)))

    assert [index for index, _ in frames] == [0, 1, 2, 3]


def test_the_capture_is_released_when_the_range_is_read_to_the_end():
    capture = FakeCapture()

    list(read_frames("u", FrameRange(start=0, end=3), _capture_of(capture)))

    assert capture.released


def test_the_capture_is_released_when_the_consumer_stops_early():
    # The generator is abandoned after one frame. Without the finally, the capture —
    # and the http connection under it — would leak until garbage collection.
    capture = FakeCapture()

    frames = read_frames("u", FrameRange(start=0, end=10), _capture_of(capture))
    next(frames)
    frames.close()

    assert capture.released


def test_a_capture_that_will_not_open_raises():
    capture = FakeCapture(opened=False)

    with pytest.raises(VideoUnavailable):
        # list() because the body of a generator does not run until it is iterated —
        # calling read_frames alone would raise nothing.
        list(read_frames("u", FrameRange(start=0, end=3), _capture_of(capture)))


def test_a_capture_that_will_not_open_is_still_released():
    capture = FakeCapture(opened=False)

    with pytest.raises(VideoUnavailable):
        list(read_frames("u", FrameRange(start=0, end=3), _capture_of(capture)))

    assert capture.released


_HAVE_FFMPEG = shutil.which("ffmpeg") is not None


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="requires ffmpeg to build a test clip")
def test_reads_a_real_video(tmp_path):
    """The one test that runs the loop against real OpenCV.

    Without it the tests above only confirm our assumptions about an API we wrote the
    fake for — in particular that a seek followed by reads lands on the frame we
    asked for.
    """
    path = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=size=32x32:rate=10:duration=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )

    whole = list(read_frames(str(path), FrameRange(start=0, end=10)))
    middle = list(read_frames(str(path), FrameRange(start=3, end=7)))

    assert [index for index, _ in middle] == [3, 4, 5, 6]
    assert all(frame.shape == (32, 32, 3) for _, frame in middle)
    # The pixels prove the seek landed. Comparing against the same frames read
    # sequentially is the only thing that would catch a seek that silently did
    # nothing, or one that stopped at the preceding keyframe.
    for (_, seeked), (_, sequential) in zip(middle, whole[3:7]):
        assert np.array_equal(seeked, sequential)


def test_an_unopenable_url_raises_video_unavailable(tmp_path):
    # Real OpenCV this time: a path with no video behind it is exactly what an
    # expired presigned url looks like from here.
    missing = tmp_path / "nothing.mp4"

    with pytest.raises(VideoUnavailable):
        list(read_frames(str(missing), FrameRange(start=0, end=1)))
