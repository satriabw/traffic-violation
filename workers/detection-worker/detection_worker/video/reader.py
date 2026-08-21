"""Turning a video url into the frames a job asked for.

Reading only. Nothing here detects, tracks or annotates — that hangs off the frames
this yields, and keeping it separate is what lets the job-to-pixels hop be tested
without a model anywhere near it.

OpenCV opens an http url through its ffmpeg backend, which range-requests the object
rather than downloading it — the same mechanism shared/video/probe.py relies on. So a
presigned url goes straight into VideoCapture and no file ever lands on disk.
"""

from typing import Callable, Iterator

import cv2
import numpy as np

from shared.models.detection import FrameRange


class VideoUnavailable(Exception):
    """The video could not be opened. Its url may have expired, the object may be
    gone, or the build of OpenCV may have no ffmpeg backend — VideoCapture reports
    all of them the same way, as a capture that simply is not open."""


def read_frames(
    url: str,
    frame_range: FrameRange,
    capture: Callable[[str], cv2.VideoCapture] = cv2.VideoCapture,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield `(index, frame)` for the half-open range, in order.

    The index is absolute — the frame's position in the video, not its position in
    this run. A job covering frames 900-1800 reports 900 for its first frame, because
    that is the number a violation has to be recorded against for anyone to find it
    in the footage later.

    `capture` is a parameter so tests can drive the loop without ffmpeg or a file.
    """
    cap = capture(url)
    if not cap.isOpened():
        # release() on a capture that never opened is harmless, and skipping it here
        # would leak whatever ffmpeg allocated before it gave up.
        cap.release()
        raise VideoUnavailable(f"could not open video at {url}")

    try:
        if frame_range.start > 0:
            # Only when there is somewhere to seek to. Asking to seek to frame 0
            # costs a backend round trip to arrive where the capture already is.
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_range.start)

        for index in range(frame_range.start, frame_range.end):
            ok, frame = cap.read()
            if not ok:
                # Not an error. frame_range.end comes from the container's declared
                # frame count, and containers over-report — a truncated upload or a
                # stream copy leaves nb_frames describing a video that is no longer
                # there. Running out early means the video ended, so stop.
                break
            yield index, frame
    finally:
        # Runs when the consumer stops early too: abandoning a generator closes it,
        # which raises GeneratorExit at the yield and unwinds through here.
        cap.release()
