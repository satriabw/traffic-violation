"""The recent history of tracked objects, kept until something makes it worth recording.

    from evidence_collector import EvidenceCollector, ObjectState

    collector = EvidenceCollector.over(seconds=5, fps=30)

    for frame_index, tracked in analysed_frames:
        collector.observe(frame_index, tracked)
        for track_id in whatever_just_happened:
            windows = collector.window_for([track_id])   # the lead-up, oldest first

Records, not pixels. What is kept is where each object was and how fast it was going,
against the frame index that can find it in the footage again — so evidence is
re-derived from the source when somebody asks, rather than guessed at and stored by
whatever was running at the time.

Frames go in, tracks come out. The buffer records by frame because that is how frames
arrive and because expiry is a property of a frame; anyone reading the evidence wants
to know what one object did, and `window_for` is that pivot.

The re-exports are deliberate, and the same departure `trajectory_collector` and
`violation_detector` make: an application's `__init__.py` stays empty because its
modules are imported by path, but a library's import path is its API, and moving a
module should not break anyone.
"""

from evidence_collector.buffer import FrameBuffer
from evidence_collector.collector import EvidenceCollector
from evidence_collector.objects import FrameEntry, ObjectState
from evidence_collector.window import TrackWindow, summarize

__all__ = [
    "EvidenceCollector",
    "FrameBuffer",
    "FrameEntry",
    "ObjectState",
    "TrackWindow",
    "summarize",
]
