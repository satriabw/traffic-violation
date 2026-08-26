"""The recent history of tracked objects, kept until something makes it worth recording.

    from evidence_collector import FrameBuffer, FrameEntry, ObjectState

    buffer = FrameBuffer.over(seconds=5, fps=30)

    for frame_index, tracked in analysed_frames:
        buffer.add(FrameEntry(frame_index=frame_index, objects=tracked))
        if something_happened:
            window = buffer.entries()   # the lead-up, oldest first

Records, not pixels. What is kept is where each object was and how fast it was going,
against the frame index that can find it in the footage again — so evidence is
re-derived from the source when somebody asks, rather than guessed at and stored by
whatever was running at the time.

The re-exports are deliberate, and the same departure `trajectory_collector` and
`violation_detector` make: an application's `__init__.py` stays empty because its
modules are imported by path, but a library's import path is its API, and moving a
module should not break anyone.
"""

from evidence_collector.buffer import FrameBuffer
from evidence_collector.objects import FrameEntry, ObjectState

__all__ = [
    "FrameBuffer",
    "FrameEntry",
    "ObjectState",
]
