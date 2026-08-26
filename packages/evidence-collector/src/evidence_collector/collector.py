"""The one class a caller names: record every frame, read a window when it matters.

The buffer and the pivot are both usable on their own and both stay public, but wiring
them together is the same two lines in every caller — build an entry, add it — and
there is no reason for each of them to write those out.
"""

from typing import Iterable

from evidence_collector.buffer import FrameBuffer
from evidence_collector.objects import FrameEntry, ObjectState
from evidence_collector.window import TrackWindow, summarize

__all__ = ["EvidenceCollector"]


class EvidenceCollector:
    """Keeps the recent past, and hands back what one moment needs of it.

    Per tracking session, like everything else keyed by tracker id. Ids restart at 1
    for every video, so a collector outliving one would answer a question about a car
    with the history of a pedestrian that happened to be numbered the same.
    """

    def __init__(self, buffer: FrameBuffer):
        self._buffer = buffer

    @classmethod
    def over(cls, seconds: float, fps: float) -> "EvidenceCollector":
        """A collector keeping `seconds` of history at this video's frame rate."""
        return cls(FrameBuffer.over(seconds=seconds, fps=fps))

    @property
    def capacity(self) -> int:
        return self._buffer.capacity

    def observe(
        self,
        frame_index: int,
        objects: Iterable[ObjectState],
        timestamp: float | None = None,
    ) -> None:
        """Record one frame.

        EVERY FRAME, including the ones with nothing on them and the ones where nothing
        happened. A window is a duration, and a caller that only recorded the
        interesting frames would hand back five entries spanning four minutes while
        claiming to be five seconds of lead-up. Indices must increase; the buffer says
        so and says why.
        """
        self._buffer.add(
            FrameEntry(frame_index=frame_index, objects=tuple(objects), timestamp=timestamp)
        )

    def window_for(self, track_ids: Iterable[int] | None = None) -> tuple[TrackWindow, ...]:
        """What the record still holds about these tracks, oldest first.

        None takes every track in the buffer — which is what a caller wants when the
        question is "who else was there", and it is bounded by the window rather than
        by the run.

        A reading taken now. Nothing here mutates the buffer, so the same moment can be
        read twice, and two rules firing on one frame each get their own answer.
        """
        return summarize(self._buffer.entries(), track_ids)

    def clear(self) -> None:
        """Forget everything. See `FrameBuffer.clear` — nothing normal calls this."""
        self._buffer.clear()

    def __len__(self) -> int:
        """How many frames are held."""
        return len(self._buffer)
