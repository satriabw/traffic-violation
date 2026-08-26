"""The ring: the last few seconds of frames, and nothing before or after them.

WHY IT ENDS AT THE PRESENT, not a fixed distance either side of something. By the time
a rule fires the violation has already happened, and what a reviewer needs is what led
up to it — the approach, the speed it was carrying, whether anyone was already in the
crossing. What came after is the consequence, and it is still in the footage for
anyone who wants it. Buffering forward would mean holding every frame until enough
future arrived to prove nothing happened, on the overwhelming majority of frames where
nothing ever does.

WHY IT ENDS SOMEWHERE. A full run's history is unbounded and mostly about objects that
never did anything. A ring gives back the frames whose relevance has expired without
anyone having to work out which those are, and it costs the same on a quiet junction
as a busy one.

WHAT IS IN IT IS NOT PIXELS. Boxes, ids, positions and speeds — some hundreds of bytes
a frame against some megabytes for the image. Keeping the frame index instead means
the pixels can always be recovered from the source, and recovered *better* later, by
whatever knows how to draw them at the time somebody asks.
"""

from collections import deque
from typing import Iterable

from evidence_collector.objects import FrameEntry


class FrameBuffer:
    """A fixed number of the most recent frames, oldest first.

    Per tracking session, like everything keyed by tracker id: ids restart at 1 for
    every video, so a buffer outliving one would hand back a window mixing two
    different objects that happened to be numbered the same.
    """

    def __init__(self, frames: int):
        if frames < 1:
            # A buffer of nothing is not a degraded buffer, it is a caller who
            # computed a size wrong — and it would report an empty window for every
            # violation, which reads exactly like a junction where nothing led up to
            # anything.
            raise ValueError(f"frames must be at least 1, got {frames}")
        self._entries: deque[FrameEntry] = deque(maxlen=frames)
        # The last index recorded, so a caller that rewinds or repeats is told rather
        # than silently given a scrambled window. See `add`.
        self._last_index: int | None = None

    @classmethod
    def over(cls, seconds: float, fps: float) -> "FrameBuffer":
        """A buffer holding `seconds` of history, at this video's frame rate.

        The named constructor is the one to reach for: a window is a duration in
        anybody's head, and frame counts are what a duration becomes once you know the
        rate. Doing that arithmetic here means one place gets the `+ 1` right.

        THE `+ 1` IS LOAD-BEARING. The frame a rule fires on is recorded before the
        window is read — a rule reports on a frame the caller has just finished
        analysing — so the buffer holds `seconds` of lead-up *plus* the moment itself.
        Sizing it to `seconds * fps` exactly would push the oldest frame of the window
        out to make room for the present one, and every window would be a frame short
        of what was asked for.
        """
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if seconds <= 0:
            raise ValueError(f"seconds must be positive, got {seconds}")
        return cls(frames=round(seconds * fps) + 1)

    @property
    def capacity(self) -> int:
        """How many frames it holds once full."""
        return self._entries.maxlen

    def add(self, entry: FrameEntry) -> None:
        """Record one frame, evicting the oldest once full.

        Indices must strictly increase. A repeated or rewound one means the caller fed
        a frame twice or restarted mid-video, and the damage is silent otherwise: the
        window would still be the right length and still look like a plausible piece
        of history, while describing a stretch of time that never happened. This is
        the one thing in here that refuses, and it refuses at the point of the mistake
        rather than at the point of reading a record nobody can check.
        """
        if self._last_index is not None and entry.frame_index <= self._last_index:
            raise ValueError(
                f"frame indices must increase: {entry.frame_index} follows {self._last_index}"
            )
        self._last_index = entry.frame_index
        self._entries.append(entry)

    def extend(self, entries: Iterable[FrameEntry]) -> None:
        """Record several, in order. Convenience; `add`'s rules apply to each."""
        for entry in entries:
            self.add(entry)

    def entries(self) -> tuple[FrameEntry, ...]:
        """Everything held, oldest first.

        A tuple, not a view: what comes back is a reading taken now, and a caller that
        held one while the buffer kept filling would find its evidence had moved on.
        Oldest first because that is the order the events happened in, and a window
        read backwards is a window someone will eventually read wrong.
        """
        return tuple(self._entries)

    def clear(self) -> None:
        """Forget everything, including how far the indices had got.

        Nothing in a normal run calls this — a buffer belongs to one tracking session
        and is discarded with it, which is cheaper and harder to get wrong than
        reusing one. It is here for a caller that genuinely restarts.
        """
        self._entries.clear()
        self._last_index = None

    def __len__(self) -> int:
        """How many frames are held, which is below `capacity` until it fills.

        Defined, so `if buffer:` means what it looks like. The pipeline this is ported
        from left it out and then guarded three call sites with `if not
        self._frame_buffer` — a plain object with no `__len__` and no `__bool__` is
        always truthy, so every one of those branches was unreachable and the empty
        case they were written for had never once been handled.
        """
        return len(self._entries)
