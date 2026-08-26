"""One track's history through the window, pivoted out of the frames that hold it.

The buffer records by frame, because that is how frames arrive and because expiry is a
property of a frame, not of a track. Anyone reading the evidence wants the opposite
cut: what did *this* object do. That pivot is all this module is.

NOTHING IS COMPUTED HERE. Every number in a window is a number somebody handed in.
Positions come from whoever projected them, boxes from whoever detected them, and this
module only decides which frames belong to which track. An earlier version of this
derived a pixel position from the box when no projection was available — the middle of
its bottom edge — which restated a formula the projecting package already owns, needed
a test to keep the two copies in step, and stored a number the window's own `bboxes`
already contained. A record that recomputes is a record that can disagree with what it
recorded.

PARALLEL LISTS, AND THEY STAY PARALLEL. Index i of every tuple on a `TrackWindow`
describes the same frame — which is what makes a position matchable to the box it was
measured from, and a speed to the moment it was carried at. It is checked on
construction rather than trusted, because the failure is otherwise silent and arrives
as a box drawn around the wrong second of footage.

A TRACK MISSING FROM A FRAME IS ABSENT, NOT PADDED. Detection flickers; a car behind a
bus for half a second is not a car that was nowhere. Padding the gap with a repeated
position would invent evidence, and padding it with a zero would invent worse. The
window is simply shorter than the buffer, and `frame_indices` says where the holes
were — the same contract the trajectory collector already keeps.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from evidence_collector.objects import FrameEntry, ObjectState


@dataclass(frozen=True)
class TrackWindow:
    """Everything the buffer still holds about one object, oldest first."""

    track_id: int
    # Absolute frame indices, ascending. The only thing that finds these moments in
    # the footage again, and — with the gaps it exposes — the only thing that says how
    # much of the window this track was actually visible for.
    frame_indices: tuple[int, ...] = ()
    # On the ground plane, in metres, exactly as they were handed in — and None on any
    # frame where nothing projected one. None rather than a stand-in derived from the
    # box: without a camera model there is no ground plane and no honest position to
    # report, which is the same reason `trajectory_collector.NullCollector` reports
    # nothing at all. A reader that wants somewhere to draw has `bboxes`.
    positions: tuple[tuple[float, float] | None, ...] = ()
    # Metres per second, None wherever nothing measured one — during a filter's warmup,
    # and throughout a window that was never projected.
    speeds: tuple[float | None, ...] = ()
    # (x1, y1, x2, y2) in pixels, always. A box is a box whether or not anything was
    # projected, and it is what draws the evidence later.
    bboxes: tuple[tuple[float, float, float, float], ...] = ()
    # What the detector called it, per frame. Per frame rather than once, because
    # classification flickers: an object read as a car for forty frames and a truck for
    # two was read both ways, and flattening that here would be this package forming an
    # opinion it has no business having. See `class_name` for the convenient answer.
    class_names: tuple[str, ...] = ()
    # The caller's clock, per frame, or None where it never supplied one.
    timestamps: tuple[float | None, ...] = ()

    def __post_init__(self):
        # The invariant is the type's whole contract, so it is checked rather than
        # documented and hoped for. Every way of building one of these goes through
        # here, `summarize` included.
        lengths = {
            len(self.frame_indices),
            len(self.positions),
            len(self.speeds),
            len(self.bboxes),
            len(self.class_names),
            len(self.timestamps),
        }
        if len(lengths) > 1:
            raise ValueError(
                f"track {self.track_id}: parallel fields must be the same length, got {sorted(lengths)}"
            )

    def __len__(self) -> int:
        """How many frames this track was actually seen on."""
        return len(self.frame_indices)

    @property
    def class_name(self) -> str | None:
        """The name it was called most often, ties going to the most recent.

        A convenience over `class_names`, for a caller that has to pick one — deciding
        whether this is a vehicle, say. Derived, never stored: the per-frame record
        stays the truth, and a caller that disagrees with this rule can implement its
        own over the same data. None for a window holding no frames.
        """
        if not self.class_names:
            return None
        # Reversed, so that among names seen equally often, Counter's first-encountered
        # ordering resolves to the one seen last.
        return Counter(reversed(self.class_names)).most_common(1)[0][0]


def summarize(
    entries: Sequence[FrameEntry],
    track_ids: Iterable[int] | None = None,
) -> tuple[TrackWindow, ...]:
    """Pivot recorded frames into one window per track.

    `track_ids` selects; None takes every track the frames mention. A requested id that
    appears on no frame comes back as nothing at all rather than as an empty window —
    "we have no record of this track" and "this track did nothing" are the same fact
    here, and handing back an empty window invites a caller to write it down as
    evidence.

    Ordered by track id, so two windows produced from one moment come out the same way
    twice. Within a window, frames stay in the order they were recorded.
    """
    wanted = None if track_ids is None else set(track_ids)
    collected: dict[int, list[tuple[FrameEntry, ObjectState]]] = {}

    for entry in entries:
        for state in entry.objects:
            if wanted is not None and state.track_id not in wanted:
                continue
            collected.setdefault(state.track_id, []).append((entry, state))

    return tuple(_window(track_id, collected[track_id]) for track_id in sorted(collected))


def _window(
    track_id: int, sightings: Sequence[tuple[FrameEntry, ObjectState]]
) -> TrackWindow:
    return TrackWindow(
        track_id=track_id,
        frame_indices=tuple(entry.frame_index for entry, _ in sightings),
        positions=tuple(state.position for _, state in sightings),
        speeds=tuple(state.speed for _, state in sightings),
        bboxes=tuple(state.bbox for _, state in sightings),
        class_names=tuple(state.class_name for _, state in sightings),
        timestamps=tuple(entry.timestamp for entry, _ in sightings),
    )
