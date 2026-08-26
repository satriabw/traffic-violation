# evidence-collector

The last few seconds of what every tracked object was doing, kept until something
makes it worth recording.

```python
from evidence_collector import EvidenceCollector, ObjectState

collector = EvidenceCollector.over(seconds=5, fps=30)

for frame_index, tracked in analysed_frames:
    collector.observe(frame_index, [
        ObjectState(track_id=7, bbox=(x1, y1, x2, y2), class_name="car"),
    ])

    for track_id in whatever_just_happened:
        windows = collector.window_for([track_id])   # the lead-up, oldest first
```

`observe` takes **every** frame, including the empty ones. A window is a duration, and
a caller that recorded only the interesting frames would hand back five entries
spanning four minutes while claiming to be five seconds of lead-up.

## Records, not pixels

What is kept is where each object was and how fast it was going, against the frame
index that finds the moment in the footage again. Some hundreds of bytes a frame,
against some megabytes for the image.

That is the whole design. The pixels can always be recovered from the source, and
recovered *better* later — by whatever knows how to draw them at the time somebody
asks, rather than by whatever happened to be running when the thing occurred. A
recording that keeps images has already decided how they should look; this one has not.

## The window ends at the present

By the time anything is worth recording it has already happened, and what a reviewer
needs is the approach: the speed it was carrying, whether anyone was already in the
way. What came after is the consequence, and it is still in the source for anyone who
wants it.

Buffering forward would mean holding every frame until enough future had arrived to
prove nothing happened — on the overwhelming majority of frames, where nothing ever
does.

`over(seconds, fps)` sizes the ring at `seconds * fps + 1`. The `+ 1` is the moment
itself: the frame is recorded before the window is read, so what comes back is the
full lead-up *plus* the present. Sizing it exactly would push the oldest frame out to
make room and leave every window a frame short.

**If you process video in chunks, keep the window no longer than the overlap between
them.** A window that reaches back past the start of a chunk is truncated in silence,
and the record simply looks like a shorter one.


## Frames in, tracks out

The buffer records by frame, because that is how frames arrive and because expiry is a
property of a frame rather than of a track. Anyone reading the evidence wants the
opposite cut — what did *this* object do — and `window_for` is that pivot.

A `TrackWindow` is six parallel tuples: `frame_indices`, `positions`, `speeds`,
`bboxes`, `class_names`, `timestamps`. **Index *i* of every one describes the same
frame**, which is what makes a position matchable to the box it was measured from. It
is checked on construction rather than trusted, because the failure is otherwise silent
and arrives as a box drawn around the wrong second of footage.

**A track missing from a frame is absent, not padded.** Detection flickers; a car
behind a bus for half a second is not a car that was nowhere. Padding the gap with a
repeated position would invent evidence and padding it with a zero would invent worse,
so the window is simply shorter and `frame_indices` says where the holes were.

`window_for()` with no argument takes every track in the buffer — what you want when
the question is "who else was there", and bounded by the window rather than by the run.
A track nobody recorded comes back as nothing at all rather than as an empty window:
handing one back invites a caller to write it down as evidence that the track did
nothing.

Reading does not consume. Two things firing on one frame each get their own answer.

## Nothing is computed here

Every number in a window is a number somebody handed in. Positions come from whoever
projected them, boxes from whoever detected them; this package only decides which
frames belong to which track.

`positions` and `speeds` are `None` on any frame nothing projected — not a stand-in
derived from the box. Without a camera model there is no ground plane and no honest
position to report, which is exactly why `trajectory_collector.NullCollector` reports
nothing at all. A reader that wants somewhere to draw has `bboxes`, which is always
present.

The frame is still part of the record either way. An object above the horizon, or one
in its filter's first frames, was visibly there.

## Classification is carried, not judged

`class_name` is carried and never compared to anything. Which names are vehicles and
which are pedestrians is a question about traffic rules, answered in one place
elsewhere; a second copy in here would be a copy to keep in step.

It is kept **per frame**, because classification flickers: an object read as a car for
forty frames and a truck for two was read both ways. `TrackWindow.class_name` is a
derived convenience for a caller that has to pick one — most often seen, ties going to
the most recent — and the per-frame record stays the truth.

## What else it will not do

**It does not read a clock.** `FrameEntry.timestamp` is the caller's. `time.time()` at
the moment of recording is when the *analysis* ran, which for a video file is hours
after the thing it describes. A caller reading a file passes footage time, one reading
a live stream passes the wall clock, and both are right.

**It does not detect, track, or project.** Boxes, ids and positions arrive as plain
numbers, from whatever produced them.

## Immutability

Everything recorded is frozen, and a list of objects handed in becomes a tuple on the
way past. `entries()` returns a reading taken then, not a view that keeps moving.

This is not ceremony. A record here is read long after the frame it describes is gone,
when nothing is left to check it against — and the pipeline this is ported from kept
its records as dicts and then, at save time, assigned into the ones still sitting in
its buffer, so reading the evidence changed it.

## Lifetime

One buffer per tracking session. Tracker ids restart at 1 for every video, so a buffer
outliving one would hand back a window mixing two different objects that happened to
be numbered the same — the same rule the tracker, the trajectory collector and the
rule modules all already follow.

## Install

```
pip install -e packages/evidence-collector   # from a checkout of traffic-violation
```

**No dependencies.** Not numpy, not OpenCV, and nothing from this repository. The
other two distributions here each earn one — `trajectory-collector` projects with a
matrix, `violation-detector` runs point-in-polygon tests — but keeping records and
handing them back is a deque and some tuples. It lives in this repository for now, and
the day a second project needs it, it moves out whole.
