# evidence-collector

The last few seconds of what every tracked object was doing, kept until something
makes it worth recording.

```python
from evidence_collector import FrameBuffer, FrameEntry, ObjectState

buffer = FrameBuffer.over(seconds=5, fps=30)

for frame_index, tracked in analysed_frames:
    buffer.add(FrameEntry(
        frame_index=frame_index,
        objects=[ObjectState(track_id=7, bbox=(x1, y1, x2, y2), class_name="car")],
    ))

    if something_happened:
        window = buffer.entries()   # the lead-up, oldest first
```

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

## What it will not do

**It does not know what anything is.** `class_name` is carried and never compared to
anything. Which names are vehicles and which are pedestrians is a question about
traffic rules, answered in one place elsewhere; a second copy in here would be a copy
to keep in step. Hold the label, decide what it means when you read the window.

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
