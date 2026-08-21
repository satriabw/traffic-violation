# trajectory-collector

Where a tracked object is on the ground, and how fast it is going, from its bounding
box in the image.

```python
from trajectory_collector import TrajectoryCollector

collector = TrajectoryCollector.from_calibration("camera_model.yml", fps=30.0)

for frame_index, boxes, track_ids in tracked_frames:
    trajectories = collector.collect(boxes, track_ids, frame_index)
    # {track_id: Trajectory(position=(x, y), speed=…)}
```

`boxes` is `(N, 4)` `xyxy` in pixels and `track_ids` is `(N,)` of ints, aligned row for
row — whatever your detector and tracker produce, in the shape every one of them
already has. Nothing here detects or tracks.

**Positions are in metres and speeds in metres per second**, on the ground plane. There
is no unit field: a calibration is built in some real-world unit and every number
downstream inherits it, so a calibration in anything other than metres is a wrong
calibration rather than a differently-configured one.

`frame_index` is the frame's absolute position in the video, not a call count. It is
what lets the filter measure how long a track was missing for, so a caller processing a
chunk starting at frame 900, or sampling every third frame, gets gaps measured in real
elapsed time.

A collector holds a filter per track and track ids restart at 1 for every tracking
session, so build one per video.

## Status

The interface and `NullCollector` — the "this video has no calibration" case — are in.
`from_calibration` and the pinhole collector behind it land next.

## Install

```
pip install -e packages/trajectory-collector   # from a checkout of traffic-violation
```

Depends on numpy and nothing else. It lives in this repository for now, but nothing in
it imports anything from this repository — the day a second project needs it, it moves
out whole.

**The name `trajectory-collector` is taken on PyPI** by an unrelated project, currently
at 0.1.1. Nothing here is published, so this only matters as a footgun: anything that
declares a bare `trajectory-collector` dependency without the local checkout alongside
it will get the stranger's package instead. Always install this one explicitly — which
is what the repo root README, the CI job and the `pythonpath` in the root
`pyproject.toml` all do. The same is true of `shared`, for the same reason. Publishing
would mean either claiming a different name or taking that one up with its owner.
