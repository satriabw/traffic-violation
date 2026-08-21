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

A track reports a position but no speed for its first few frames — a filter needs a
velocity to start from, and one sighting gives none, so the warmup is spent measuring
one. A box whose bottom edge lands on or above the horizon has no ground point at all
and is skipped for that frame rather than reported.

## Calibration documents

```json
{
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "rot_matrix":    [[...], [...], [...]],
  "tvec":          [tx, ty, tz]
}
```

Pass the path to a file like that, or the parsed mapping. Any field may also be a flat
row-major list — the form OpenCV's `FileStorage` writes — and `tvec` is accepted as a
column. Extra keys are ignored.

OpenCV `.yml` calibrations are not read directly, because parsing one needs OpenCV and
this package depends on numpy alone. Convert once:

```python
import cv2, json
fs = cv2.FileStorage("camera_model.yml", cv2.FILE_STORAGE_READ)
json.dump({
    "camera_matrix": fs.getNode("camera_matrix").mat().tolist(),
    "rot_matrix": fs.getNode("rot_matrix").mat().tolist(),
    "tvec": fs.getNode("tvec").mat().reshape(-1).tolist(),
}, open("camera_model.json", "w"))
```

An unusable document raises `CalibrationInvalid` from `from_calibration` — at
construction, never on some frame in the middle of a video.

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
