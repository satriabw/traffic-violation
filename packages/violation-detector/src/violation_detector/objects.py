"""What a caller hands in, and what it gets back.

The whole vocabulary this package has for the outside world is here: an object that
has been detected and tracked, and a violation that some object committed. Neither
mentions a detection library, a model, or a frame source, which is what lets the same
rules run behind supervision here and behind something else elsewhere.

OBJECTS ARE NAMED, NOT NUMBERED. The pipeline this is ported from matched detections
by class id — `np.isin(detections.class_id, list(VEHICLE_CLASSES.keys()))` — against a
table that had to be edited by hand for every model, and its file still carries three
commented-out blocks for YOLO, MIO-TCD and COCO with different numbers for the same
car. A class id means nothing without knowing which model produced it; a name means
the same thing everywhere. Translating one into the other belongs at the detector,
which is the only thing that knows what it loaded.
"""

from dataclasses import dataclass

# Names, so a detector emitting COCO's "motorbike" and one emitting "motorcycle" are
# both understood without either of them being reconfigured.
#
# MOTORBIKES ARE VEHICLES. In the pipeline this is ported from that is true in one rule
# and false in the other. Its `VEHICLE_CLASSES` includes motorbike, but its `VEHICLES`
# set — {car, bus, truck} — does not, and which of the two a rule ends up consulting is
# decided by its parser. The red-light parser stamps `type: "car"` on every vehicle it
# selects, so a motorbike is a vehicle there. The pedestrian parser uses the real class
# name, so a motorbike misses `VEHICLES`, falls through the `else` branch and is counted
# as a *pedestrian*, which makes every car sharing its region a violator. One definition
# used by everything is the fix: it matches the red-light rule exactly as it stands, and
# changes only the pedestrian rule, which is called out where that rule lands.
VEHICLES = frozenset({"car", "bus", "truck", "motorbike", "motorcycle"})

# A cyclist is a person on a bicycle, and every detector names that differently. All
# of them land here: what these rules care about is that the thing is vulnerable and
# has right of way, not what it is riding.
PEDESTRIANS = frozenset({"person", "pedestrian", "bicycle", "cyclist", "e-scooter"})


@dataclass(frozen=True)
class TrackedObject:
    """One object on one frame, after detection and tracking.

    Frozen, and built from plain numbers rather than holding a row of somebody's
    array: a module may keep one of these in a cache for as long as a track lives, and
    a view into a per-frame buffer would say something different by the time it is
    read.
    """

    # The tracker's id. Stable across the frames of one tracking session and
    # meaningless outside it — ids restart at 1 for every video.
    track_id: int
    # (x1, y1, x2, y2) in pixels, top-left origin. The convention every detection
    # library in reach already uses.
    bbox: tuple[float, float, float, float]
    # What the detector called it. Compared against the sets above, and an unrecognised
    # name is simply an object no rule here has an opinion about — never an error.
    class_name: str

    @property
    def is_vehicle(self) -> bool:
        return self.class_name in VEHICLES

    @property
    def is_pedestrian(self) -> bool:
        return self.class_name in PEDESTRIANS


@dataclass(frozen=True)
class Violation:
    """One object, breaking one rule, on one frame."""

    # The canonical type, as the registry names it — "red_light_running", not the
    # config's "rlr_violation". Callers record this, so it is the name that has to
    # stay stable, and it is deliberately the one the queue's ViolationType uses.
    type: str
    # Which object. The tracker's id, so it only means anything alongside the run that
    # produced it.
    track_id: int
    # WHEN IT HAPPENED, which is not necessarily the frame that was being analysed
    # when this was returned. A rule module reports on the frame it was given, but a
    # module that buffers — anything working on a clip rather than a frame — reports
    # several frames late, and this is the index that has to reach the record.
    frame_index: int
    # None from a rule: a rule either fired or it did not, and 1.0 would be a score
    # nobody computed. A model-based module puts its score here.
    confidence: float | None = None
