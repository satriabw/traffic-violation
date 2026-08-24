"""What one frame shows: where each vehicle is, and what each light is doing.

The first of the three stages a rule module runs. This one looks and reports, and it
remembers nothing — hand it the same frame twice and it answers the same thing twice.
Everything that depends on what happened *earlier* lives in `context`, and everything
that decides whether that amounts to a violation lives in `rules`.

Keeping the stages apart is what makes the rules testable: a truth table over a
handful of frozen values, with no frame, no OpenCV and no tracker anywhere near it.

TYPED, NOT DICTS. The pipeline this is ported from passed dicts between the three
stages, and its context builder emitted `{"id", "last_lane", "in_roi",
"enter_roi_frame"}` while its rule read `ctx["vehicle"]["prev_in_roi"]` — a key nobody
put there. That is a KeyError on the first vehicle to reach a controlled lane, which
is to say red-light running has never once run to completion in that pipeline. The
same mistake against these dataclasses does not survive being written down.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from violation_detector.geometry import RegionIndex, crop, polygon_to_bbox
from violation_detector.objects import TrackedObject
from violation_detector.regions import Regions
from violation_detector.traffic_light import state_from_pixels


@dataclass(frozen=True)
class LightObservation:
    """One light on one frame."""

    id: str
    # One of the constants in `traffic_light` — UNKNOWN as readily as a colour.
    state: str


@dataclass(frozen=True)
class VehicleObservation:
    """One vehicle on one frame, placed against the scene's regions."""

    track_id: int
    # The lane it is in, or None for a vehicle between lanes, off the annotated road,
    # or in a part of the junction no lane covers. Not an error: most of the frame is
    # not a lane.
    lane: str | None
    # The region of interest it is in, by the same rule. For red-light running this is
    # the box past the stop line — being inside it is what "crossed" means.
    roi: str | None


@dataclass(frozen=True)
class Occupant:
    """One object on one frame, placed against the regions of interest alone.

    What a rule about a crossing needs and all it needs: which region something is in,
    if any. No lane, because a crossing does not care where a vehicle came from.
    """

    track_id: int
    roi: str | None


@dataclass(frozen=True)
class Observation:
    """Everything one frame has to say about red-light running."""

    lights: tuple[LightObservation, ...]
    vehicles: tuple[VehicleObservation, ...]


@dataclass(frozen=True)
class CrossingObservation:
    """Everything one frame has to say about who is in a crossing."""

    vehicles: tuple[Occupant, ...]
    pedestrians: tuple[Occupant, ...]


class RedLightRunningObserver:
    """Reads a frame for the things red-light running is about.

    Built once per job. The region indices and the lights' crop rectangles are
    prepared here rather than per frame: the source pipeline recomputed every light's
    bounding box twice on every frame — once to crop it and once to report it — and
    rebuilt every polygon's contour for every vehicle it tested.
    """

    def __init__(self, regions: Regions):
        self._lanes = RegionIndex(regions.lanes)
        self._rois = RegionIndex(regions.rois)
        self._lights = tuple(
            (light.id, polygon_to_bbox(light.points)) for light in regions.traffic_lights
        )

    def observe(
        self, frame: np.ndarray, tracked_objects: Sequence[TrackedObject]
    ) -> Observation:
        """Look at one frame.

        Every configured light is reported on every frame, whether or not it could be
        read — a light missing from this list would be indistinguishable to a rule
        from a light that is not red.
        """
        lights = tuple(
            LightObservation(id=id, state=state_from_pixels(crop(frame, bbox)))
            for id, bbox in self._lights
        )
        # Only vehicles. A pedestrian cannot run a red light, and locating one costs a
        # point-in-polygon test against every lane and every ROI in the scene.
        vehicles = tuple(
            VehicleObservation(
                track_id=object.track_id,
                lane=self._lanes.locate(object.bbox),
                roi=self._rois.locate(object.bbox),
            )
            for object in tracked_objects
            if object.is_vehicle
        )
        return Observation(lights=lights, vehicles=vehicles)


class PedestrianRightOfWayObserver:
    """Reads a frame for who is standing in a crossing and who is driving into it.

    Simpler than the red-light observer in one way that matters: there is nothing in
    the pixels it needs. A crossing has no light to read, so the frame is accepted and
    ignored, and every question is answered by geometry against the regions of
    interest.

    A cyclist counts as a pedestrian and a motorbike does not. What these rules care
    about is whether a thing is vulnerable and has right of way, not what it is
    riding — see `objects`.
    """

    def __init__(self, regions: Regions):
        self._rois = RegionIndex(regions.rois)

    def observe(
        self, frame: np.ndarray, tracked_objects: Sequence[TrackedObject]
    ) -> CrossingObservation:
        vehicles, pedestrians = [], []
        for object in tracked_objects:
            # Anything that is neither is simply not this rule's business — a traffic
            # light, or a class the model had no name for. The pipeline this is ported
            # from could not express that: its parser admitted only vehicles and
            # people, and its context builder filed everything that was not a vehicle
            # under pedestrians, so a third kind of thing would have been counted as
            # somebody waiting to cross.
            if object.is_vehicle:
                vehicles.append(Occupant(object.track_id, self._rois.locate(object.bbox)))
            elif object.is_pedestrian:
                pedestrians.append(Occupant(object.track_id, self._rois.locate(object.bbox)))
        return CrossingObservation(
            vehicles=tuple(vehicles), pedestrians=tuple(pedestrians)
        )
