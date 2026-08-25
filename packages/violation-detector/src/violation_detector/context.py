"""What has happened up to now, kept across frames.

The second stage. A single frame cannot tell you that a vehicle *entered* the box — it
only shows you a vehicle inside it — and it cannot tell you the light was already red
when that happened. Both are differences between frames, and this is the only thing in
the package that remembers one frame while looking at the next.

LIFETIME IS THE JOB. The caches here are keyed by tracker id, and tracker ids restart
at 1 for every video, so a builder that outlived its job would answer questions about
a completely different vehicle. That is the same rule the tracker and the trajectory
collector already follow: one per job, discarded with it.

NOTHING IS EVICTED. A vehicle that left the frame a thousand frames ago keeps its
entry. A junction sees thousands of vehicles in an hour and each one costs a handful
of fields, which is nothing next to the frames themselves — and eviction would need to
know that a track is finished, which nothing here is told.
"""

from collections import defaultdict
from dataclasses import dataclass

from violation_detector.observations import (
    CrossingObservation,
    Observation,
    Occupant,
    VehicleObservation,
)
from violation_detector.traffic_light import RED

# No entry recorded. -1 rather than None so the comparison in the rule — did this
# vehicle enter the box after the light turned red — is always between two numbers.
NEVER = -1


@dataclass(frozen=True)
class LightState:
    """A light, and when it last turned red."""

    state: str
    # The frame this light most recently went from not-red to red, or NEVER. What
    # makes "entered on red" different from "was already inside when it changed": a
    # vehicle that crossed on green and is still in the box when the light changes has
    # an entry frame earlier than this one.
    start_red: int


@dataclass(frozen=True)
class VehicleState:
    """A vehicle, and what it has done so far."""

    track_id: int
    # The last lane it was seen in, which is not always the lane it is in now. A
    # vehicle inside the box past the stop line is, by then, in no lane at all — so
    # the lane that decides which light governs it has to be remembered from before it
    # got there. This is the whole reason this stage exists.
    last_lane: str | None
    in_roi: bool
    # Whether it was already inside the box on an earlier frame. What keeps one
    # crossing from being reported on every frame of the crossing.
    prev_in_roi: bool
    enter_roi_frame: int


@dataclass(frozen=True)
class Context:
    """Everything the rules are allowed to look at."""

    lights: dict[str, LightState]
    vehicles: tuple[VehicleState, ...]


@dataclass
class _LightMemory:
    state: str
    start_red: int


@dataclass
class _VehicleMemory:
    last_lane: str | None
    enter_roi_frame: int
    prev_in_roi: bool


class RedLightRunningContext:
    """Folds each frame's observation into what is known so far."""

    def __init__(self) -> None:
        self._lights: dict[str, _LightMemory] = {}
        self._vehicles: dict[int, _VehicleMemory] = {}

    def update(self, observation: Observation, frame_index: int) -> Context:
        return Context(
            lights={
                light.id: self._light(light.id, light.state, frame_index)
                for light in observation.lights
            },
            vehicles=tuple(
                self._vehicle(vehicle, frame_index) for vehicle in observation.vehicles
            ),
        )

    def _light(self, id: str, state: str, frame_index: int) -> LightState:
        memory = self._lights.get(id)
        if memory is None:
            # First sighting. A light that is already red has been red since as far
            # back as anything here can see, and this frame is the earliest defensible
            # answer — which makes every vehicle already inside the box look like it
            # entered on green. That is the conservative direction, and at a chunk
            # boundary it is why the job's chunks overlap.
            memory = _LightMemory(state=state, start_red=frame_index if state == RED else NEVER)
            self._lights[id] = memory
        elif state == RED and memory.state != RED:
            memory.start_red = frame_index

        memory.state = state
        return LightState(state=memory.state, start_red=memory.start_red)

    def _vehicle(self, observation: VehicleObservation, frame_index: int) -> VehicleState:
        in_roi = observation.roi is not None
        memory = self._vehicles.get(observation.track_id)

        if memory is None:
            memory = _VehicleMemory(
                last_lane=observation.lane,
                enter_roi_frame=frame_index if in_roi else NEVER,
                prev_in_roi=False,
            )
            self._vehicles[observation.track_id] = memory

        # Only a real lane overwrites the remembered one; None means "not in a lane
        # right now", which is exactly the case this field exists to survive.
        if observation.lane is not None:
            memory.last_lane = observation.lane

        # ORDER MATTERS between these two. On the frame a vehicle enters the box the
        # entry is still unset, so prev_in_roi stays False and the rule may fire; from
        # the next frame on it is True and the same crossing cannot fire again.
        #
        # An entry is never cleared on the way out, so prev_in_roi stays True for the
        # life of the track: a vehicle that crosses, leaves and crosses again is
        # reported for the first crossing only. That is the source pipeline's
        # behaviour and it is the safe direction to be wrong in, but it is a real
        # limit — at a junction a car can legitimately re-enter the box minutes later.
        #
        # A vehicle first seen ALREADY INSIDE the box is the exception: its entry was
        # set the moment its memory was created, so this marks it immediately and it
        # can never be reported. Deliberate, and preserved from the source pipeline —
        # we never saw it enter, so we cannot say it entered on red. It is also why
        # the LLD has consecutive chunks overlap: a vehicle mid-crossing at a chunk
        # boundary would otherwise be invisible to both chunks.
        if in_roi and memory.enter_roi_frame != NEVER:
            memory.prev_in_roi = True
        if in_roi and memory.enter_roi_frame == NEVER:
            memory.enter_roi_frame = frame_index

        return VehicleState(
            track_id=observation.track_id,
            last_lane=memory.last_lane,
            in_roi=in_roi,
            prev_in_roi=memory.prev_in_roi,
            enter_roi_frame=memory.enter_roi_frame,
        )


@dataclass(frozen=True)
class CrossingVehicle:
    """A vehicle, and whether this frame is the one it drove in on."""

    track_id: int
    in_roi: bool
    # Whether it was already inside on an earlier frame. This is what separates
    # *entering* an occupied crossing from *being* in one that later becomes occupied —
    # a car queued in the box before anybody stepped off the kerb has not failed to
    # yield to them. It is also what keeps one crossing from being reported on every
    # frame of the crossing.
    prev_in_roi: bool


@dataclass(frozen=True)
class Occupancy:
    """Who is in one region of interest, right now."""

    # Only those actually inside it. A vehicle short of the line and a pedestrian on
    # the far pavement are not in this crossing and are not here.
    vehicles: tuple[CrossingVehicle, ...]
    pedestrians: tuple[Occupant, ...]


@dataclass(frozen=True)
class CrossingContext:
    """Every region of interest that has anything in it, keyed by its id.

    Keyed and grouped because the rule is about co-presence: a vehicle and a pedestrian
    matter to each other only when they are in the *same* crossing. A junction with two
    crossings would otherwise have one pedestrian make violators of vehicles entering
    the other.
    """

    rois: dict[str, Occupancy]


@dataclass
class _CrossingMemory:
    enter_roi_frame: int
    prev_in_roi: bool


class PedestrianRightOfWayContext:
    """Folds each frame's crossing observation into what is known so far.

    Per job, and for the same reason everything else here is: the cache is keyed by
    tracker id, and those restart at 1 for every video.
    """

    def __init__(self) -> None:
        self._vehicles: dict[int, _CrossingMemory] = {}

    def update(self, observation: CrossingObservation, frame_index: int) -> CrossingContext:
        vehicles: dict[str, list[CrossingVehicle]] = defaultdict(list)
        pedestrians: dict[str, list[Occupant]] = defaultdict(list)

        for occupant in observation.vehicles:
            # Every vehicle is folded into the cache, including one outside every
            # crossing: leaving it out would make its next arrival look like a first
            # sighting. Only those actually inside one are grouped.
            vehicle = self._vehicle(occupant, frame_index)
            if occupant.roi is not None:
                vehicles[occupant.roi].append(vehicle)

        for occupant in observation.pedestrians:
            if occupant.roi is not None:
                pedestrians[occupant.roi].append(occupant)

        return CrossingContext(
            rois={
                roi: Occupancy(
                    vehicles=tuple(vehicles[roi]), pedestrians=tuple(pedestrians[roi])
                )
                for roi in vehicles.keys() | pedestrians.keys()
            }
        )

    def _vehicle(self, occupant: Occupant, frame_index: int) -> CrossingVehicle:
        in_roi = occupant.roi is not None
        memory = self._vehicles.get(occupant.track_id)

        if memory is None:
            memory = _CrossingMemory(
                enter_roi_frame=frame_index if in_roi else NEVER, prev_in_roi=False
            )
            self._vehicles[occupant.track_id] = memory

        # The same two lines, in the same order, as the red-light context, and they
        # carry the same three consequences: not a repeat on the frame it enters, a
        # repeat from the next frame on, and a vehicle first seen already inside marked
        # immediately and never reportable. See `RedLightRunningContext._vehicle`.
        if in_roi and memory.enter_roi_frame != NEVER:
            memory.prev_in_roi = True
        if in_roi and memory.enter_roi_frame == NEVER:
            memory.enter_roi_frame = frame_index

        return CrossingVehicle(
            track_id=occupant.track_id, in_roi=in_roi, prev_in_roi=memory.prev_in_roi
        )
