"""The seam: one rule, behind an interface that does not say how it decides.

A module is handed a frame and the objects tracked on it, and answers with whatever
violations it can see. Nothing outside asks how — which is the whole point of putting
the boundary here rather than inside the worker. A model-based detector is another
class implementing `detect`, registered under its own name, and neither the worker nor
the frame pipeline learns that anything changed. A site can name both in its
configuration and run them side by side.
"""

from abc import ABC, abstractmethod
from typing import ClassVar, Sequence

import numpy as np

from violation_detector.context import (
    PedestrianRightOfWayContext,
    RedLightRunningContext,
)
from violation_detector.objects import TrackedObject, Violation
from violation_detector.observations import (
    PedestrianRightOfWayObserver,
    RedLightRunningObserver,
)
from violation_detector.regions import Regions
from violation_detector.rules import failing_to_yield, is_running_red

# The canonical name, as everything outside this package knows it — deliberately the
# value the queue's ViolationType uses, not the "rlr_violation" a document says.
RED_LIGHT_RUNNING = "red_light_running"
PEDESTRIAN_RIGHT_OF_WAY = "pedestrian_right_of_way"


class ViolationModule(ABC):
    """One kind of violation, watched for across a job."""

    # What this module reports. Set by each subclass and used as `Violation.type`, so
    # nothing has to keep a table mapping classes to names.
    type: ClassVar[str]

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        tracked_objects: Sequence[TrackedObject],
        frame_index: int,
    ) -> list[Violation]:
        """Violations visible as of this frame. Empty is the normal answer."""

    def finish(self) -> list[Violation]:
        """Anything still held back, now that there are no more frames.

        Empty for a rule module: a rule decides on the frame it is given and has
        nothing in hand when the video ends. A module that works on a *clip* is the
        reason this exists — a classifier over a sliding window is always holding a
        partial one, and without somewhere to flush it the last seconds of every job
        would be silently dropped. The worker drains this once after the frame loop.

        It is also what makes `Violation.frame_index` load-bearing rather than
        decorative. A buffering module reports frame 900 while being called with frame
        930, so a caller that records the loop's index instead of the violation's will
        be wrong by the length of the window.
        """
        return []


class RedLightRunningModule(ViolationModule):
    """Crossing the stop line after the light governing your lane turned red.

    Per job — it holds the caches in `RedLightRunningContext`, which are keyed by
    tracker ids that mean nothing outside the video that produced them.
    """

    type: ClassVar[str] = RED_LIGHT_RUNNING

    def __init__(self, regions: Regions):
        # A scene with no light, no region of interest, or no light declaring
        # `controls` builds a module that can never fire. That is the source
        # pipeline's behaviour and it is left alone deliberately: refusing here would
        # reject documents that run today.
        self._controlled_lanes = {
            light.id: light.controls for light in regions.traffic_lights if light.controls
        }
        self._observer = RedLightRunningObserver(regions)
        self._context = RedLightRunningContext()

    def detect(
        self,
        frame: np.ndarray,
        tracked_objects: Sequence[TrackedObject],
        frame_index: int,
    ) -> list[Violation]:
        observation = self._observer.observe(frame, tracked_objects)
        context = self._context.update(observation, frame_index)
        return [
            # No confidence: a rule fired or it did not, and a 1.0 here would be a
            # score nobody computed.
            Violation(type=self.type, track_id=vehicle.track_id, frame_index=frame_index)
            for vehicle in context.vehicles
            if is_running_red(vehicle, context.lights, self._controlled_lanes)
        ]


class PedestrianRightOfWayModule(ViolationModule):
    """Driving into a crossing somebody is already in.

    Per job, on the same terms as every other module here: its context caches what it
    has seen keyed by tracker id.
    """

    type: ClassVar[str] = PEDESTRIAN_RIGHT_OF_WAY

    def __init__(self, regions: Regions):
        self._observer = PedestrianRightOfWayObserver(regions)
        self._context = PedestrianRightOfWayContext()

    def detect(
        self,
        frame: np.ndarray,
        tracked_objects: Sequence[TrackedObject],
        frame_index: int,
    ) -> list[Violation]:
        observation = self._observer.observe(frame, tracked_objects)
        context = self._context.update(observation, frame_index)
        return [
            Violation(type=self.type, track_id=track_id, frame_index=frame_index)
            # Sorted, so two crossings reported on one frame come out in a stable order
            # rather than in whatever order a dict happened to iterate.
            for roi in sorted(context.rois)
            for track_id in failing_to_yield(context.rois[roi])
        ]
