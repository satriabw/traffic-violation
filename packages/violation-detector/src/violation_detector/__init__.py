"""Traffic violations, from tracked objects and a site's annotated regions.

    from violation_detector import Configuration, TrackedObject, get_detector

    configuration = Configuration.from_document(document)
    detector = get_detector(configuration, types=[t.value for t in job.types])

    for index, frame in enumerate(frames):
        violations = detector.detect(frame, tracked(index), index)

    violations = detector.finish()

Which rules a site runs, and the lanes, lights and regions of interest they run
against, are all in that document. Nothing here has to be told what a junction looks
like in code, and nothing outside has to know which rules exist.

The re-exports are deliberate, and the same departure `trajectory_collector` makes: an
application's `__init__.py` stays empty because its modules are imported by path, but
a library's import path is its API, and moving a module should not break anyone.

WHAT IS NOT HERE. Pedestrian right of way, which lands next and needs no change on
this side of the boundary — it registers itself under its own name and any document
already asking for it starts getting it.
"""

from violation_detector.detector import Detector, get_detector
from violation_detector.modules import (
    RED_LIGHT_RUNNING,
    RedLightRunningModule,
    ViolationModule,
)
from violation_detector.objects import PEDESTRIANS, VEHICLES, TrackedObject, Violation
from violation_detector.regions import (
    Configuration,
    ConfigurationInvalid,
    Region,
    Regions,
    TrafficLight,
)
from violation_detector.registry import ModuleContext, register, registered

__all__ = [
    "PEDESTRIANS",
    "RED_LIGHT_RUNNING",
    "VEHICLES",
    "Configuration",
    "ConfigurationInvalid",
    "Detector",
    "ModuleContext",
    "RedLightRunningModule",
    "Region",
    "Regions",
    "TrackedObject",
    "TrafficLight",
    "Violation",
    "ViolationModule",
    "get_detector",
    "register",
    "registered",
]
