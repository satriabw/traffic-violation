"""Traffic violations, from tracked objects and a site's annotated regions.

    from violation_detector import Configuration, TrackedObject

    configuration = Configuration.from_document(document)

Which rules a site runs, and the lanes, lights and regions of interest they run
against, are all in that document. Nothing here has to be told what a junction looks
like in code.

The re-exports are deliberate, and the same departure `trajectory_collector` makes: an
application's `__init__.py` stays empty because its modules are imported by path, but
a library's import path is its API, and moving a module should not break anyone.

WHAT IS NOT HERE. The rules themselves, and the factory that builds them from a
configuration. This package currently parses and validates a site's document and
answers where an object is; detecting anything is the next thing to land.
"""

from violation_detector.objects import PEDESTRIANS, VEHICLES, TrackedObject, Violation
from violation_detector.regions import (
    Configuration,
    ConfigurationInvalid,
    Region,
    Regions,
    TrafficLight,
)

__all__ = [
    "PEDESTRIANS",
    "VEHICLES",
    "Configuration",
    "ConfigurationInvalid",
    "Region",
    "Regions",
    "TrackedObject",
    "TrafficLight",
    "Violation",
]
