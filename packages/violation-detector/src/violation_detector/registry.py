"""Which rules exist, and how to build one.

A document names its rules the way the annotation tooling always has —
`"rlr_violation"` — and everything outside this package names them the way the queue
does, `"red_light_running"`. The registry is the one place that holds both, so a
document keeps the vocabulary its authors type and a caller never has to carry a
translation table.

FACTORIES, NOT CLASSES. The source pipeline's registry mapped a name straight to a
class and called it. That works while every rule is cheap to construct, and stops
working the moment one is not: a model-based module needs a session loaded from
weights, which costs seconds and belongs to the process rather than to the job. A
factory can close over a session prepared once at startup and hand it to every job's
module, which a class in a dict cannot. Rules pay nothing for this — theirs ignore the
context and call a constructor.

`register` is public for the same reason. A rule living outside this package — a
model-based detector shipped separately, an experiment — becomes available to every
document by registering under its own name, with nothing here edited.
"""

from dataclasses import dataclass
from typing import Callable

from violation_detector.modules import (
    RED_LIGHT_RUNNING,
    RedLightRunningModule,
    ViolationModule,
)
from violation_detector.regions import ConfigurationInvalid, Regions


@dataclass(frozen=True)
class ModuleContext:
    """Everything a rule may need to build itself, and nothing job-specific beyond it.

    Deliberately small, and deliberately not the `Configuration`: a module has no
    business reading which *other* rules a site runs.
    """

    regions: Regions
    # Frames per second, when the caller knows it. A rule ignores this — it works in
    # frame indices throughout. A module that reasons about a span of time cannot: a
    # classifier over a two-second window has to know how many frames that is, and
    # nothing downstream can recover it from the frames alone.
    fps: float | None = None


# Built once per job, from the context. Returning the module rather than a class is
# what lets a factory close over something expensive that outlives the job.
ModuleFactory = Callable[[ModuleContext], ViolationModule]

# Keyed by the name a document uses, valued by the canonical type it reports and the
# factory that builds it.
_REGISTRY: dict[str, tuple[str, ModuleFactory]] = {}


def register(name: str, type: str, factory: ModuleFactory) -> None:
    """Make a rule available to any document naming it.

    Replacing an existing entry is allowed and is how a deployment swaps a rule
    implementation for its own — the last registration wins.
    """
    _REGISTRY[name] = (type, factory)


def registered() -> tuple[str, ...]:
    """The document names this build understands."""
    return tuple(sorted(_REGISTRY))


def factory_for(name: str) -> tuple[str, ModuleFactory]:
    """Resolve a document's rule name, or say what would have been understood."""
    if name not in _REGISTRY:
        raise ConfigurationInvalid(
            f"configuration.violations names {name!r}, which is not a rule this build "
            f"carries — expected one of {', '.join(registered())}"
        )
    return _REGISTRY[name]


register(
    "rlr_violation",
    RED_LIGHT_RUNNING,
    lambda context: RedLightRunningModule(context.regions),
)
