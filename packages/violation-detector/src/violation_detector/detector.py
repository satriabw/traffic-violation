"""One call in, every violation out.

The whole package behind three names. A caller hands over a site's configuration and
gets something with `detect`; which rules that turns into, and how many, is the
document's business and never the caller's.

    detector = get_detector(configuration, types=[t.value for t in job.types])

    for index, frame in enumerate(frames):
        violations = detector.detect(frame, tracked_objects, index)

    violations = detector.finish()

That the worker holds no table mapping its `ViolationType` values to rule names is the
point of the `types` argument taking canonical values.

WHY A FACADE RATHER THAN A LIST. A caller given a list has to loop it, flatten the
results, and remember to drain each module at the end — three chances to differ
between the worker, a test, and a benchmark. Aggregation belongs in one place.
"""

from typing import Iterable, Sequence

import numpy as np

from violation_detector.modules import ViolationModule
from violation_detector.objects import TrackedObject, Violation
from violation_detector.regions import Configuration
from violation_detector.registry import ModuleContext, factory_for


class Detector:
    """Every rule a job runs, asked together.

    Per job, because its modules are: they hold caches keyed by tracker ids, which
    restart at 1 for every video.
    """

    def __init__(self, modules: Sequence[ViolationModule] = ()):
        self._modules = tuple(modules)

    def detect(
        self,
        frame: np.ndarray,
        tracked_objects: Sequence[TrackedObject],
        frame_index: int,
    ) -> list[Violation]:
        """Every violation visible as of this frame, from every rule.

        In registration order, which is the document's order. Nothing deduplicates
        across modules: two rules reporting the same vehicle on the same frame are two
        different offences, and which of them to keep is the caller's judgement.
        """
        violations = []
        for module in self._modules:
            violations.extend(module.detect(frame, tracked_objects, frame_index))
        return violations

    def finish(self) -> list[Violation]:
        """Anything still held back, now that the frames have run out.

        Must be called once at the end of a job, and the violations it returns carry
        their own `frame_index` — earlier than the last frame, for anything that
        buffers. Empty when every rule is a rule.
        """
        violations = []
        for module in self._modules:
            violations.extend(module.finish())
        return violations

    def get_modules(self) -> tuple[ViolationModule, ...]:
        """The rules this detector runs. For tests and for a worker's summary line."""
        return self._modules


def get_detector(
    configuration: Configuration,
    types: Iterable[str] | None = None,
    fps: float | None = None,
) -> Detector:
    """Build the rules a configuration asks for.

    `types` narrows that to what a job actually wants, in canonical values —
    "red_light_running", not "rlr_violation". None means every rule the document
    names.

    The two ways the sets can fail to meet are both no-ops rather than errors. A job
    asking for a violation this site is not configured for gets nothing for it: the
    site is the authority on what can be watched for. A site configured for a rule the
    job did not ask for does not run it: the job is the authority on what was wanted.
    Neither is a mistake, and a job refused for either would be a job refused for
    asking a reasonable question.
    """
    wanted = None if types is None else set(types)
    context = ModuleContext(regions=configuration.regions, fps=fps)

    modules = []
    for name in configuration.violations:
        type, factory = factory_for(name)
        if wanted is None or type in wanted:
            modules.append(factory(context))
    return Detector(modules)
