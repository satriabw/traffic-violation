"""A site's configuration document, parsed.

    {
      "version": 1,
      "violations": ["rlr_violation"],
      "regions": {
        "lanes":          [{"id": "lane_1", "points": [[115, 640], ...]}],
        "traffic_lights": [{"id": "tl_1", "points": [...], "controls": ["lane_1"]}],
        "rois":           [{"id": "roi_1", "points": [...]}]
      }
    }

Everything a rule needs about a place is in here, and the document is the only thing
that knows it. Which lanes a traffic light governs is a property of the junction, not
of the code, and `controls` is where it is stated.

WHAT THIS REPLACES. The pipeline this is ported from had no configuration document. It
read an annotation tool's export and sorted the polygons by *sniffing their labels* —
`"lane" in label`, `"roi" in label`, `"traffic_light" in label or "tl_" in label` —
which quietly filed a polygon named "roi_at_lane_3" under lanes, and had no way at all
to say which light governed which lane (that arrived separately, as a list of
one-key dicts). Sections are explicit here, and a name is just a name.

WHAT IS AND IS NOT CHECKED. Shape, types, and cross-references between regions, all of
which this module is the only place able to check. NOT which violation names are real
— that is the registry's business, and a document is not wrong for naming a rule this
build happens not to carry.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

# The only format understood. A document announcing anything else is refused rather
# than parsed on the assumption that the parts we recognise still mean what they used
# to: a configuration is read once and used for a whole run, and a silently
# misread one produces a run that looks completely normal.
SUPPORTED_VERSION = 1

# Rejected if a document names a section outside this set — the one place where being
# strict pays. A typo here ("roi" for "rois") would otherwise drop every polygon in
# it, and the only symptom would be a rule that never fires. Unknown keys at the top
# level are left alone by contrast, because a document carrying a name or a note
# alongside its content is harmless.
SECTIONS = ("lanes", "rois", "traffic_lights")

# A polygon needs three points to enclose anything. Two describe a line, which every
# point-in-polygon test in here would answer "outside" for, always.
MINIMUM_POINTS = 3


class ConfigurationInvalid(ValueError):
    """The document cannot be used to detect anything.

    Raised while building a detector, never mid-video — the same bargain
    `CalibrationInvalid` strikes. A configuration is checked once and consulted tens of
    thousands of times, and a job that refused to start is far cheaper than one that
    watched an hour of footage through the wrong polygons.

    The message names the path it failed at (`regions.traffic_lights[0].points`),
    because whoever has to fix it is looking at the json, not at this file.
    """


# eq=False on both region types: `points` is an array, and dataclass equality would
# compare it elementwise and then try to read a bool out of the result. Regions are
# identified by their id anyway, and nothing has cause to ask whether two of them are
# the same object.
@dataclass(frozen=True, eq=False)
class Region:
    """One named area of the scene."""

    id: str
    # (N, 2) float32, in pixels. Float rather than int because these are projected to
    # the ground plane when a camera model is available, and one dtype for both paths
    # means the point-in-polygon test below does not care which space it is in.
    points: np.ndarray


@dataclass(frozen=True, eq=False)
class TrafficLight(Region):
    """A light, and the lanes it governs.

    `controls` is what makes red-light running expressible: a vehicle only runs *this*
    light if it came from a lane this light is responsible for. Validated against the
    declared lanes at parse time, so a rule can index them without checking.
    """

    controls: tuple[str, ...] = ()


@dataclass(frozen=True)
class Regions:
    """Every region in a scene, by kind.

    Empty tuples rather than None for a section the document left out: a site with no
    traffic lights is an ordinary site, and the rules that need one say so when they
    are built.
    """

    lanes: tuple[Region, ...] = ()
    rois: tuple[Region, ...] = ()
    traffic_lights: tuple[TrafficLight, ...] = ()

    def controlled_lanes(self) -> dict[str, tuple[str, ...]]:
        """Which lanes each light governs, keyed by light id."""
        return {light.id: light.controls for light in self.traffic_lights}


@dataclass(frozen=True)
class Configuration:
    """A site's document: which rules to run, and the places they run against."""

    version: int
    # As written, not as the registry knows them — "rlr_violation", not
    # "red_light_running". Resolving one to the other is the registry's job.
    violations: tuple[str, ...]
    regions: Regions

    @classmethod
    def from_document(cls, document: Any) -> "Configuration":
        """Parse and validate a configuration document.

        Takes the parsed json rather than bytes or a path: whoever fetched it already
        has a dict, and a library that insisted on doing its own IO would be one more
        thing to mock.
        """
        body = _mapping(document, "configuration")

        version = body.get("version")
        if version != SUPPORTED_VERSION:
            raise ConfigurationInvalid(
                f"configuration.version is {version!r}, expected {SUPPORTED_VERSION}"
            )

        return cls(
            version=version,
            violations=_violations(body.get("violations"), "configuration.violations"),
            regions=_regions(body.get("regions"), "configuration.regions"),
        )


def _mapping(value: Any, path: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigurationInvalid(f"{path} is {_described(value)}, expected an object")
    return value


def _sequence(value: Any, path: str) -> list:
    # A list, not merely something iterable. A string would otherwise iterate into its
    # letters — turning a misplaced "rlr_violation" into a pile of one-letter rule
    # names — and a dict into its keys.
    if not isinstance(value, (list, tuple)):
        raise ConfigurationInvalid(f"{path} is {_described(value)}, expected a list")
    return list(value)


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationInvalid(
            f"{path} is {_described(value)}, expected a non-empty string"
        )
    return value


def _described(value: Any) -> str:
    """A value in an error message: what it was, without pasting a whole polygon in."""
    return "missing" if value is None else f"{type(value).__name__} {value!r:.40}"


def _violations(value: Any, path: str) -> tuple[str, ...]:
    names = [_text(name, f"{path}[{i}]") for i, name in enumerate(_sequence(value, path))]
    if not names:
        raise ConfigurationInvalid(f"{path} is empty, expected at least one violation")
    # Refused rather than deduplicated. The same rule twice is a mistake in the
    # document, and the two copies would each keep their own caches and report the
    # same vehicle twice.
    duplicates = _duplicates(names)
    if duplicates:
        raise ConfigurationInvalid(f"{path} names {duplicates[0]!r} more than once")
    return tuple(names)


def _points(value: Any, path: str) -> np.ndarray:
    points = _sequence(value, path)
    if len(points) < MINIMUM_POINTS:
        raise ConfigurationInvalid(
            f"{path} has {len(points)} points, at least {MINIMUM_POINTS} are needed to "
            "enclose an area"
        )

    try:
        array = np.asarray(points, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ConfigurationInvalid(f"{path} is not numeric: {error}") from error

    if array.ndim != 2 or array.shape[1] != 2:
        raise ConfigurationInvalid(
            f"{path} has shape {array.shape}, expected (N, 2) — a list of [x, y] pairs"
        )
    if not np.isfinite(array).all():
        raise ConfigurationInvalid(f"{path} contains a coordinate that is not a number")
    return array


def _region(value: Any, path: str) -> tuple[str, np.ndarray]:
    body = _mapping(value, path)
    return _text(body.get("id"), f"{path}.id"), _points(body.get("points"), f"{path}.points")


def _regions(value: Any, path: str) -> Regions:
    body = _mapping(value, path)

    unknown = sorted(set(body) - set(SECTIONS))
    if unknown:
        raise ConfigurationInvalid(
            f"{path} has unknown section {unknown[0]!r}, expected one of "
            f"{', '.join(SECTIONS)}"
        )

    lanes = _plain_regions(body.get("lanes", []), f"{path}.lanes")
    rois = _plain_regions(body.get("rois", []), f"{path}.rois")
    lights = _traffic_lights(
        body.get("traffic_lights", []),
        f"{path}.traffic_lights",
        lanes={lane.id for lane in lanes},
    )
    return Regions(lanes=lanes, rois=rois, traffic_lights=lights)


def _plain_regions(value: Any, path: str) -> tuple[Region, ...]:
    regions = tuple(
        Region(*_region(entry, f"{path}[{i}]"))
        for i, entry in enumerate(_sequence(value, path))
    )
    _reject_duplicate_ids(regions, path)
    return regions


def _traffic_lights(value: Any, path: str, lanes: set[str]) -> tuple[TrafficLight, ...]:
    lights = []
    for i, entry in enumerate(_sequence(value, path)):
        light_path = f"{path}[{i}]"
        body = _mapping(entry, light_path)
        id = _text(body.get("id"), f"{light_path}.id")
        points = _points(body.get("points"), f"{light_path}.points")
        controls = tuple(
            _text(lane, f"{light_path}.controls[{j}]")
            for j, lane in enumerate(
                _sequence(body.get("controls", []), f"{light_path}.controls")
            )
        )
        # Checked here because here is the only place that can: a rule is handed the
        # mapping and has no way to tell a lane that was never declared from one it
        # simply has not seen a vehicle in yet. Left unchecked, a light controlling
        # "lane_l" would never fire and never explain itself.
        for lane in controls:
            if lane not in lanes:
                raise ConfigurationInvalid(
                    f"{light_path}.controls names {lane!r}, which is not a declared lane"
                )
        lights.append(TrafficLight(id=id, points=points, controls=controls))

    lights = tuple(lights)
    _reject_duplicate_ids(lights, path)
    return lights


def _reject_duplicate_ids(regions: tuple[Region, ...], path: str) -> None:
    # Within a section, not across the document: "roi_1" and "lane_1" are looked up in
    # separate lists, and nothing is served by forbidding a site from numbering both
    # from one.
    duplicates = _duplicates([region.id for region in regions])
    if duplicates:
        raise ConfigurationInvalid(f"{path} has more than one region with id {duplicates[0]!r}")


def _duplicates(names: list[str]) -> list[str]:
    return sorted({name for name in names if names.count(name) > 1})
