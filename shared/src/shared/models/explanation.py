"""What an explainer is given to work with.

The request llm-service accepts, and the one site-service sends. Here rather than in
either of them because it is the contract between the two, and a copy on each side is
a copy free to drift.

WHAT IS NOT IN IT: the violation's id, the site's id, anything that identifies a row.
llm-service does not read the database and does not write to it — it is handed a
description of one event and returns an account of it. Keeping the identifiers out is
what makes that true rather than merely intended.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from shared.models.detection import ViolationType
from shared.models.violation import (
    EvidenceStatus,
    ViolationExplanation,
    ViolationMetadata,
)


class ExplainRequest(BaseModel):
    violation_type: ViolationType
    # When the event happened, as the detector recorded it. Not when anybody asked for
    # an explanation.
    detected_at: datetime
    site_name: str
    frame_index: int | None = None

    # WHAT TURNS FRAMES INTO SECONDS, and the reason the explanation can avoid naming a
    # frame at all. The reader is a clerk with a video player, so "2.6 seconds in" is the
    # useful form and "frame 159" is an implementation detail leaking into a document
    # about a person.
    #
    # A property of the footage rather than an identifier, which is what keeps it on the
    # right side of the boundary this model draws. None on a source that was never
    # probed, or whose probe found no frame rate — the prompt then places the event
    # without a number rather than falling back to the frame index, because a number
    # nobody can interpret is worse than no number.
    fps: float | None = None

    # WHETHER THERE IS FOOTAGE TO GO AND LOOK AT. Only three answers are possible about a
    # plate here — pull the clip and read it, put it through recognition, or nothing
    # settles it — and two of the three are claims about evidence that may not exist.
    # Without this the explainer cannot tell a violation whose clip is sitting in storage
    # from one that was never cut, so it can only ever say "inconclusive", which makes
    # the field decorative.
    #
    # A fact about the violation, not a pointer to a row, so it does not cross the line
    # this model exists to hold. None means nothing was ever queued to build evidence for
    # it — see EvidenceStatus, where the same None means the same thing.
    evidence_status: EvidenceStatus | None = None

    # WHETHER THE MOTION DATA MEANS ANYTHING. None says the violation was recorded with
    # no camera calibration pinned, so there is no valid pixel-to-world mapping and
    # every distance and speed derived through it is a number without units anybody can
    # trust. The prompt withholds them in that case; see llm_service.prompt.
    #
    # It is the id rather than a boolean because the caller has the id and turning it
    # into a flag here would throw away which calibration, for a reader that later
    # wants to know.
    calibration_id: str | None = None

    # The site configuration in force when this was judged — lane polygons, traffic
    # light regions, which rules were enabled. Opaque here: it is somebody's JSON
    # document and this model does not own its shape.
    configuration: dict[str, Any] | None = None

    # Every track the detector held when the rule fired, and which one it convicted.
    # None on a violation whose blob is missing.
    metadata: ViolationMetadata | None = None


class ExplainResponse(BaseModel):
    """Thin, and deliberately not just the explanation itself.

    A wrapper leaves somewhere to put facts about the call rather than about the
    violation — which provider answered, which model — without those leaking into
    ViolationExplanation, which is a statement about the event and is what gets stored.
    """

    explanation: ViolationExplanation
    model: str = Field(description="The model that produced it, as reported by the provider.")
