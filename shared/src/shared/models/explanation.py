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
from shared.models.violation import ViolationExplanation, ViolationMetadata


class ExplainRequest(BaseModel):
    violation_type: ViolationType
    # When the event happened, as the detector recorded it. Not when anybody asked for
    # an explanation.
    detected_at: datetime
    site_name: str
    frame_index: int | None = None

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
