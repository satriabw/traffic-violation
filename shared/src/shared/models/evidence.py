"""The evidence job — the one thing detection-worker and evidence-worker both know.

Like DetectionJob it travels as json over a queue, so this module is a contract between
two processes. It carries two fields, and the reason it is exactly two is worth stating,
because the two arrive for opposite reasons.

`violation_id` is a pointer, and everything reachable from it is deliberately left
behind. DetectionJob pins its source and document versions into the message because the
alternative is asking the database what is *active now* — a job enqueued against v3 and
consumed after someone attached v4 would read a different video than the one it was
created for. There is no equivalent hazard here: a violation row does not have versions
and is never rewritten, so its source, frame index and site read the same late as early.
Carrying them anyway would also mean carrying the window, which runs to ~13.5KB per
track — not a queue payload.

`evidence_seconds` is the exception, and it is here because it is the one thing the
worker needs that the row cannot answer. It lives in the site's configuration document
in object storage, so the alternatives are an S3 fetch per violation or this. The
detection worker resolved it already, on the job that produced the violation — see
detection_worker.context.
"""

from pydantic import BaseModel, Field


class EvidenceJob(BaseModel):
    """Cut the thumbnail and the clip for one violation.

    One job per violation rather than one per detection job, which means several seeks
    into the same video when a chunk produced several violations. That is the trade
    taken deliberately: a violation is the unit that can fail, be retried, and be
    reported on, and batching them would make one bad cut poison the rest.
    """

    violation_id: str
    # HOW MUCH LEAD-UP THE CLIP CARRIES, and it is the same number the detector sized
    # its ring buffer with — which is the whole point of passing it rather than picking
    # one here. The record holds boxes for exactly this span, so a clip cut to any other
    # length is either missing footage the record describes or showing footage it does
    # not. A reviewer drawing the blob's boxes over the clip would see them start late
    # or run out early, and either one reads as a detector that cannot keep time.
    #
    # A junction is the thing that knows how much approach is worth watching. That is
    # why it is a site's configuration and not a constant anywhere — including here.
    evidence_seconds: float = Field(gt=0)
