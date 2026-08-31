import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from shared.models.violation import ViolationListResponse, ViolationResponse

from site_service import service
from site_service.db import get_db
from site_service.actor import Explain, ExplanationActor, get_actor
from site_service.routers.source import require_site
from site_service.storage import Storage

# Nested under a site like sources and detection, and here there is nothing else it
# could be: the site is the only input, because the setup to filter on is resolved
# rather than asked for.
router = APIRouter(prefix="/sites/{site_id}/violations", tags=["violations"])
DbConnection = Annotated[sqlite3.Connection, Depends(get_db)]
# Injected like the database and the storage client, so a test substitutes a fake
# through dependency_overrides and no test can reach llm-service or spend a key.
ActorDep = Annotated[ExplanationActor, Depends(get_actor)]


@router.get("", response_model=ViolationListResponse)
def list_violations(
    site_id: str,
    con: DbConnection,
    storage: Storage,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """The violations that hold under the setup this site is running now.

    NO FILTER PARAMETERS, and the absence is the design. The calibration and
    configuration are resolved to the site's active versions in the service layer —
    see list_violations — so this cannot be asked for a superseded setup. Re-calibrate
    a site and the list changes without a violation being touched, which is what the
    two id columns on the row exist to make possible.

    Every item carries its thumbnail and clip as signed links, minted per read. None of
    them carries the metadata blob: rendering a page of violations does not need every
    track's trajectory, and the boxes belong to whoever draws over the clip — which is
    the detail endpoint below, and the only reader that pays for them.
    """
    require_site(con, site_id)
    return service.list_violations(con, storage, site_id, limit=limit, offset=offset)


@router.get("/{violation_id}", response_model=ViolationResponse)
def get_violation(site_id: str, violation_id: str, con: DbConnection, storage: Storage):
    """One violation, with the metadata blob the list deliberately does not carry.

    THE READ NEVER EXPLAINS ANYTHING. It returns whatever explanation exists on the row
    and None when there is none — asking for one is a POST, because it spends money and
    writes. A GET that quietly called a model would do both on a browser's prefetch.

    404 covers both "no such violation" and "not this site's violation", and the two are
    deliberately indistinguishable: a different answer for the second would confirm the
    id is real and belongs to somebody else.
    """
    require_site(con, site_id)
    violation = service.get_violation(con, storage, site_id, violation_id)
    if violation is None:
        raise HTTPException(status_code=404, detail="Violation not found")
    return violation


@router.post(
    "/{violation_id}/explain", response_model=ViolationResponse, status_code=202
)
def explain_violation(
    site_id: str,
    violation_id: str,
    con: DbConnection,
    storage: Storage,
    actor: ActorDep,
    response: Response,
):
    """Ask for one violation to be explained, and say what state it is in now.

    202 BECAUSE NOTHING IS FINISHED. The explanation is a call to a model that thinks
    before it answers — tens of seconds — and holding a connection open through it parks
    a request thread and blocks whatever is waiting on the other end. So this accepts the
    work, hands it to the actor, and returns the violation with `status` set to pending.
    The client polls that status, on this violation or in the site's list, which already
    carries it for every row.

    200 WHEN THERE WAS NOTHING TO DO, which is the honest code for a violation that is
    already explained or already pending: no work was accepted, because none was needed.
    A client that only looks at the body cannot tell the difference and does not need to
    — the status says which it is either way.

    STILL A POST, and still safe to call on every page load. The first one spends money
    and every one after it does not: an explained violation is handed back from the row,
    and a pending one is left alone rather than queued twice. A GET that did the same
    would spend the first call on a browser's prefetch.

    ASKING AGAIN IS THE RETRY. A violation whose explanation failed is accepted again
    here; nothing retries on its own.

    NO 503 OR 502 ANY MORE. Those mapped llm-service's own split for a caller that was
    waiting on the answer, and there is no such caller now. A provider that is
    unreachable or refuses lands on the row as `failed`, where the client polling finds
    it alongside everything else it is already watching.
    """
    require_site(con, site_id)
    accepted = service.request_explanation(con, storage, site_id, violation_id)
    if accepted is None:
        raise HTTPException(status_code=404, detail="Violation not found")

    violation, send = accepted
    if not send:
        response.status_code = 200
        return violation
    # After request_explanation, never before: it writes 'pending' first so a fast actor
    # cannot finish and be overwritten by the status that says it has not started.
    actor.send(Explain(site_id=site_id, violation_id=violation_id))
    return violation
