import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from shared.models.violation import ViolationListResponse, ViolationResponse

from site_service import service
from site_service.db import get_db
from site_service.llm import Explainer, ExplainerRefused, ExplainerUnavailable, get_explainer
from site_service.routers.source import require_site
from site_service.storage import Storage

# Nested under a site like sources and detection, and here there is nothing else it
# could be: the site is the only input, because the setup to filter on is resolved
# rather than asked for.
router = APIRouter(prefix="/sites/{site_id}/violations", tags=["violations"])
DbConnection = Annotated[sqlite3.Connection, Depends(get_db)]
# Injected like the database and the storage client, so a test substitutes a fake
# through dependency_overrides and no test can reach llm-service or spend a key.
ExplainerDep = Annotated[Explainer, Depends(get_explainer)]


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


@router.post("/{violation_id}/explain", response_model=ViolationResponse)
def explain_violation(
    site_id: str,
    violation_id: str,
    con: DbConnection,
    storage: Storage,
    explainer: ExplainerDep,
):
    """Explain one violation, or hand back the explanation it already has.

    A POST because the first one writes and spends money. Every one after it is a
    database read — the explanation is stored on the row and the evidence it was formed
    from does not change, so there is nothing for a second call to improve. That makes
    this safe to call on every page load, which is the point of it not being a GET: a
    GET that did the same would spend the first call on a browser's prefetch.

    Returns the same shape the detail endpoint does, so a client can POST once and
    render the result without a second request.

    503 and 502 come straight from llm-service's own split and mean the same things
    here: 503 is worth retrying, 502 is not.
    """
    require_site(con, site_id)
    try:
        violation = service.explain_violation(con, storage, explainer, site_id, violation_id)
    except ExplainerUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ExplainerRefused as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    if violation is None:
        raise HTTPException(status_code=404, detail="Violation not found")
    return violation
