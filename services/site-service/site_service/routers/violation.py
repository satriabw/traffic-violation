import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from shared.models.violation import ViolationListResponse

from site_service import service
from site_service.db import get_db
from site_service.routers.source import require_site
from site_service.storage import Storage

# Nested under a site like sources and detection, and here there is nothing else it
# could be: the site is the only input, because the setup to filter on is resolved
# rather than asked for.
router = APIRouter(prefix="/sites/{site_id}/violations", tags=["violations"])
DbConnection = Annotated[sqlite3.Connection, Depends(get_db)]


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
    track's trajectory, and the boxes belong to whoever draws over the clip in a detail
    view that does not exist yet.
    """
    require_site(con, site_id)
    return service.list_violations(con, storage, site_id, limit=limit, offset=offset)
