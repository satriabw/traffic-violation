"""Video probing dependency.

Injected the same way get_storage is, so tests substitute a fake through
app.dependency_overrides — and so no test can accidentally shell out to ffprobe or
reach R2.
"""

from typing import Annotated, Callable

from fastapi import Depends

from shared.models.source import SourceMetadata
from shared.video.probe import probe


def get_probe():
    return probe


Probe = Annotated[Callable[[str], SourceMetadata], Depends(get_probe)]
