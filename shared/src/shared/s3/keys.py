"""Object key layout, shared so every producer of files agrees on it.

Kept free of boto3 on purpose: the detection worker builds evidence-frame keys with
the same function, and pure string work needs no client.
"""

import posixpath
import re

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

# Long enough for any real filename, short enough to stay clear of key length limits
# once the type and id prefixes are added.
_MAX_NAME = 128


def _safe_name(name: str) -> str:
    # basename first: the client sends a display name, so any directory part in it is
    # either noise or an attempt to escape the id prefix.
    name = posixpath.basename(name.replace("\\", "/")).strip()

    # Split before sanitising so a stem that scrubs away to nothing (a fully
    # non-ASCII filename, say) cannot take the extension down with it — consumers
    # and browsers key content handling off the extension.
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    stem = _UNSAFE.sub("_", stem)
    ext = _UNSAFE.sub("_", ext)

    if not stem.strip("._-"):
        stem = "file"

    budget = _MAX_NAME - (len(ext) + 1 if ext else 0)
    if len(stem) > budget:
        stem = stem[:budget]

    return f"{stem}.{ext}" if ext else stem


def build_key(file_type: str, file_id: str, name: str) -> str:
    """`{type}/{file_id}/{safe_name}` — the id prefix is what guarantees two files
    with the same display name never overwrite each other."""
    return f"{file_type}/{file_id}/{_safe_name(name)}"
