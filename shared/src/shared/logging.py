"""One JSON line per log record, to stdout.

Every service that has this on its path gets the same shape: a timestamp, the record's
own severity and logger name, the message, and which service emitted it — plus
whatever structured context the call site attached with `extra=`. That last part is
the actual reason this exists rather than `logging.basicConfig`: `extra={"job_id": x}`
attaches `job_id` directly onto the LogRecord, not under a nested `.extra` — there is
no such attribute — so pulling it back out means diffing against what a bare record
already carries.

Nothing here assumes where the line ends up. JSON on stdout is what every option reads
without extra plumbing: GKE's node-level Cloud Logging agent auto-parses it, and it is
exactly the input Filebeat/Fluentd expect if this ends up shipped into a self-run
ELK/EFK stack instead.

The one thing this has to do beyond installing a formatter is take the server's own
loggers off their private handlers — see _SERVER_LOGGERS.
"""

import json
import logging
import sys
from datetime import datetime, timezone

# Attribute names already on a bare LogRecord. Diffed against record.__dict__ below to
# find caller-supplied extra= fields.
_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message"}


class JSONFormatter(logging.Formatter):
    def __init__(self, service: str):
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # Not "level": this is also the field GCP Cloud Logging reads to power
            # its own severity filtering, for free, if that is where the line lands.
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self._service,
        }
        payload.update((k, v) for k, v in record.__dict__.items() if k not in _RESERVED)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # default=str: an extra field that is not JSON-serializable (a UUID, a Path)
        # must not crash the formatter and lose the line — see Handler.handleError.
        return json.dumps(payload, default=str)


# The loggers that arrive already holding handlers of their own. uvicorn dictConfigs
# these when its Config is built, which is before it imports the app module — so by the
# time configure_logging runs they have a text formatter, a stream, and propagate=False,
# and the request log is the one part of a service's output this formatter never sees.
# Handing them back to the root logger is what makes a service's output one stream in
# one shape, access lines included.
#
# Named explicitly rather than found by walking logging.root.manager.loggerDict: this
# is uvicorn's published logging config, and a sweep over every configured logger would
# also strip handlers from a library that means to have its own.
_SERVER_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


def configure_logging(service: str, level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter(service))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    for name in _SERVER_LOGGERS:
        server_logger = logging.getLogger(name)
        server_logger.handlers = []
        server_logger.propagate = True
        # NOTSET, not `level`: propagation consults a handler's level and the emitting
        # logger's, never an ancestor's, so a logger left at uvicorn's own INFO would
        # keep emitting access lines through the root handler no matter what LOG_LEVEL
        # said. Cleared, it inherits — and LOG_LEVEL is the only knob again.
        server_logger.setLevel(logging.NOTSET)
