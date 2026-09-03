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


def configure_logging(service: str, level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter(service))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
