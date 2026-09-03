import io
import json
import logging
import logging.config
import sys
import uuid

import pytest

from shared.logging import JSONFormatter, configure_logging


def _format(record: logging.LogRecord, service: str = "test-service") -> dict:
    return json.loads(JSONFormatter(service).format(record))


def _record(**kwargs) -> logging.LogRecord:
    return logging.LogRecord(
        name="shared.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="something happened",
        args=(),
        exc_info=None,
        **kwargs,
    )


def test_formats_the_standard_fields_as_json():
    payload = _format(_record())

    assert payload["message"] == "something happened"
    assert payload["severity"] == "INFO"
    assert payload["logger"] == "shared.test"
    assert "timestamp" in payload


def test_uses_the_service_name_it_was_given_not_a_hardcoded_one():
    assert _format(_record(), service="site-service")["service"] == "site-service"
    assert _format(_record(), service="llm-service")["service"] == "llm-service"


def test_extra_fields_appear_at_the_top_level():
    # extra={"job_id": ...} attaches job_id directly onto the record — there is no
    # record.extra — which is what this formatter has to account for. Simulate what
    # Logger.makeRecord does with extra= rather than going through a logger.
    record = _record()
    record.job_id = "job-1"
    record.site_id = "site-1"

    payload = _format(record)

    assert payload["job_id"] == "job-1"
    assert payload["site_id"] == "site-1"


def test_exception_info_is_formatted_into_the_payload():
    try:
        raise ValueError("boom")
    except ValueError:
        record = _record()
        record.exc_info = sys.exc_info()

    payload = _format(record)

    assert "ValueError: boom" in payload["exception"]


def test_a_non_json_serialisable_extra_value_does_not_raise():
    record = _record()
    record.job_id = uuid.uuid4()

    payload = _format(record)

    assert payload["job_id"] == str(record.job_id)


# The shape uvicorn's own LOGGING_CONFIG leaves behind: its three loggers each holding a
# handler, the two it configures with handlers kept off the root logger. Rebuilt here
# rather than imported from uvicorn.config, because uvicorn is a service dependency and
# shared/ is tested without one — the behaviour under test is a logger with a handler
# and propagate=False, whoever put it there.
UVICORN_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"default": {"class": "logging.StreamHandler"}},
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
    },
}

SERVER_LOGGERS = ["uvicorn", "uvicorn.error", "uvicorn.access"]


@pytest.fixture
def logging_state():
    """Undo what configure_logging does to the process, for the tests that call it."""
    root = logging.getLogger()
    saved_root = (root.handlers[:], root.level)
    saved = [(logging.getLogger(n), logging.getLogger(n).handlers[:],
              logging.getLogger(n).propagate, logging.getLogger(n).level)
             for n in SERVER_LOGGERS]
    yield
    root.handlers, root.level = saved_root
    for logger, handlers, propagate, level in saved:
        logger.handlers, logger.propagate, logger.level = handlers, propagate, level


@pytest.fixture
def configured(logging_state):
    """Configure logging the way a service does, after uvicorn already has.

    The order is the point: uvicorn builds its Config — and dictConfigs its loggers —
    before it imports the app module, so configure_logging always runs second and has
    to undo what it finds.
    """
    logging.config.dictConfig(UVICORN_LOGGING_CONFIG)
    stream = io.StringIO()

    def configure(level: str = "INFO") -> io.StringIO:
        configure_logging("site-service", level)
        # Point the installed handler somewhere readable: it holds sys.stdout from
        # before capsys, and the formatter is what these tests are about, not the fd.
        logging.getLogger().handlers[0].setStream(stream)
        return stream

    return configure


def _lines(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_the_servers_own_lines_are_formatted_like_everything_else(configured):
    # Without this the access log — the only record of what a caller actually asked for
    # — stays plain text on uvicorn's own handler, in a stream of JSON.
    stream = configured()

    logging.getLogger("uvicorn.access").info('%s - "%s %s" %d', "127.0.0.1", "GET", "/x", 200)
    logging.getLogger("uvicorn.error").info("Application startup complete.")

    assert [line["message"] for line in _lines(stream)] == [
        '127.0.0.1 - "GET /x" 200',
        "Application startup complete.",
    ]
    # And they arrive attributed to the service, like any other line.
    assert {line["service"] for line in _lines(stream)} == {"site-service"}


def test_the_servers_lines_are_not_also_written_by_its_own_handler(configured):
    configured()

    assert all(logging.getLogger(name).handlers == [] for name in SERVER_LOGGERS)
    assert all(logging.getLogger(name).propagate for name in SERVER_LOGGERS)


def test_log_level_governs_the_servers_lines_too(configured):
    # uvicorn pins these loggers at INFO, and propagation never consults an ancestor's
    # level — so left alone they would go on emitting access lines under LOG_LEVEL=WARNING.
    stream = configured("WARNING")

    logging.getLogger("uvicorn.access").info('%s - "%s %s" %d', "127.0.0.1", "GET", "/x", 200)
    logging.getLogger("uvicorn.error").warning("Invalid HTTP request received.")

    assert [line["message"] for line in _lines(stream)] == ["Invalid HTTP request received."]
