import json
import logging
import sys
import uuid

from shared.logging import JSONFormatter


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
