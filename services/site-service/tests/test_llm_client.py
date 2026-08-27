"""The HTTP round trip to llm-service, and how its failures come back.

httpx is driven through a MockTransport rather than a live server: what these check
is the request site-service builds and the meaning it assigns to a status code, and
neither needs a socket.
"""

from datetime import datetime, timezone

import httpx
import pytest
from shared import config
from shared.models.detection import ViolationType
from shared.models.explanation import ExplainRequest

from site_service.llm import Explainer, ExplainerRefused, ExplainerUnavailable

REQUEST = ExplainRequest(
    violation_type=ViolationType.RED_LIGHT_RUNNING,
    detected_at=datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc),
    site_name="Junction 5",
    frame_index=159,
)

ANSWER = {
    "explanation": {
        "flag_sustained": True,
        "explanation": "Entered against a red signal.",
        "severity": "MEDIUM",
        "severity_basis": [],
        "observations": [],
        "evidence_concerns": [],
        "confidence": 0.6,
    },
    "model": "fake-model-1",
}


def _explainer(handler) -> Explainer:
    return Explainer(
        url="http://llm-service:8002",
        token="t0ken",
        timeout=5,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


@pytest.fixture
def calls():
    return []


def test_the_call_goes_to_the_versioned_path_with_the_token(calls):
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=ANSWER)

    _explainer(handler).explain(REQUEST)

    assert calls[0].url.path == f"{config.API_V1_PREFIX}/explain"
    # Without this header llm-service answers 401 — it publishes no port, and this is
    # the other half of who is allowed to spend the key behind it.
    assert calls[0].headers["X-LLM-Token"] == "t0ken"


def test_the_answer_comes_back_as_the_model():
    response = _explainer(lambda request: httpx.Response(200, json=ANSWER)).explain(REQUEST)

    assert response.explanation.explanation == "Entered against a red signal."
    assert response.model == "fake-model-1"


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_an_outage_is_worth_retrying(status):
    with pytest.raises(ExplainerUnavailable):
        _explainer(lambda request: httpx.Response(status, json={"detail": "x"})).explain(REQUEST)


@pytest.mark.parametrize("status", [400, 401, 422])
def test_a_rejected_request_is_not(status):
    # 401 in particular: the two sides disagree about LLM_SERVICE_TOKEN, which is a
    # deployment fault and will not fix itself on a retry.
    with pytest.raises(ExplainerRefused):
        _explainer(lambda request: httpx.Response(status, json={"detail": "x"})).explain(REQUEST)


def test_a_service_that_cannot_be_reached_is_worth_retrying():
    def handler(request):
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(ExplainerUnavailable):
        _explainer(handler).explain(REQUEST)
