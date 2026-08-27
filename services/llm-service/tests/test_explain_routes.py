import anthropic
import httpx
import pytest
from shared.models.violation import Severity

TOKEN = "test-token"

REQUEST = {
    "violation_type": "red_light_running",
    "detected_at": "2026-08-21T10:00:00Z",
    "site_name": "Junction 5",
    "frame_index": 159,
    "calibration_id": None,
    "configuration": {"violations": ["rlr_violation"]},
    "metadata": {"vehicles": [{"track_id": 19, "frame_idxs": [45, 159]}], "violator_track_id": 19},
}


def _post(client, body=None, token=TOKEN):
    headers = {"X-LLM-Token": token} if token is not None else {}
    return client.post("/api/v1/explain", json=body or REQUEST, headers=headers)


def test_explaining_a_violation_returns_the_provider_s_answer(client):
    response = _post(client)

    assert response.status_code == 200
    body = response.json()
    assert body["explanation"]["severity"] == Severity.MEDIUM.value
    assert body["explanation"]["explanation"] == "Entered against a red signal."
    # The model is reported alongside, so a stored explanation can be traced to what
    # produced it rather than to the setting that asked for it.
    assert body["model"] == "fake-model-1"


def test_the_service_refuses_a_caller_with_no_token(client):
    assert _post(client, token=None).status_code == 401


def test_the_service_refuses_a_caller_with_the_wrong_token(client):
    assert _post(client, token="not-the-token").status_code == 401


def test_a_refused_caller_never_reaches_the_provider(client, provider):
    # The guard is the point of this service being unreachable, so it has to run
    # before anything that spends money.
    _post(client, token="not-the-token")

    assert provider.calls == []


def test_a_violation_with_no_calibration_is_described_without_its_speeds(client, provider):
    _post(client)

    _, prompt = provider.calls[0]
    assert "WITHHELD" in prompt
    # The ban is on the conclusion rather than on a list of fields, because there is
    # always another field carrying the same information.
    assert "ban on the conclusion" in prompt
    assert "bounding boxes included" in prompt


def test_a_violation_judged_under_a_calibration_keeps_its_speeds(client, provider):
    _post(client, {**REQUEST, "calibration_id": "cal-1"})

    _, prompt = provider.calls[0]
    assert "WITHHELD" not in prompt
    assert "cal-1" in prompt


def test_the_prompt_says_there_are_no_images(client, provider):
    # Without this the model reports what the signal displayed and who was in the
    # crosswalk, neither of which it can possibly know from a track record.
    _post(client)

    _, prompt = provider.calls[0]
    assert "There is no imagery" in prompt


def test_the_prompt_carries_the_scene_the_detector_recorded(client, provider):
    _post(client)

    _, prompt = provider.calls[0]
    assert "convicted track 19" in prompt
    assert "Vehicles on the scene: 1" in prompt
    assert "Pedestrians on the scene: 0" in prompt


@pytest.mark.parametrize(
    "error, expected",
    [
        (
            anthropic.RateLimitError(
                "rate limited",
                response=httpx.Response(429, request=httpx.Request("POST", "https://x")),
                body=None,
            ),
            503,
        ),
        (anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")), 503),
        (
            anthropic.InternalServerError(
                "boom",
                response=httpx.Response(500, request=httpx.Request("POST", "https://x")),
                body=None,
            ),
            503,
        ),
        (
            anthropic.BadRequestError(
                "bad",
                response=httpx.Response(400, request=httpx.Request("POST", "https://x")),
                body=None,
            ),
            502,
        ),
    ],
)
def test_provider_failures_split_into_retryable_and_not(client_for, error, expected):
    # 503 says "this will work later", 502 says "it will not". A caller that cannot
    # tell them apart either retries forever or gives up on an outage.
    assert _post(client_for(error)).status_code == expected


def test_health_needs_no_token(client):
    # compose's healthcheck has no token, and a liveness probe that fails for
    # authorisation reasons reports the wrong thing.
    assert client.get("/health").status_code == 200
