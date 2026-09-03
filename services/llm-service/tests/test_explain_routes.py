import logging

import anthropic
import httpx
import pytest
from shared.models.violation import Severity

from llm_service import main

TOKEN = "test-token"

REQUEST = {
    "violation_type": "red_light_running",
    "detected_at": "2026-08-21T10:00:00Z",
    "site_name": "Junction 5",
    "frame_index": 159,
    "calibration_id": None,
    "configuration": {"violations": ["rlr_violation"]},
    "fps": 60.0,
    "metadata": {
        "vehicles": [
            {
                "track_id": 19,
                "frame_idxs": [45, 159],
                # Enough box movement to read as traffic rather than as a signal head.
                "bboxes": [[0.0, 0.0, 200.0, 150.0], [400.0, 0.0, 600.0, 150.0]],
            }
        ],
        "violator_track_id": 19,
    },
}


def _post(client, body=None, token=TOKEN):
    headers = {"X-LLM-Token": token} if token is not None else {}
    return client.post("/api/v1/explain", json=body or REQUEST, headers=headers)


def test_explaining_a_violation_returns_the_provider_s_answer(client):
    response = _post(client)

    assert response.status_code == 200
    body = response.json()
    assert body["explanation"]["severity"] == Severity.MEDIUM.value
    assert body["explanation"]["explanation"] == (
        "A vehicle drove into the junction after the signal had turned red."
    )
    # The clerk-facing fields travel with it.
    assert body["explanation"]["evidence_strength"] == "WEAK"
    assert body["explanation"]["license_plate"]["recoverability"] == "inconclusive"
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
    assert "SPEED AND DISTANCE: unusable" in prompt
    assert "no camera calibration set up" in prompt
    # The ban is on the conclusion rather than on a list of fields, because there is
    # always another field carrying the same information.
    assert "ban on the conclusion" in prompt


def test_a_violation_judged_under_a_working_calibration_keeps_its_speeds(client, provider):
    # Plausible speeds under a pinned calibration stay usable. The id itself does not
    # travel: it identifies a row, and the clerk reading this cannot do anything with it.
    _post(client, {**REQUEST, "calibration_id": "cal-1"})

    _, prompt = provider.calls[0]
    assert "SPEED AND DISTANCE: available" in prompt
    assert "cal-1" not in prompt


def test_the_prompt_says_there_are_no_images(client, provider):
    # Without this the model reports what the signal displayed and who was in the
    # crosswalk, neither of which it can possibly know from a track record.
    _post(client)

    _, prompt = provider.calls[0]
    assert "no imagery" in prompt
    assert "must not write as though you can" in prompt


def test_the_prompt_carries_the_scene_without_carrying_its_identifiers(client, provider):
    # The scene reaches the model as counts and roles. The track id and the frame index
    # behind them do not — a clerk cannot act on either, so the model is never given the
    # vocabulary to repeat them.
    _post(client)

    _, prompt = provider.calls[0]
    assert "Nobody on foot was there at that moment." in prompt
    assert "2.6 seconds into the footage" in prompt
    assert "track 19" not in prompt
    assert "159" not in prompt


# (what the provider raised, the status it maps to, the line it logs). One list for
# both tests below, because a failure branch is only right when it does both.
PROVIDER_FAILURES = [
    (
        anthropic.RateLimitError(
            "rate limited",
            response=httpx.Response(429, request=httpx.Request("POST", "https://x")),
            body=None,
        ),
        503,
        main.PROVIDER_RATE_LIMITED,
    ),
    (
        anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")),
        503,
        main.PROVIDER_UNREACHABLE,
    ),
    (
        anthropic.InternalServerError(
            "boom",
            response=httpx.Response(500, request=httpx.Request("POST", "https://x")),
            body=None,
        ),
        503,
        main.PROVIDER_UNAVAILABLE,
    ),
    (
        anthropic.BadRequestError(
            "bad",
            response=httpx.Response(400, request=httpx.Request("POST", "https://x")),
            body=None,
        ),
        502,
        main.PROVIDER_REJECTED,
    ),
]


@pytest.mark.parametrize("error, expected, message", PROVIDER_FAILURES)
def test_provider_failures_split_into_retryable_and_not(client_for, error, expected, message):
    # 503 says "this will work later", 502 says "it will not". A caller that cannot
    # tell them apart either retries forever or gives up on an outage.
    assert _post(client_for(error)).status_code == expected


@pytest.mark.parametrize("error, expected, message", PROVIDER_FAILURES)
def test_a_provider_failure_is_logged_before_it_becomes_a_status(
    client_for, caplog, error, expected, message
):
    # The caller only ever sees the status, so the log line is the only record of which
    # provider failure it was. Asserted against the module's constants rather than
    # against their text: an alert is pinned to the constant, and a reworded message
    # that nothing else has to follow is then not a test failure.
    with caplog.at_level(logging.WARNING, logger="llm_service.main"):
        _post(client_for(error))

    logged = [record for record in caplog.records if record.name == "llm_service.main"]
    assert [record.getMessage() for record in logged] == [message]
    # The context travels as fields, and stops at what this service is allowed to know:
    # no violation or site id, because it is never given one.
    assert logged[0].violation_type == "red_light_running"
    assert logged[0].site_name == "Junction 5"
    # exc_info=True, so what the provider actually raised reaches the line too.
    assert logged[0].exc_info is not None


def test_health_needs_no_token(client):
    # compose's healthcheck has no token, and a liveness probe that fails for
    # authorisation reasons reports the wrong thing.
    assert client.get("/health").status_code == 200
