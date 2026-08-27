"""site-service's handle on llm-service.

A dependency like get_db and get_storage, injected the same way and for the same
reason: a test substitutes a fake through app.dependency_overrides rather than
monkeypatching a module, so no test can reach the network or spend an API key.

Thin on purpose. Everything about how a violation gets explained — the prompt, the
provider, the model — lives on the other side of this call. What is here is the round
trip and what to do when it fails.
"""

import httpx
from shared import config
from shared.models.explanation import ExplainRequest, ExplainResponse


class ExplainerUnavailable(RuntimeError):
    """The explainer could not answer, and trying again later might work.

    Distinct from ExplainerRefused below because the caller does different things with
    them, and the router turns them into different statuses.
    """


class ExplainerRefused(RuntimeError):
    """The explainer rejected the request. Sending it again will be rejected again."""


class Explainer:
    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        timeout: float | None = None,
        client: httpx.Client | None = None,
    ):
        self._url = (url if url is not None else config.LLM_SERVICE_URL).rstrip("/")
        self._token = token if token is not None else config.LLM_SERVICE_TOKEN
        self._timeout = timeout if timeout is not None else config.LLM_TIMEOUT_SECONDS
        # Injectable, so a test drives a MockTransport rather than patching httpx —
        # and so the connection is reused across violations rather than dialled again
        # per request. httpx.Client is thread-safe, which FastAPI's threadpool needs.
        self._client = client or httpx.Client(timeout=self._timeout)

    def explain(self, request: ExplainRequest) -> ExplainResponse:
        """One call, and no retry.

        llm-service does not retry either — the Anthropic SDK already does, and each
        layer added on top multiplies the wall-clock a caller holds a connection open
        through. A failure here comes back to the client, which can ask again.

        The timeout is generous because the model thinks before it answers. It is still
        a timeout: without one a stalled connection parks a request thread for good.
        """
        try:
            response = self._client.post(
                f"{self._url}{config.API_V1_PREFIX}/explain",
                json=request.model_dump(mode="json"),
                headers={"X-LLM-Token": self._token},
                timeout=self._timeout,
            )
        except httpx.RequestError as error:
            raise ExplainerUnavailable(f"Cannot reach the explanation service: {error}") from error

        if response.status_code >= 500 or response.status_code == 503:
            raise ExplainerUnavailable(
                f"The explanation service is unavailable ({response.status_code})"
            )
        if response.status_code >= 400:
            # 401 lands here, and it is worth naming: it means the two sides disagree
            # about LLM_SERVICE_TOKEN, which is a deployment fault rather than anything
            # about this violation, and it will not fix itself on a retry.
            raise ExplainerRefused(
                f"The explanation service refused the request ({response.status_code})"
            )
        return ExplainResponse.model_validate(response.json())


_explainer: Explainer | None = None


def get_explainer() -> Explainer:
    """One Explainer for the process, so its connection pool is one pool.

    Built lazily rather than at import so that reading the configuration happens when
    the app runs, not when a test imports the module — the same reason
    missing_s3_settings is a function.
    """
    global _explainer
    if _explainer is None:
        _explainer = Explainer()
    return _explainer
