"""llm-service — one violation in, one explanation out.

NO DATABASE, NO OBJECT STORAGE, NO QUEUE. It is handed a description of an event and
returns an account of it. That is what keeps a second provider cheap: everything this
service knows how to do lives in llm_service.providers, and nothing else in the system
has to move when that changes.

Storing the answer is site-service's job, and so is deciding whether to ask at all —
this service will happily explain the same violation twice, because it has no way to
know it is the same one and no business having one.
"""

import logging
from contextlib import asynccontextmanager
from typing import Annotated

import anthropic
from fastapi import Depends, FastAPI, Header, HTTPException
from shared import config
from shared.config import API_V1_PREFIX
from shared.logging import configure_logging
from shared.models.explanation import ExplainRequest, ExplainResponse

from llm_service import prompt as prompt_builder
from llm_service.providers import Provider, get_provider

# The name this service is known by, in the logs and in its own OpenAPI document.
SERVICE_NAME = "llm-service"

configure_logging(SERVICE_NAME, config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# One constant per provider failure branch, rather than the message written inline at
# the call site. A log message is read by whoever is on call — it is what a saved
# search or an alert on "the explanation provider is down" matches — so it is defined
# once here, and the tests assert on these constants rather than on their text.
# Rewording one is then a change to a name with known readers, not a string edit that
# quietly stops matching.
PROVIDER_RATE_LIMITED = "explanation provider rate limited"
PROVIDER_UNREACHABLE = "cannot reach explanation provider"
PROVIDER_UNAVAILABLE = "explanation provider unavailable"
PROVIDER_REJECTED = "explanation provider rejected the request"

# Built once at startup rather than per request: constructing a provider means
# constructing an HTTP client, and doing that per violation would throw away
# connection reuse for no gain.
_provider: Provider | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Checked before the port is bound, the same way site-service checks its object
    # storage. A missing key should stop the service on the way up with the name of
    # what is missing — not surface as a 401 from the provider on the first violation
    # somebody asks about.
    missing = config.missing_llm_settings()
    if missing:
        raise RuntimeError(
            f"The explanation service is not configured: {', '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} empty. Load the environment "
            "before starting, e.g. `set -a; source .env; set +a`."
        )
    global _provider
    _provider = get_provider()
    yield


def require_caller(x_llm_token: Annotated[str | None, Header()] = None) -> None:
    """This service answers site-service and nothing else.

    Two mechanisms, because neither is sufficient alone. The port is not published, so
    the service is unreachable from the host — that is the real boundary, and it is in
    docker-compose rather than here. This is the other half: without it, anything that
    can reach the compose network can spend the API key, and the day somebody publishes
    a port to debug something the only thing left is this.

    Compared with `!=` rather than a constant-time compare, deliberately: the secret is
    a deployment-local token on a private network, and a timing oracle against it needs
    the network access the token exists to gate.
    """
    if x_llm_token != config.LLM_SERVICE_TOKEN:
        raise HTTPException(status_code=401, detail="Not authorised to call this service")


def get_current_provider() -> Provider:
    # Indirection so tests override the dependency rather than reaching for the module
    # global — the same reason site-service injects its storage.
    if _provider is None:
        raise HTTPException(status_code=503, detail="Provider is not ready")
    return _provider


app = FastAPI(title=SERVICE_NAME, lifespan=lifespan)


@app.post(
    f"{API_V1_PREFIX}/explain",
    response_model=ExplainResponse,
    dependencies=[Depends(require_caller)],
)
def explain(
    request: ExplainRequest,
    provider: Annotated[Provider, Depends(get_current_provider)],
):
    """Explain one violation.

    Errors are mapped rather than swallowed, and the split is between what is worth
    retrying and what is not. A rate limit or an outage will succeed later, so it comes
    back as 503 and the caller may try again; a request the provider rejected will be
    rejected identically next time, and 502 says so. Nothing here retries on its own —
    the SDK already does, and a second layer of it would multiply the wait a caller
    is holding a connection open through.
    """
    # violation_type and site_name only, never a violation/site id — see
    # shared.models.explanation.ExplainRequest on why this service is never given one.
    log_context = {"violation_type": request.violation_type, "site_name": request.site_name}
    text = prompt_builder.build_prompt(request)
    try:
        explanation = provider.explain(prompt_builder.SYSTEM, text)
    except anthropic.RateLimitError as error:
        logger.warning(PROVIDER_RATE_LIMITED, exc_info=True, extra=log_context)
        raise HTTPException(status_code=503, detail="Explanation provider is rate limited") from error
    except anthropic.APIConnectionError as error:
        logger.warning(PROVIDER_UNREACHABLE, exc_info=True, extra=log_context)
        raise HTTPException(status_code=503, detail="Cannot reach the explanation provider") from error
    except anthropic.APIStatusError as error:
        if error.status_code >= 500:
            logger.warning(PROVIDER_UNAVAILABLE, exc_info=True, extra=log_context)
            raise HTTPException(status_code=503, detail="Explanation provider is unavailable") from error
        logger.warning(PROVIDER_REJECTED, exc_info=True, extra=log_context)
        raise HTTPException(
            status_code=502, detail=f"Explanation provider rejected the request: {error.message}"
        ) from error
    return ExplainResponse(explanation=explanation, model=provider.model)


@app.get("/health", include_in_schema=False)
def health():
    # Unauthenticated, and it has to be: compose's healthcheck has no token, and a
    # liveness probe that can fail for authorisation reasons is a probe that reports
    # the wrong thing.
    return {"status": "ok"}
