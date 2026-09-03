import os

# One SQLite file, shared by every process: site-service writes it, detection-worker
# reads context from it and appends violations. Not named for site-service any more,
# because site-service no longer owns it.
DB_PATH = os.environ.get("TRAFFIC_DB_PATH", "./data/traffic_violations.sqlite")

# Every service mounts its resource routers under this prefix, and callers use the
# same path whether they reach the service through the gateway or directly by its
# container name. The gateway routes on this prefix but must not rewrite it.
API_V1_PREFIX = "/api/v1"

# Passed straight to shared.logging.configure_logging. A name, not a level constant,
# because os.environ hands back a string and logging.Logger.setLevel accepts one.
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")


# --- Object storage -------------------------------------------------------
# Cloudflare R2 in practice, but nothing here is R2-specific: any S3-compatible
# endpoint works, which is what lets tests point at a local fake.
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", "")
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID", "")
S3_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY", "")

# R2 has no regions. SigV4 still requires one in the signature, and "auto" is the
# only value R2 accepts.
S3_REGION = os.environ.get("S3_REGION", "auto")

S3_PRESIGN_EXPIRY_SECONDS = int(os.environ.get("S3_PRESIGN_EXPIRY_SECONDS", "3600"))

# Largest upload a client may request a URL for. Enforced twice: rejected here
# before a URL is ever minted, and signed into that URL so R2 rejects a body of a
# different size. Raise it per environment rather than in code.
S3_MAX_UPLOAD_BYTES = int(os.environ.get("S3_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))

# Empty means "no public read path" — downloads get a presigned URL instead.
S3_PUBLIC_BASE_URL = os.environ.get("S3_PUBLIC_BASE_URL", "").rstrip("/")


# --- Message queue --------------------------------------------------------
# A Redis list carries detection jobs from site-service to detection-worker. Only a
# real run needs it: the test suite drives the queue through an injected fake.
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
DETECTION_QUEUE_NAME = os.environ.get("DETECTION_QUEUE_NAME", "detection:jobs")

# A second list, carrying one job per violation from detection-worker to
# evidence-worker. Its own list rather than a type tag on the first: the two are
# consumed by different processes at wildly different rates — one is a GPU decoding a
# video, the other is ffmpeg cutting a few seconds out of one — and a shared list would
# make each pop a message the popper cannot handle.
EVIDENCE_QUEUE_NAME = os.environ.get("EVIDENCE_QUEUE_NAME", "evidence:jobs")


# --- Detection model ------------------------------------------------------
# A local filesystem path to an ONNX export. Pulling the weights from R2 instead —
# the pseudo-registry — resolves through the same setting; see
# detection_worker.model.resolve_model_path.
DETECTION_MODEL_PATH = os.environ.get("DETECTION_MODEL_PATH", "")

# Minimum confidence for a detection to leave the model at all. Everything below is
# dropped before tracking, so this is the coarsest of the pipeline's filters.
DETECTION_THRESHOLD = float(os.environ.get("DETECTION_THRESHOLD", "0.5"))

# onnxruntime tries these in order and falls back to the next when one is
# unavailable. CPU is the default because it is the only provider guaranteed to
# exist, and a worker that silently refuses to start without a GPU is worse than a
# slow one. Set to "CUDAExecutionProvider,CPUExecutionProvider" where there is a GPU.
DETECTION_MODEL_PROVIDERS = [
    provider.strip()
    for provider in os.environ.get("DETECTION_MODEL_PROVIDERS", "CPUExecutionProvider").split(",")
    if provider.strip()
]


# --- Evidence -------------------------------------------------------------
# NOTHING HERE SIZES A CLIP, deliberately. How much lead-up a violation's clip carries
# is the site's `evidence_seconds`, out of its configuration document, because that is
# the number the detector sized its ring buffer with — the record holds boxes for
# exactly that span, and a clip cut to any other length disagrees with it. A default
# here would be a second source of truth for one number, free to drift from the one
# that produced the evidence. It travels on the job instead; see shared.models.evidence.
#
# How long one ffmpeg invocation gets. Generous: it is a range-read of a remote object
# over somebody's network, not a decode of the whole video. What it is really for is
# the hang — a stalled connection would otherwise park a worker on one violation for
# good, and there is only one of them.
EVIDENCE_CUT_TIMEOUT_SECONDS = float(os.environ.get("EVIDENCE_CUT_TIMEOUT_SECONDS", "120"))


# --- Video probing --------------------------------------------------------
# How long ffprobe gets to read a video's header before we give up and call the
# attempt transient. It runs inside a request, so this is also the worst case a
# client waits on source creation.
VIDEO_PROBE_TIMEOUT_SECONDS = float(os.environ.get("VIDEO_PROBE_TIMEOUT_SECONDS", "20"))


# --- Explanation ----------------------------------------------------------
# llm-service turns one violation into an explanation. Its own service rather than a
# module inside site-service because the provider behind it is going to change and the
# thing that changes should be swappable on its own — and because an API call that can
# take a minute does not belong in the same process as the endpoints that have to stay
# responsive.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Which provider llm-service loads. One value today; the point of the setting is that
# adding a second means a file in llm_service.providers and a new string here, not a
# change to anything that calls it.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "claude")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-opus-5")

# Where site-service finds llm-service. The compose service name, because that is the
# only place it is reachable from: llm-service publishes no port — see docker-compose.
LLM_SERVICE_URL = os.environ.get("LLM_SERVICE_URL", "http://llm-service:8002").rstrip("/")

# A shared secret, sent by site-service and checked by llm-service. Not defence against
# an attacker inside the network so much as a statement of who the caller is meant to
# be: unpublishing the port keeps the service off the host, and this keeps it from
# answering anything else on the compose network that happens to find it.
LLM_SERVICE_TOKEN = os.environ.get("LLM_SERVICE_TOKEN", "")

# How long site-service waits on one explanation. Generous, because the model thinks
# before it answers and the whole call is one round trip that either lands or does not.
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "120"))


def missing_s3_settings() -> list[str]:
    """Names of the object-storage settings that have no value.

    Read at call time rather than import time so a caller sees the current
    environment. S3_REGION and S3_PRESIGN_EXPIRY_SECONDS have working defaults and
    S3_PUBLIC_BASE_URL is genuinely optional, so none of them appear here.
    """
    return [
        name
        for name, value in (
            ("S3_ENDPOINT_URL", S3_ENDPOINT_URL),
            ("S3_BUCKET", S3_BUCKET),
            ("S3_ACCESS_KEY_ID", S3_ACCESS_KEY_ID),
            ("S3_SECRET_ACCESS_KEY", S3_SECRET_ACCESS_KEY),
        )
        if not value
    ]


def missing_llm_settings() -> list[str]:
    """Names of the explanation settings that have no value.

    The same shape as missing_s3_settings, and read by llm-service at startup for the
    same reason: a key that is missing should stop the service on the way up with the
    name of what is missing, not surface as a 401 from the provider on the first
    violation somebody asks about.

    LLM_MODEL and LLM_PROVIDER have working defaults and do not appear. LLM_SERVICE_URL
    is site-service's setting, not this one's.
    """
    return [
        name
        for name, value in (
            ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
            ("LLM_SERVICE_TOKEN", LLM_SERVICE_TOKEN),
        )
        if not value
    ]
