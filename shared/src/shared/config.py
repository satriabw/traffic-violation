import os

DB_PATH = os.environ.get("SITE_SERVICE_DB_PATH", "./data/site_service.duckdb")

# Every service mounts its resource routers under this prefix, and callers use the
# same path whether they reach the service through the gateway or directly by its
# container name. The gateway routes on this prefix but must not rewrite it.
API_V1_PREFIX = "/api/v1"


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


# --- Video probing --------------------------------------------------------
# How long ffprobe gets to read a video's header before we give up and call the
# attempt transient. It runs inside a request, so this is also the worst case a
# client waits on source creation.
VIDEO_PROBE_TIMEOUT_SECONDS = float(os.environ.get("VIDEO_PROBE_TIMEOUT_SECONDS", "20"))


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
