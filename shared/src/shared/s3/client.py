"""S3 helper, used against Cloudflare R2.

Only presigning and existence checks live here: file bytes never pass through any
service, so there is deliberately no upload() or download() call to reach for.

One bounded exception exists outside this module. shared.video.probe hands a
presigned URL to ffprobe, which range-requests a video's container header — a couple
of megabytes whatever the object's size — to read its metadata once at source
creation. That is a header read, not a transfer, and it is still no reason to add a
download() here.
"""

import json
from functools import lru_cache

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from shared import config


@lru_cache(maxsize=1)
def get_client():
    """The process-wide S3 client. Cached because boto3 client construction parses
    bundled JSON service models and costs far more than the calls we make on it."""
    return boto3.client(
        "s3",
        endpoint_url=config.S3_ENDPOINT_URL,
        aws_access_key_id=config.S3_ACCESS_KEY_ID,
        aws_secret_access_key=config.S3_SECRET_ACCESS_KEY,
        region_name=config.S3_REGION,
        config=Config(
            signature_version="s3v4",
            # boto3 >= 1.36 attaches a CRC32 checksum header to every PUT by
            # default. R2 rejects it on presigned uploads and the client only sees
            # an opaque 400, so limit checksums to operations that require them.
            request_checksum_calculation="when_required",
            response_checksum_validation="when_supported",
        ),
    )


def reset_client() -> None:
    """Drop the cached client so a later call rebuilds it from current config.
    Tests need this because config is read once at import."""
    get_client.cache_clear()


def _expiry(expires_in: int | None) -> int:
    return config.S3_PRESIGN_EXPIRY_SECONDS if expires_in is None else expires_in


def presigned_put(
    key: str,
    content_type: str | None = None,
    content_length: int | None = None,
    expires_in: int | None = None,
) -> str:
    """A URL the client can PUT bytes to directly.

    Anything passed here becomes part of the signature, so the client MUST send the
    identical header or the upload fails. content_length is what stops a caller from
    declaring a small file and then streaming an unbounded one.
    """
    params = {"Bucket": config.S3_BUCKET, "Key": key}
    if content_type:
        params["ContentType"] = content_type
    if content_length is not None:
        params["ContentLength"] = content_length
    return get_client().generate_presigned_url(
        "put_object", Params=params, ExpiresIn=_expiry(expires_in)
    )


def presigned_get(key: str, expires_in: int | None = None) -> str:
    return get_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": config.S3_BUCKET, "Key": key},
        ExpiresIn=_expiry(expires_in),
    )


def public_url(key: str) -> str | None:
    """The unauthenticated URL for a key, or None when the bucket has no public
    read path configured."""
    if not config.S3_PUBLIC_BASE_URL:
        return None
    return f"{config.S3_PUBLIC_BASE_URL}/{key.lstrip('/')}"


def download_url(key: str, expires_in: int | None = None) -> str:
    """Prefer the public URL when one is configured — it is stable and cacheable —
    and fall back to a presigned URL, which always works but expires."""
    return public_url(key) or presigned_get(key, expires_in)


def get_json(key: str) -> dict:
    """Read a small JSON document out of the bucket.

    The second bounded exception to "file bytes never pass through a service", and it
    is the same kind as ffprobe's: a calibration or a configuration is a few kilobytes
    of settings the worker has to *evaluate*, not media it is moving on someone's
    behalf. Handing it a presigned URL and an HTTP client would be the same transfer
    with more moving parts.

    Deliberately not a general download(). If a future caller wants this for a video,
    that is the signal to reach for a stream rather than widen this.
    """
    body = get_client().get_object(Bucket=config.S3_BUCKET, Key=key)["Body"]
    try:
        return json.loads(body.read())
    finally:
        body.close()


def head(key: str) -> dict | None:
    """Object metadata, or None if it does not exist.

    This is how a service confirms a client-direct upload actually landed; nothing
    else in the flow proves it.
    """
    try:
        return get_client().head_object(Bucket=config.S3_BUCKET, Key=key)
    except ClientError as exc:
        # R2 answers a missing key on HEAD with a bare 404 and no error code, so
        # matching on the status is more reliable than on Error/Code here.
        if exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
            return None
        raise
