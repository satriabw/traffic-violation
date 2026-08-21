from urllib.parse import parse_qs, urlparse

import pytest

from shared import config
from shared.s3 import client


@pytest.fixture(autouse=True)
def s3_config(monkeypatch):
    """Presigning is pure local computation — no network, no real account — so
    fixed fake credentials are enough to assert on the generated URLs."""
    monkeypatch.setattr(config, "S3_ENDPOINT_URL", "https://acct123.r2.cloudflarestorage.com")
    monkeypatch.setattr(config, "S3_BUCKET", "test-bucket")
    monkeypatch.setattr(config, "S3_ACCESS_KEY_ID", "AKIATEST")
    monkeypatch.setattr(config, "S3_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setattr(config, "S3_REGION", "auto")
    monkeypatch.setattr(config, "S3_PRESIGN_EXPIRY_SECONDS", 3600)
    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "")
    client.reset_client()
    yield
    client.reset_client()


def test_presigned_put_targets_the_bucket_and_key():
    url = client.presigned_put("video/abc/clip.mp4")
    parsed = urlparse(url)
    assert parsed.hostname == "acct123.r2.cloudflarestorage.com"
    assert parsed.path == "/test-bucket/video/abc/clip.mp4"


def test_presigned_put_is_signed_with_sigv4_for_region_auto():
    query = parse_qs(urlparse(client.presigned_put("k")).query)
    assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
    # R2 only accepts "auto" in the credential scope; a stray real region here is
    # the usual cause of SignatureDoesNotMatch.
    assert "/auto/s3/aws4_request" in query["X-Amz-Credential"][0]


def test_presigned_put_expiry_defaults_to_config_and_is_overridable():
    assert parse_qs(urlparse(client.presigned_put("k")).query)["X-Amz-Expires"] == ["3600"]
    url = client.presigned_put("k", expires_in=60)
    assert parse_qs(urlparse(url).query)["X-Amz-Expires"] == ["60"]


def test_content_type_is_part_of_the_signature_when_given():
    signed_headers = parse_qs(urlparse(client.presigned_put("k", content_type="video/mp4")).query)[
        "X-Amz-SignedHeaders"
    ][0]
    assert "content-type" in signed_headers
    # Without it the client is free to send any Content-Type.
    plain = parse_qs(urlparse(client.presigned_put("k")).query)["X-Amz-SignedHeaders"][0]
    assert "content-type" not in plain


def test_presigned_get_is_a_get_url_for_the_key():
    url = client.presigned_get("evidence/1.jpg")
    assert urlparse(url).path == "/test-bucket/evidence/1.jpg"
    assert "X-Amz-Signature" in parse_qs(urlparse(url).query)


def test_public_url_is_none_without_a_configured_base(monkeypatch):
    assert client.public_url("k") is None


def test_public_url_joins_base_and_key(monkeypatch):
    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "https://pub-abc.r2.dev")
    assert client.public_url("evidence/1.jpg") == "https://pub-abc.r2.dev/evidence/1.jpg"


def test_download_url_prefers_the_public_url_when_configured(monkeypatch):
    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "https://pub-abc.r2.dev")
    assert client.download_url("evidence/1.jpg") == "https://pub-abc.r2.dev/evidence/1.jpg"


def test_download_url_falls_back_to_presigning():
    url = client.download_url("evidence/1.jpg")
    assert "X-Amz-Signature" in parse_qs(urlparse(url).query)


def test_head_returns_none_for_a_missing_object(monkeypatch):
    from botocore.exceptions import ClientError

    def raise_404(**kwargs):
        raise ClientError({"ResponseMetadata": {"HTTPStatusCode": 404}}, "HeadObject")

    monkeypatch.setattr(client.get_client(), "head_object", raise_404)
    assert client.head("missing") is None


def test_head_reraises_errors_that_are_not_a_missing_object(monkeypatch):
    from botocore.exceptions import ClientError

    def raise_403(**kwargs):
        raise ClientError({"ResponseMetadata": {"HTTPStatusCode": 403}}, "HeadObject")

    monkeypatch.setattr(client.get_client(), "head_object", raise_403)
    with pytest.raises(ClientError):
        client.head("forbidden")


def test_presigned_put_signs_content_length_when_given():
    # Signed, so R2 itself rejects a PUT whose body is a different size. Without
    # this a client could declare 1 MB and upload 5 GB.
    query = parse_qs(urlparse(client.presigned_put("k", content_length=1024)).query)

    assert "content-length" in query["X-Amz-SignedHeaders"][0]


def test_presigned_put_omits_content_length_when_not_given():
    query = parse_qs(urlparse(client.presigned_put("k")).query)

    assert "content-length" not in query["X-Amz-SignedHeaders"][0]


class FakeBody:
    """A boto3 StreamingBody with the two methods the readers call."""

    def __init__(self, payload: bytes):
        self._payload = payload
        self.closed = False

    def read(self) -> bytes:
        return self._payload

    def close(self) -> None:
        self.closed = True


def test_get_bytes_returns_the_object_uninterpreted(monkeypatch):
    # A calibration is whatever the tool that produced it wrote — in practice an
    # OpenCV FileStorage document, not JSON — so this must not try to parse it.
    body = FakeBody(b"%YAML:1.0\n---\ncamera_matrix: !!opencv-matrix\n")
    monkeypatch.setattr(
        client.get_client(), "get_object", lambda **kwargs: {"Body": body}
    )

    assert client.get_bytes("calibration/f1/camera_model.yml").startswith(b"%YAML")


def test_get_bytes_closes_the_body(monkeypatch):
    # The stream holds a connection from the pool until it is closed, and a worker
    # that leaked one per job would run out.
    body = FakeBody(b"anything")
    monkeypatch.setattr(
        client.get_client(), "get_object", lambda **kwargs: {"Body": body}
    )

    client.get_bytes("calibration/f1/camera_model.yml")

    assert body.closed
