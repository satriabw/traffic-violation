from shared import config


def test_missing_s3_settings_is_empty_when_all_are_present(monkeypatch):
    for name in ("S3_ENDPOINT_URL", "S3_BUCKET", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY"):
        monkeypatch.setattr(config, name, "set")

    assert config.missing_s3_settings() == []


def test_missing_s3_settings_names_every_empty_setting(monkeypatch):
    monkeypatch.setattr(config, "S3_ENDPOINT_URL", "")
    monkeypatch.setattr(config, "S3_BUCKET", "")
    monkeypatch.setattr(config, "S3_ACCESS_KEY_ID", "set")
    monkeypatch.setattr(config, "S3_SECRET_ACCESS_KEY", "set")

    assert config.missing_s3_settings() == ["S3_ENDPOINT_URL", "S3_BUCKET"]


def test_missing_s3_settings_ignores_the_optional_ones(monkeypatch):
    # A public base URL is genuinely optional — its absence means "presign downloads".
    for name in ("S3_ENDPOINT_URL", "S3_BUCKET", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY"):
        monkeypatch.setattr(config, name, "set")
    monkeypatch.setattr(config, "S3_PUBLIC_BASE_URL", "")

    assert config.missing_s3_settings() == []
