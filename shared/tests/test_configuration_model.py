import pytest
from pydantic import ValidationError

from shared.models.configuration import ConfigurationCreate, ConfigurationResponse


def test_configuration_create_accepts_a_file_id():
    assert ConfigurationCreate(file_id="file-1").file_id == "file-1"


def test_configuration_create_requires_a_file_id():
    with pytest.raises(ValidationError):
        ConfigurationCreate()


def test_configuration_create_no_longer_accepts_a_url():
    # A url was an unverifiable claim; a file_id points at a row whose upload the
    # service has confirmed.
    assert "url" not in ConfigurationCreate.model_fields


def test_configuration_create_ignores_site_id_from_body():
    # site_id is owned by the path, never the payload — a client that sends one
    # must not be able to attach a configuration to a different site.
    configuration = ConfigurationCreate(file_id="file-1", site_id="other-site")
    assert not hasattr(configuration, "site_id")


def test_configuration_response_carries_version_and_site():
    response = ConfigurationResponse(
        id="cfg-1",
        site_id="site-1",
        file_id="file-1",
        version=2,
        created_at="2026-08-18T00:00:00",
        updated_at="2026-08-18T00:00:00",
    )

    assert response.version == 2
    assert response.site_id == "site-1"
    assert response.file_id == "file-1"
