import pytest
from fastapi.testclient import TestClient
from shared import config
from shared.models.violation import Severity, ViolationExplanation

from llm_service.main import app, get_current_provider

TOKEN = "test-token"


class FakeProvider:
    """A provider that answers without a network.

    Structural, not a subclass — Provider is a Protocol, which is most of the reason
    it is one. Records what it was asked so the prompt can be asserted on without a
    second seam for inspecting it.
    """

    model = "fake-model-1"

    def __init__(self, explanation=None, error=None):
        self.explanation = explanation or ViolationExplanation(
            explanation="Entered against a red signal.",
            severity=Severity.MEDIUM,
            severity_basis=["two other vehicle tracks present"],
            observations=["three tracks on the scene"],
            evidence_concerns=[],
            confidence=0.6,
        )
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def explain(self, system, prompt):
        self.calls.append((system, prompt))
        if self.error:
            raise self.error
        return self.explanation


@pytest.fixture(autouse=True)
def token(monkeypatch):
    # Every test authenticates, so the guard is set up once here rather than in each.
    # The one test that checks the guard overrides the header instead.
    monkeypatch.setattr(config, "LLM_SERVICE_TOKEN", TOKEN)


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture
def client(provider):
    app.dependency_overrides[get_current_provider] = lambda: provider
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def client_for():
    """A client whose provider fails in a named way.

    A factory rather than a parametrised fixture because the errors under test are
    SDK exception instances, which are easier to build at the call site than to
    thread through fixture params.
    """
    clients = []

    def build(error):
        app.dependency_overrides[get_current_provider] = lambda: FakeProvider(error=error)
        client = TestClient(app, raise_server_exceptions=False)
        clients.append(client)
        return client

    yield build
    app.dependency_overrides.clear()
