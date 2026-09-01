"""The shape every provider has to have.

A Protocol rather than a base class: a provider is one method, nothing here has state
worth inheriting, and structural typing means a test fake is a small class rather than
a subclass of something it does not otherwise care about.
"""

from typing import Protocol

from shared.models.violation import ViolationExplanation


class Provider(Protocol):
    # Reported back on the response so a stored explanation can be traced to what
    # produced it. The provider's own name for the model, not the setting that asked
    # for it — those differ the moment anything aliases or falls back.
    model: str

    def explain(self, system: str, prompt: str) -> ViolationExplanation:
        """One shot, no memory, no tools. Raises on anything that is not an answer."""
