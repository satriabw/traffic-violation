"""Choosing the provider, once, at startup.

The registry is a dict rather than a chain of ifs so that adding a provider is one
import and one entry. Nothing outside this package names a provider class: main.py
asks for whatever LLM_PROVIDER selected, which is what makes "we will extend it later"
a change confined to this directory.
"""

from shared import config

from llm_service.providers.base import Provider
from llm_service.providers.claude import ClaudeProvider

_PROVIDERS = {"claude": ClaudeProvider}


class UnknownProvider(RuntimeError):
    pass


def get_provider() -> Provider:
    """Build the configured provider.

    Raises rather than falling back to a default. A service that quietly ran on a
    different model than the one it was configured for would store explanations
    attributed to something that never produced them.
    """
    try:
        provider = _PROVIDERS[config.LLM_PROVIDER]
    except KeyError:
        raise UnknownProvider(
            f"LLM_PROVIDER is {config.LLM_PROVIDER!r}, which is not one of: "
            f"{', '.join(sorted(_PROVIDERS))}."
        ) from None
    return provider()


__all__ = ["Provider", "UnknownProvider", "get_provider"]
