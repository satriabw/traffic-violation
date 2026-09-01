"""Claude, through the Anthropic SDK.

One request, one response, no conversation state. The violation is fully described by
the prompt, so there is nothing for a second turn to add — and an explanation that
depended on what was asked before it would not be reproducible from the row it is
stored on.
"""

import anthropic
from shared import config
from shared.models.violation import ViolationExplanation


class ClaudeProvider:
    def __init__(self, client: anthropic.Anthropic | None = None, model: str | None = None):
        # Injectable so tests never construct a real client, and so a deployment that
        # needs different timeouts or retries can pass one in rather than reaching into
        # this module.
        self._client = client or anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.model = model or config.LLM_MODEL

    def explain(self, system: str, prompt: str) -> ViolationExplanation:
        """Ask for a ViolationExplanation and get one back, already validated.

        `messages.parse` constrains the response to the model's schema and validates it
        on the way out, which is why nothing here parses JSON or repairs it. It also
        means the prompt says nothing about output format — an instruction to "return
        JSON only" alongside a schema is a second, weaker copy of the same rule.

        Not streamed. The response is a few hundred tokens and the caller is an HTTP
        request that either gets an explanation or does not; streaming would buy
        nothing and cost the simplicity.

        Thinking is left at the model's default, which on Opus 5 is adaptive. Judging
        severity against a rubric is exactly the kind of thing worth thinking about,
        and the cost of it is small against the round trip.
        """
        response = self._client.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=ViolationExplanation,
        )
        return response.parsed_output
