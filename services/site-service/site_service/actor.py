"""The thing that explains violations while the caller gets on with their day.

WHY THERE IS AN ACTOR HERE AT ALL. Explaining a violation is a call to a model that
thinks before it answers — commonly twenty to ninety seconds. Doing that inside the
request that asked for it parks a thread and blocks a browser tab on something with
nothing to show until it finishes. So the request hands the work to this and returns; the
row's status is what the client watches instead.

ONE MAILBOX, ONE THREAD, ONE MESSAGE AT A TIME. Nothing here shares mutable state with a
request thread: the message carries what to work on, and everything else the handler
needs it reads for itself. That is what makes the concurrency story short enough to hold
in your head — there is no lock in this file, and none is missing.

WHAT THIS IS NOT. It is not a queue, and the difference is durability. detection-worker
and evidence-worker are separate processes fed by Redis precisely so their jobs outlive
the process that enqueued them; a message in this mailbox does not. If site-service
stops, whatever it was holding is gone, and the row is left claiming to be pending —
which is why `fail_pending_explanations` runs at startup and turns that claim into a
verdict a client can act on. The trade was taken knowingly: no broker, no second
deployment unit, and no job surviving a restart.
"""

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Callable

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# How long stop() waits for the thread to finish what it is holding. Long enough for a
# handler that is writing its result to get the write out, short enough that a shutdown
# is not held open by a model call that has barely started — those take minutes, and
# waiting for one would turn every deploy into a two-minute pause.
STOP_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class Explain:
    """Explain this one violation.

    Ids and nothing else. The handler reads the violation, the site and the configuration
    itself, so a message that waited in the mailbox works from the row as it is when the
    work starts rather than as it was when somebody asked — and a large object like the
    metadata blob never sits in memory waiting its turn.

    Frozen because a message that could be edited after it was sent is a race with no
    upside.
    """

    site_id: str
    violation_id: str


# Put on the mailbox by stop(). An instance rather than None: None is a value a caller
# could send by accident, and a sentinel that cannot be confused with a message is worth
# the one line.
_STOP = object()


class ExplanationActor:
    """A mailbox, a thread, and a handler to run over each message.

    Deliberately knows nothing about violations, databases or explanations. It receives
    messages and runs a callable over them, which is what makes it testable without any
    of those — and what keeps everything about *how* a violation gets explained in the
    service layer where the rest of that logic already lives.
    """

    def __init__(self, handle: Callable[[Explain], None]):
        self._handle = handle
        # Unbounded. A bounded mailbox would mean deciding what to do when it fills, and
        # the honest answer at this size is that it will not: violations arrive at human
        # speed, from people opening them one at a time. Backpressure here would be
        # machinery guarding a load that does not exist.
        self._mailbox: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Begin consuming. Idempotent, so a double start is not an error worth raising."""
        if self._thread is not None:
            return
        # Not a daemon. A daemon thread is killed mid-statement at interpreter exit,
        # which for this one means killed part-way through writing a result — and the
        # row it was about to update would keep saying pending with no process left to
        # correct it. stop() is what ends this thread, and lifespan always calls it.
        self._thread = threading.Thread(
            target=self._run, name="explanation-actor", daemon=False
        )
        self._thread.start()

    def send(self, message: Explain) -> None:
        """Hand over one violation. Returns immediately; the work happens on the thread."""
        self._mailbox.put(message)

    def stop(self, timeout: float = STOP_TIMEOUT_SECONDS) -> None:
        """Ask the thread to finish and wait for it, briefly.

        The sentinel goes to the back of the mailbox, so anything already waiting is
        handled first — a shutdown drains what it can rather than dropping it. What it
        cannot drain within `timeout` is abandoned, and those rows are exactly the ones
        the next startup turns into 'failed'.
        """
        if self._thread is None:
            return
        self._mailbox.put(_STOP)
        self._thread.join(timeout)
        if self._thread.is_alive():
            logger.warning(
                "explanation actor did not stop within %.0fs; "
                "anything it was holding will be marked failed at next startup",
                timeout,
            )
        self._thread = None

    def _run(self) -> None:
        while True:
            message = self._mailbox.get()
            if message is _STOP:
                return
            try:
                self._handle(message)
            except Exception:
                # EVERY exception, and the breadth is the point. This thread has no
                # supervisor: if it dies the service goes on accepting explanations and
                # silently never performs one, which is the worst failure available here
                # because it looks exactly like working. The handler already writes
                # 'failed' for the failures it expects; this is for the ones nobody
                # thought of, and it costs one violation rather than all of them.
                #
                # The row may be left saying 'pending' — the handler raised before it
                # could write anything else. That is the same state a crash leaves, and
                # it is resolved the same way, by startup.
                logger.exception(
                    "explaining violation %s failed unexpectedly", message.violation_id
                )


_actor: ExplanationActor | None = None


def build_actor() -> ExplanationActor:
    """The process's actor, wired to the real handler.

    Imported inside rather than at module scope, because `service` imports the llm client
    and this module is imported by the router that `service` sits beneath — a top-level
    import here closes the circle. It is also the only place in this file that knows what
    a violation is, which is the seam that keeps the actor itself testable with a fake
    handler and no database.

    The connection is not passed in. `get_db` is thread-local, so calling it from the
    actor's thread hands that thread its own sqlite connection — which is exactly the
    property the request threadpool already relies on, and the reason this needs no
    connection plumbing of its own.
    """
    from site_service import service
    from site_service.db import get_db
    from site_service.llm import get_explainer
    from site_service.storage import get_storage

    def handle(message: Explain) -> None:
        service.perform_explanation(
            get_db(), get_storage(), get_explainer(), message.site_id, message.violation_id
        )

    return ExplanationActor(handle)


def start_actor() -> ExplanationActor:
    """Build the process's actor and start its thread. Called once, from lifespan."""
    global _actor
    if _actor is None:
        _actor = build_actor()
        _actor.start()
    return _actor


def get_actor() -> ExplanationActor:
    """The running actor, as a dependency.

    NOT LAZY, and that is the point. Building here on first use would mean any request
    that reached the endpoint could spin up a thread and a live connection to llm-service
    as a side effect — including from a test that forgot to override this, which is how
    an entire suite ends up hanging on a real two-minute timeout. The thread's lifetime
    belongs to the app, so `lifespan` starts it and this only ever hands it out.

    503 when it is not running, the same answer llm-service's `get_current_provider`
    gives for the same situation: the service is up but the thing behind this endpoint is
    not ready, and the caller should try again rather than be told their violation is at
    fault.

    A dependency rather than a module global reached for directly, so a test substitutes
    a fake through `app.dependency_overrides` — the same reason `get_explainer` and
    `get_storage` are dependencies.
    """
    if _actor is None:
        raise HTTPException(
            status_code=503, detail="The explanation service is not ready"
        )
    return _actor


def stop_actor() -> None:
    """Stop the process's actor, if one was ever started. Called from lifespan on the way out."""
    global _actor
    if _actor is not None:
        _actor.stop()
        _actor = None
