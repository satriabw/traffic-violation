"""The thread behind the explain endpoint.

Nothing here touches a database, an explainer or a violation. The actor takes messages
and runs a callable over them, which is the whole of what it knows — so these tests are
about delivery, ordering, and staying alive, and the handler is a fake that records.

NOTHING SLEEPS. Every wait is on a threading.Event with a timeout, so a broken actor
fails in milliseconds and a working one is never paced by an arbitrary delay. A test that
slept would be slow when it passed and unreliable when it failed.
"""

import threading

import pytest
from fastapi import HTTPException

from site_service import actor as actor_module
from site_service.actor import Explain, ExplanationActor, get_actor, stop_actor

# Generous: it only ever elapses when something is actually broken, and a thread handing
# a message over takes microseconds.
WAIT = 5.0


@pytest.fixture
def running():
    """Start actors and guarantee they are stopped.

    Without the teardown a failing test leaves a non-daemon thread behind, and the whole
    suite hangs at exit rather than reporting the failure.
    """
    started: list[ExplanationActor] = []

    def start(handle):
        instance = ExplanationActor(handle)
        instance.start()
        started.append(instance)
        return instance

    yield start
    for instance in started:
        instance.stop()


def _message(violation_id: str = "v-1") -> Explain:
    return Explain(site_id="s-1", violation_id=violation_id)


def test_a_sent_message_reaches_the_handler(running):
    seen = []
    arrived = threading.Event()

    def handle(message):
        seen.append(message)
        arrived.set()

    running(handle).send(_message())

    assert arrived.wait(WAIT), "the handler was never called"
    assert seen == [_message()]


def test_messages_are_handled_in_the_order_they_were_sent(running):
    seen = []
    done = threading.Event()

    def handle(message):
        seen.append(message.violation_id)
        if len(seen) == 3:
            done.set()

    instance = running(handle)
    for violation_id in ("first", "second", "third"):
        instance.send(_message(violation_id))

    assert done.wait(WAIT), "not every message was handled"
    assert seen == ["first", "second", "third"]


def test_a_handler_that_raises_does_not_kill_the_thread(running):
    """The failure mode this actor most needs to survive.

    There is no supervisor. A dead thread means the service goes on accepting
    explanations and silently never performs one, which looks exactly like working —
    the worst outcome available here, and worse than losing the one violation that
    caused it.
    """
    seen = []
    second = threading.Event()

    def handle(message):
        seen.append(message.violation_id)
        if message.violation_id == "explodes":
            raise RuntimeError("something nobody thought of")
        second.set()

    instance = running(handle)
    instance.send(_message("explodes"))
    instance.send(_message("survives"))

    assert second.wait(WAIT), "the thread died with the first message"
    assert seen == ["explodes", "survives"]


def test_stopping_ends_the_thread(running):
    instance = running(lambda message: None)

    instance.stop()

    assert instance._thread is None


def test_stopping_drains_what_is_already_waiting(running):
    """The sentinel goes to the back of the mailbox, not the front.

    A shutdown finishes what it has been given rather than dropping it; only what it
    cannot finish in time is left for the next startup to mark failed.
    """
    handled = []
    holding = threading.Event()
    released = threading.Event()

    def handle(message):
        if message.violation_id == "blocks":
            holding.set()
            released.wait(WAIT)
        handled.append(message.violation_id)

    instance = running(handle)
    instance.send(_message("blocks"))
    # Only queue the rest once the actor is definitely occupied, so they are genuinely
    # waiting behind it when stop() is called rather than racing it.
    assert holding.wait(WAIT)
    instance.send(_message("queued"))
    released.set()

    instance.stop()

    assert handled == ["blocks", "queued"]


def test_starting_twice_is_not_an_error(running):
    instance = running(lambda message: None)

    instance.start()

    assert instance._thread is not None


def test_stopping_something_never_started_is_not_an_error():
    ExplanationActor(lambda message: None).stop()


def test_the_dependency_refuses_when_no_actor_is_running():
    """503 rather than building one, and the distinction is the point.

    A dependency that started a thread on first use would let any request bring one into
    existence as a side effect — including a test that forgot to override it, which is
    how a suite ends up hanging on a real two-minute call to a service nobody meant to
    reach. The thread's lifetime belongs to the app, so lifespan owns it.
    """
    assert actor_module._actor is None

    with pytest.raises(HTTPException) as raised:
        get_actor()

    assert raised.value.status_code == 503


def test_stopping_when_none_was_started_is_not_an_error():
    stop_actor()

    assert actor_module._actor is None
