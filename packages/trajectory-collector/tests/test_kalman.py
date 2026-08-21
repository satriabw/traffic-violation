import numpy as np
import pytest

from trajectory_collector.kalman import ACCELERATION, POSITION, VELOCITY, KalmanFilter

DT = 1.0 / 30.0


def _state(kalman: KalmanFilter) -> tuple[np.ndarray, np.ndarray]:
    return kalman.x[POSITION].reshape(-1), kalman.x[VELOCITY].reshape(-1)


def test_predicting_moves_the_state_by_its_velocity():
    kalman = KalmanFilter(x=0.0, y=0.0, vx=3.0, vy=0.0, dt=1.0)

    kalman.predict()

    position, _ = _state(kalman)
    assert position == pytest.approx(np.array([3.0, 0.0]))


def test_correcting_pulls_the_state_towards_what_was_measured():
    kalman = KalmanFilter(x=0.0, y=0.0, vx=0.0, vy=0.0, dt=DT)

    kalman.predict()
    kalman.correct(np.array([1.0, 0.0]))

    position, _ = _state(kalman)
    # Not all the way there — a measurement is evidence, not truth — but most of the
    # way, because measurement noise is small against the initial velocity variance.
    assert 0.5 < position[0] <= 1.0


def test_a_constant_speed_track_converges_on_its_true_speed():
    # The one thing this filter exists to produce. Ten metres per second along x,
    # measured exactly, and after a second of frames the estimate should agree.
    truth = 10.0
    kalman = KalmanFilter(x=0.0, y=0.0, vx=truth, vy=0.0, dt=DT)

    for step in range(1, 31):
        kalman.predict()
        kalman.correct(np.array([truth * step * DT, 0.0]))

    _, velocity = _state(kalman)
    assert float(np.linalg.norm(velocity)) == pytest.approx(truth, abs=0.05)


def test_a_noisy_track_still_converges():
    # Measurements jitter by centimetres, which is what differencing consecutive
    # positions would amplify into wild speeds and the filter is here to absorb.
    truth = 10.0
    rng = np.random.default_rng(0)
    kalman = KalmanFilter(x=0.0, y=0.0, vx=truth, vy=0.0, dt=DT)

    for step in range(1, 91):
        kalman.predict()
        kalman.correct(np.array([truth * step * DT + rng.normal(0, 0.05), 0.0]))

    _, velocity = _state(kalman)
    assert float(np.linalg.norm(velocity)) == pytest.approx(truth, abs=0.5)


def test_a_longer_step_moves_the_state_further():
    kalman = KalmanFilter(x=0.0, y=0.0, vx=2.0, vy=0.0, dt=1.0)

    kalman.predict(dt=5.0)

    position, _ = _state(kalman)
    assert position == pytest.approx(np.array([10.0, 0.0]))


def test_a_step_over_a_gap_does_not_leak_into_the_next_one():
    # The transition matrix is rebuilt for a longer step, and it has to be rebuilt back
    # again afterwards. Without remembering which interval the matrix was last built
    # for, a nominal step following a gap silently advances the state by the gap's
    # length — every frame after a single dropped detection, forever.
    kalman = KalmanFilter(x=0.0, y=0.0, vx=2.0, vy=0.0, dt=1.0)

    kalman.predict(dt=5.0)
    kalman.predict()

    position, _ = _state(kalman)
    assert position == pytest.approx(np.array([12.0, 0.0]))


def test_a_longer_step_admits_more_uncertainty():
    near = KalmanFilter(x=0.0, y=0.0, vx=0.0, vy=0.0, dt=1.0)
    far = KalmanFilter(x=0.0, y=0.0, vx=0.0, vy=0.0, dt=1.0)

    near.predict(dt=1.0)
    far.predict(dt=10.0)

    # Otherwise a filter that skipped ten frames would come back as confident as one
    # that skipped none.
    assert far.P[0, 0] > near.P[0, 0]


def test_inflating_widens_velocity_and_acceleration_but_not_position():
    # After a gap the object is roughly where the model says, but what it was doing a
    # second ago has stopped being evidence about what it is doing now.
    kalman = KalmanFilter(x=0.0, y=0.0, vx=1.0, vy=0.0, dt=DT)
    before = kalman.P.diagonal().copy()

    kalman.inflate_covariance(4.0)

    after = kalman.P.diagonal()
    assert after[POSITION] == pytest.approx(before[POSITION])
    assert after[VELOCITY] == pytest.approx(before[VELOCITY] * 4.0)
    assert after[ACCELERATION] == pytest.approx(before[ACCELERATION] * 4.0)


def test_an_inflated_filter_is_moved_further_by_the_next_measurement():
    # The point of inflating: the measurement after a gap should be able to move the
    # estimate sharply, rather than be dismissed as noise by a filter still confident
    # about a speed from a second ago.
    def displacement(inflate: bool) -> float:
        kalman = KalmanFilter(x=0.0, y=0.0, vx=10.0, vy=0.0, dt=DT)
        if inflate:
            kalman.inflate_covariance(10.0)
        kalman.predict(dt=DT)
        before = kalman.x[POSITION].reshape(-1).copy()
        kalman.correct(np.array([5.0, 0.0]))
        return abs(float(kalman.x[POSITION].reshape(-1)[0] - before[0]))

    assert displacement(inflate=True) > displacement(inflate=False)
