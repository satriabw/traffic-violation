"""A constant-jerk Kalman filter over a point moving on a plane.

Trajectories are filtered rather than differenced because the measurements are noisy in
a way that differencing amplifies. A detection box jitters by a few pixels frame to
frame, and near the horizon a few pixels is metres — so a raw position difference over
one frame at 30fps produces speeds that swing wildly while the object itself moves
smoothly. The filter is what turns a jittering box into a speed anyone would put in a
report.

Six states: position, velocity and acceleration in each axis. Jerk — the derivative of
acceleration — is the process noise, so the model is "acceleration drifts", which is
what a vehicle actually does. A constant-velocity model would fight every genuine
acceleration, and a constant-acceleration one with no noise term would coast straight
through a braking event.
"""

from dataclasses import dataclass

import numpy as np

# Indices into the state vector, named because `x[2:4]` reads as nothing.
POSITION = slice(0, 2)
VELOCITY = slice(2, 4)
ACCELERATION = slice(4, 6)


@dataclass(frozen=True)
class FilterParams:
    """Tuning. Frozen so one track's filter cannot alter the next one's."""

    # Standard deviation of the jerk driving the process noise, in m/s³. The single
    # knob that trades responsiveness against smoothness: raise it and the filter
    # follows a hard brake sooner, at the cost of following the jitter too.
    jerk_std: float = 0.7
    # Variance of a position measurement, in m². Small relative to the initial
    # velocity variance below, which is what makes the filter trust where an object
    # is much more than it trusts how fast it was going when it first appeared.
    measurement_noise: float = 0.01
    # Initial uncertainty. Position is near-certain — it *is* the first measurement —
    # while velocity is a two-point estimate over a few frames and acceleration is a
    # guess at zero.
    position_variance: float = 0.01
    velocity_variance: float = 0.5
    acceleration_variance: float = 0.1


class KalmanFilter:
    """One track's filter. Stateful, and belongs to exactly one tracker id."""

    def __init__(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        dt: float,
        ax: float = 0.0,
        ay: float = 0.0,
        params: FilterParams = FilterParams(),
    ):
        self._params = params
        self._dt = dt

        self.x = np.array(
            [[float(x)], [float(y)], [float(vx)], [float(vy)], [float(ax)], [float(ay)]]
        )
        self.P = np.diag(
            [
                params.position_variance,
                params.position_variance,
                params.velocity_variance,
                params.velocity_variance,
                params.acceleration_variance,
                params.acceleration_variance,
            ]
        )

        # The interval A and Q were last built for. Tracked separately from `_dt`,
        # which stays the nominal frame interval: a step over a gap rebuilds both
        # matrices for that longer interval, and without remembering that, the next
        # normal-length step would silently reuse the gap's transition matrix and
        # advance the state several frames too far.
        self._matrix_dt = dt
        self._rebuild(dt, params.jerk_std)

        # Position only. Nothing measures velocity — that is the whole point of
        # estimating it.
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=float)
        self.R = np.eye(2) * params.measurement_noise

    def _rebuild(self, dt: float, jerk_std: float) -> None:
        half_dt_squared = (dt**2) * 0.5
        self.A = np.array(
            [
                [1, 0, dt, 0, half_dt_squared, 0],
                [0, 1, 0, dt, 0, half_dt_squared],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )

        # The standard continuous-jerk discretisation. Every term is an integral of
        # dt over the interval, which is why a longer step admits more uncertainty in
        # every state at once rather than only in position.
        q = jerk_std**2
        self.Q = (
            np.array(
                [
                    [dt**5 / 20, 0, dt**4 / 8, 0, dt**3 / 6, 0],
                    [0, dt**5 / 20, 0, dt**4 / 8, 0, dt**3 / 6],
                    [dt**4 / 8, 0, dt**3 / 3, 0, dt**2 / 2, 0],
                    [0, dt**4 / 8, 0, dt**3 / 3, 0, dt**2 / 2],
                    [dt**3 / 6, 0, dt**2 / 2, 0, dt, 0],
                    [0, dt**3 / 6, 0, dt**2 / 2, 0, dt],
                ],
                dtype=float,
            )
            * q
        )
        self._matrix_dt = dt

    def predict(self, dt: float | None = None) -> np.ndarray:
        """Advance the state by `dt`, defaulting to the nominal frame interval."""
        step = self._dt if dt is None else dt
        if step != self._matrix_dt:
            # A longer step is a bigger leap in the dark, so the noise driving it grows
            # with it — otherwise a filter that skipped ten frames would come back as
            # confident as one that skipped none.
            self._rebuild(step, self._params.jerk_std * np.sqrt(step / self._dt))

        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x

    def inflate_covariance(self, scale: float) -> None:
        """Widen the velocity and acceleration uncertainty after a gap.

        Position survives a gap reasonably well — the object is roughly where the model
        says. Velocity and acceleration do not: the longer the track was missing, the
        less its old velocity says about its current one. Inflating them is what lets
        the next measurement move the estimate sharply instead of being dismissed as
        noise by a filter that is still confident about a speed from a second ago.
        """
        for index in (VELOCITY, ACCELERATION):
            for i in range(index.start, index.stop):
                self.P[i, i] *= scale

    def correct(self, measurement: np.ndarray) -> np.ndarray:
        """Fold in a measured position and return the corrected state."""
        z = np.asarray(measurement, dtype=float).reshape(2, 1)

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = self.P - K @ self.H @ self.P
        return self.x
