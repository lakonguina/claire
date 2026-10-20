"""Reply length: sensor fusion and the self-report noise."""

import dataclasses
import math

import numpy as np

from app.model import stats

STUDENT_DOF = 4.0
EXACT_VAR = 1e-6


@dataclasses.dataclass
class Gaussian3:
    """Joint Gaussian over (log length, log compliance, log writing speed)."""

    mean: np.ndarray
    cov: np.ndarray

    @classmethod
    def independent(cls, means, variances) -> "Gaussian3":
        """Returns a Gaussian with a diagonal covariance."""
        return cls(np.array(means, dtype=float), np.diag(variances))

    def observe(
        self, h, z: float, noise_var: float, robust: bool = False
    ) -> None:
        """Applies a scalar observation z = h.x + e (Kalman update).

        Args:
            h: The observation vector.
            z: The observed value.
            noise_var: The variance of e.
            robust: Down-weights large residuals (Student-t) if true.
        """
        h = np.asarray(h, dtype=float)
        ph = self.cov @ h
        residual = z - h @ self.mean
        prior_var = h @ ph
        if robust:
            ratio = residual**2 / (prior_var + noise_var)
            noise_var *= (STUDENT_DOF + ratio) / (STUDENT_DOF + 1)
        gain = ph / (prior_var + noise_var)
        self.mean = self.mean + gain * residual
        self.cov = self.cov - np.outer(gain, ph)


@dataclasses.dataclass
class Sensors:
    """The length sensors of one turn; any may be missing."""

    exact_words: int | None = None
    self_report_words: int | None = None
    writing_seconds: float | None = None  # Duration minus overhead.
    requested_words: int | None = None


def fuse(
    prior: Gaussian3, sensors: Sensors, noise: dict[str, float]
) -> Gaussian3:
    """Fuses the sensors present into the prior.

    Args:
        prior: The prior over (L, kappa, v).
        sensors: The turn's sensor readings.
        noise: Sensor name -> noise standard deviation.

    Returns:
        The posterior over (L, kappa, v).
    """
    post = Gaussian3(prior.mean.copy(), prior.cov.copy())
    if sensors.exact_words:
        post.observe([1, 0, 0], math.log(sensors.exact_words), EXACT_VAR)
    if sensors.self_report_words:
        post.observe(
            [1, 0, 0],
            math.log(sensors.self_report_words),
            noise["self_report"] ** 2,
        )
    if sensors.writing_seconds and sensors.writing_seconds > 0:
        post.observe(
            [1, 0, -1],
            math.log(sensors.writing_seconds),
            noise["writing_time"] ** 2,
            robust=True,
        )
    if sensors.requested_words:
        post.observe(
            [1, -1, 0],
            math.log(sensors.requested_words),
            noise["requested"] ** 2,
        )
    return post


def self_report_variance(
    self_reports: list[float], requested: list[float], requested_var: float
) -> float:
    """Returns sigma1^2 from Var(s1 - s3) = sigma1^2 + sigma3^2."""
    diffs = [a - b for a, b in zip(self_reports, requested)]
    return max(stats.mean_var(diffs)[1] - requested_var, 0.01)
