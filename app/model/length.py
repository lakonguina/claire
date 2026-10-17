"""Reply length (section 6.2): sensor fusion over (L, kappa, v) and the three-cornered hat."""

import math
from dataclasses import dataclass

from app.model.stats import mean_var

STUDENT_DOF = 4.0
EXACT_VAR = 1e-6


@dataclass
class Gaussian3:
    """Joint Gaussian over x = (L, kappa, v): log length, compliance, log writing speed."""

    mean: list[float]
    cov: list[list[float]]

    @classmethod
    def independent(cls, means: list[float], variances: list[float]) -> "Gaussian3":
        return cls(list(means), [[variances[i] if i == j else 0.0 for j in range(3)] for i in range(3)])

    def observe(self, h: list[float], z: float, noise_var: float, robust: bool = False) -> None:
        """Scalar linear observation z = h.x + e (Kalman update), Student-t down-weighting if robust."""
        ph = [sum(self.cov[i][j] * h[j] for j in range(3)) for i in range(3)]
        residual = z - sum(hi * xi for hi, xi in zip(h, self.mean))
        s = sum(hi * phi for hi, phi in zip(h, ph)) + noise_var
        if robust:
            noise_var *= (STUDENT_DOF + residual**2 / s) / (STUDENT_DOF + 1)
            s = sum(hi * phi for hi, phi in zip(h, ph)) + noise_var
        gain = [p / s for p in ph]
        self.mean = [m + g * residual for m, g in zip(self.mean, gain)]
        self.cov = [[self.cov[i][j] - gain[i] * ph[j] for j in range(3)] for i in range(3)]


@dataclass
class Sensors:
    exact_words: int | None = None
    self_report_words: int | None = None
    writing_seconds: float | None = None  # duration minus overhead
    requested_words: int | None = None


def fuse(prior: Gaussian3, sensors: Sensors, noise: dict[str, float]) -> Gaussian3:
    """Fuse whichever sensors are present. noise maps sensor name to its standard deviation."""
    post = Gaussian3([*prior.mean], [row[:] for row in prior.cov])
    if sensors.exact_words:
        post.observe([1, 0, 0], math.log(sensors.exact_words), EXACT_VAR)
    if sensors.self_report_words:
        post.observe([1, 0, 0], math.log(sensors.self_report_words), noise["self_report"] ** 2)
    if sensors.writing_seconds and sensors.writing_seconds > 0:
        post.observe([1, 0, -1], math.log(sensors.writing_seconds), noise["writing_time"] ** 2, robust=True)
    if sensors.requested_words:
        post.observe([1, -1, 0], math.log(sensors.requested_words), noise["requested"] ** 2)
    return post


def three_cornered_hat(s1: list[float], s2: list[float], s3: list[float]) -> tuple[float, float, float]:
    """Noise variances of three independent sensors of the same quantity, from pairwise differences."""
    v12 = mean_var([a - b for a, b in zip(s1, s2)])[1]
    v13 = mean_var([a - c for a, c in zip(s1, s3)])[1]
    v23 = mean_var([b - c for b, c in zip(s2, s3)])[1]
    return (
        max((v12 + v13 - v23) / 2, 0.01),
        max((v12 + v23 - v13) / 2, 0.01),
        max((v13 + v23 - v12) / 2, 0.01),
    )
