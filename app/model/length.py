"""Reply length (section 6.2): sensor fusion over (L, kappa, v) and the self-report noise estimate."""

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


def self_report_variance(self_reports: list[float], requested: list[float], requested_var: float) -> float:
    """Var(s1 - s3) = sigma1^2 + sigma3^2 for two independent sensors, with sigma3^2 known from widget turns."""
    return max(mean_var([a - b for a, b in zip(self_reports, requested)])[1] - requested_var, 0.01)
