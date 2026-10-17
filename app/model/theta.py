"""Length likelihood and the multilevel theta / alpha grids (SPEC 6.5)."""

import dataclasses
import math

from app import vocab
from app.model import reaction
from app.model import stats

SUFFICIENT = frozenset({vocab.Outcome.MOVED_ON, vocab.Outcome.TOO_LONG})
INSUFFICIENT = frozenset({vocab.Outcome.TOO_SHORT, vocab.Outcome.MISSING_INFO})
# Profile-page length choices: the ratio of needed words they mean.
EDIT_RATIOS = {
    vocab.LengthChoice.SHORT: 0.4,
    vocab.LengthChoice.MEDIUM: 0.7,
    vocab.LengthChoice.DETAILED: 1.0,
}
EDIT_SD = 0.15
CELL_SEPARATOR = "@"  # Between a request type and its platform split.


@dataclasses.dataclass
class Hyper:
    """Hyperparameters of the multilevel length model."""

    mu0: float = math.log(0.6)
    sigma_alpha: float = 0.5
    sigma_gamma: float = 0.3
    sigma_eta: float = 0.3
    tau: float = 0.35
    gamma: dict[str, float] = dataclasses.field(default_factory=dict)

    def gamma_of(self, cell: str) -> float:
        """Returns gamma(t) of the request type behind a length cell."""
        return self.gamma.get(request_type_of(cell), 0.0)


def cell_of(request_type: str, platform: str | None) -> str:
    """Returns the length cell: the request type, split out on mobile."""
    mobile = platform == vocab.Platform.MOBILE
    return (
        f"{request_type}{CELL_SEPARATOR}{platform}" if mobile else request_type
    )


def request_type_of(cell: str) -> str:
    """Returns the request type of a length cell."""
    return cell.partition(CELL_SEPARATOR)[0]


def length_likelihood(
    r_hat: float, offset: float, sigma: float, outcome_probs: list[float]
) -> list[float]:
    """Returns sum_y P(y) l_y(theta) over the grid.

    Args:
        r_hat: The observed log ratio of length to needed words.
        offset: The assistant bias added to theta.
        sigma: The standard deviation of the sufficiency threshold.
        outcome_probs: P(y) for every outcome of reaction.OUTCOMES.

    Returns:
        The likelihood of each grid point.
    """
    probs = dict(zip(reaction.OUTCOMES, outcome_probs))
    p_suff = sum(probs[y] for y in SUFFICIENT)
    p_insuff = sum(probs[y] for y in INSUFFICIENT)
    out = []
    for theta in stats.GRID:
        s = stats.phi((r_hat - theta - offset) / sigma)
        out.append(p_suff * s + p_insuff * (1 - s) + 1 - p_suff - p_insuff)
    return out


def alpha_prior(hyper: Hyper) -> list[float]:
    """Returns the population prior of a user's alpha grid."""
    return stats.normal_grid(hyper.mu0, hyper.sigma_alpha**2)


def new_cell(alpha: list[float], cell: str, hyper: Hyper) -> list[float]:
    """Returns the prior of a cell with no data: from alpha and gamma(t)."""
    mean, var = stats.grid_mean_var(alpha)
    return stats.normal_grid(
        mean + hyper.gamma_of(cell), hyper.sigma_eta**2 + var
    )


def update(
    theta: list[float],
    alpha: list[float],
    cell: str,
    r_hat: float,
    length_var: float,
    bias: tuple[float, float],
    outcome_probs: list[float],
    hyper: Hyper,
) -> tuple[list[float], list[float]]:
    """Applies one turn's censored observation to theta and alpha.

    Args:
        theta: The cell's grid posterior.
        alpha: The user's grid posterior.
        cell: The length cell of the turn.
        r_hat: The observed log ratio of length to needed words.
        length_var: The variance of the fused log length.
        bias: The assistant bias b(c) and its variance.
        outcome_probs: P(y) for every outcome of reaction.OUTCOMES.
        hyper: The hyperparameters.

    Returns:
        The updated theta and alpha grids.
    """
    b, b_var = bias
    var = hyper.tau**2 + length_var + b_var
    theta = stats.grid_update(
        theta, length_likelihood(r_hat, b, math.sqrt(var), outcome_probs)
    )
    alpha_sd = math.sqrt(var + hyper.sigma_eta**2 + hyper.sigma_gamma**2)
    alpha = stats.grid_update(
        alpha,
        length_likelihood(
            r_hat, b + hyper.gamma_of(cell), alpha_sd, outcome_probs
        ),
    )
    return theta, alpha


def edit(theta: list[float], choice: str) -> list[float]:
    """Applies a profile-page length correction, a sharp observation."""
    target = stats.normal_grid(math.log(EDIT_RATIOS[choice]), EDIT_SD**2)
    return stats.grid_update(theta, target)
