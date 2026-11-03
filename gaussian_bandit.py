"""Gaussian Thompson sampling with known noise sigma.

Model:   R | A=i ~ Normal(mu_i, sigma^2), sigma known, mu_i unknown
Prior:   mu_i ~ Normal(0, 25) for every alternative
Update:  precision-weighted Normal-Normal conjugate update
"""

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

PRIOR_MEAN = 0.0
PRIOR_VAR = 25.0


@dataclass
class GaussianBanditResult:
    true_means: List[float]
    sigma: float
    n_rounds: int
    seed: int
    post_means: List[float]
    post_vars: List[float]
    counts: List[int]
    total_reward: float
    recommendation: int
    rewards: List[float] = field(default_factory=list)
    choices: List[int] = field(default_factory=list)


def validate_inputs(sigma: float, n_rounds) -> None:
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a positive number")
    if float(n_rounds) != int(n_rounds) or int(n_rounds) < 1:
        raise ValueError("number of rounds must be a positive integer")


def run_gaussian_bandit(
    true_means: Sequence[float], sigma: float, n_rounds: int, seed: int
) -> GaussianBanditResult:
    validate_inputs(sigma, n_rounds)
    n_rounds = int(n_rounds)

    rng = np.random.default_rng(int(seed))
    mu = np.asarray(true_means, dtype=float)  
    k = len(mu)

    reward_prec = 1.0 / sigma**2
    post_mean = np.full(k, PRIOR_MEAN, dtype=float)
    post_prec = np.full(k, 1.0 / PRIOR_VAR, dtype=float)
    counts = np.zeros(k, dtype=int)
    rewards: List[float] = []
    choices: List[int] = []

    for _ in range(n_rounds):
        # 1) sample a plausible mean from each current posterior
        samples = rng.normal(post_mean, 1.0 / np.sqrt(post_prec))
        # 2) pick the largest sample (learned beliefs only, never mu)
        a = int(np.argmax(samples))
        # 3) simulator draws one continuous reward from the TRUE mean
        r = float(rng.normal(mu[a], sigma))
        # 4) update ONLY the selected alternative
        new_prec = post_prec[a] + reward_prec
        post_mean[a] = (post_prec[a] * post_mean[a] + reward_prec * r) / new_prec
        post_prec[a] = new_prec

        counts[a] += 1
        rewards.append(r)
        choices.append(a)

    return GaussianBanditResult(
        true_means=mu.tolist(),
        sigma=float(sigma),
        n_rounds=n_rounds,
        seed=int(seed),
        post_means=post_mean.tolist(),
        post_vars=(1.0 / post_prec).tolist(),
        counts=counts.tolist(),
        total_reward=float(np.sum(rewards)),
        recommendation=int(np.argmax(post_mean)),
        rewards=rewards,
        choices=choices,
    )
