from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ActivitySample:
    a: np.ndarray
    device_locations: np.ndarray
    event_locations: np.ndarray
    activation_probabilities: np.ndarray


def uniform_disk(rng: np.random.Generator, count: int, R: float) -> np.ndarray:
    r = float(R) * np.sqrt(rng.random(count))
    theta = 2.0 * np.pi * rng.random(count)
    return np.column_stack((r * np.cos(theta), r * np.sin(theta)))


def event_activation_probabilities(device_locations: np.ndarray, event_locations: np.ndarray, kappa: float, D: float) -> np.ndarray:
    delta = device_locations[:, None, :] - event_locations[None, :, :]
    d = np.sqrt(np.sum(delta * delta, axis=2))
    if kappa <= 0:
        p_iv = (d == 0.0).astype(float)
    else:
        denom = 1.0 - np.exp(-D / kappa)
        raw = (np.exp(-d / kappa) - np.exp(-D / kappa)) / denom
        p_iv = np.where(d <= D, raw, 0.0)
        p_iv = np.clip(p_iv, 0.0, 1.0)
    return 1.0 - np.prod(1.0 - p_iv, axis=1)


def generate_activity_sample(
    rng: np.random.Generator,
    N: int,
    V: int,
    R: float,
    kappa: float,
    D: float,
    S: int,
    exact_sparsity: bool = True,
    max_attempts: int = 100000,
) -> ActivitySample:
    for _ in range(max_attempts):
        device_locations = uniform_disk(rng, N, R)
        event_locations = uniform_disk(rng, V, R)
        p = event_activation_probabilities(device_locations, event_locations, kappa, D)
        a = rng.random(N) < p
        if (not exact_sparsity) or int(np.sum(a)) == int(S):
            return ActivitySample(a=a, device_locations=device_locations, event_locations=event_locations, activation_probabilities=p)
    raise RuntimeError("Could not obtain requested sparsity. Check activity parameters.")


def generate_activity_dataset(
    seed: int,
    num_samples: int,
    N: int,
    V: int,
    R: float,
    kappa: float,
    D: float,
    S: int,
    exact_sparsity: bool = True,
    max_attempts: int = 100000,
) -> list[ActivitySample]:
    seq = np.random.SeedSequence(seed)
    return [
        generate_activity_sample(
            np.random.default_rng(child), N=N, V=V, R=R, kappa=kappa, D=D,
            S=S, exact_sparsity=exact_sparsity, max_attempts=max_attempts,
        )
        for child in seq.spawn(num_samples)
    ]
