from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .posterior import posterior_mmv


@dataclass(frozen=True)
class SBLResult:
    gamma: np.ndarray
    mu: np.ndarray
    iterations: int
    converged: bool
    gamma_history: np.ndarray | None = None
    nmse_history: np.ndarray | None = None


def sbl(
    Theta: np.ndarray,
    Y: np.ndarray,
    noise_var: float,
    gamma_init: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-4,
    keep_history: bool = False,
    Z_true: np.ndarray | None = None,
) -> SBLResult:
    N = Theta.shape[1]
    M = Y.shape[1]
    gamma = np.full(N, float(gamma_init), dtype=float)
    gamma_hist: list[np.ndarray] = []
    nmse_hist: list[float] = []
    converged = False

    for k in range(1, max_iter + 1):
        posterior = posterior_mmv(Theta, Y, noise_var, gamma)
        eta = posterior.Sigma_diag + np.sum(np.abs(posterior.mu) ** 2, axis=1) / float(M)
        gamma_new = np.maximum(eta.real, 0.0)
        if keep_history:
            gamma_hist.append(gamma_new.copy())
            if Z_true is not None:
                denom = max(float(np.linalg.norm(Z_true) ** 2), np.finfo(float).eps)
                nmse_hist.append(float(np.linalg.norm(Z_true - posterior.mu) ** 2 / denom))
        delta = np.linalg.norm(gamma_new - gamma)
        gamma = gamma_new
        if delta <= tol:
            converged = True
            break

    posterior = posterior_mmv(Theta, Y, noise_var, gamma)
    return SBLResult(
        gamma=gamma,
        mu=posterior.mu,
        iterations=k,
        converged=converged,
        gamma_history=np.asarray(gamma_hist) if keep_history else None,
        nmse_history=np.asarray(nmse_hist) if (keep_history and Z_true is not None) else None,
    )
