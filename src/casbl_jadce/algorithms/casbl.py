from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..correlation import build_Omega, interaction_phi
from .posterior import posterior_mmv


@dataclass(frozen=True)
class CASBLResult:
    gamma: np.ndarray
    mu: np.ndarray
    iterations: int
    converged: bool
    gamma_history: np.ndarray | None = None
    phi_history: np.ndarray | None = None
    nmse_history: np.ndarray | None = None


def stable_gamma_update(eta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Numerically stable positive root, with SBL fallback for phi <= 0.

    For phi > 0:
        gamma = 2 eta / (sqrt(1 + 4 phi eta) + 1)
    For phi <= 0:
        gamma = eta
    """
    eta = np.maximum(np.asarray(eta, dtype=float), 0.0)
    phi = np.asarray(phi, dtype=float)
    if eta.shape != phi.shape:
        raise ValueError("eta and phi must have the same shape.")
    gamma_new = eta.copy()
    mask = phi > 0.0
    radicand = 1.0 + 4.0 * phi[mask] * eta[mask]
    gamma_new[mask] = 2.0 * eta[mask] / (np.sqrt(radicand) + 1.0)
    return np.maximum(gamma_new, 0.0)


def casbl(
    Theta: np.ndarray,
    Y: np.ndarray,
    noise_var: float,
    C: np.ndarray,
    alpha: float,
    beta: float,
    gamma_init: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-4,
    keep_history: bool = False,
    Z_true: np.ndarray | None = None,
) -> CASBLResult:
    """Correlation-aware SBL with corrected complex-MMV scaling.

    eta_i = Sigma_ii + ||mu_i||^2/M
    Omega = alpha (beta * 1 - C)
    phi = (Omega gamma)/M
    """
    N = Theta.shape[1]
    M = Y.shape[1]
    if C.shape != (N, N):
        raise ValueError("C must have shape (N, N).")

    gamma = np.full(N, float(gamma_init), dtype=float)
    gamma_hist: list[np.ndarray] = []
    phi_hist: list[np.ndarray] = []
    nmse_hist: list[float] = []
    converged = False

    for k in range(1, max_iter + 1):
        posterior = posterior_mmv(Theta, Y, noise_var, gamma)
        eta = posterior.Sigma_diag + np.sum(np.abs(posterior.mu) ** 2, axis=1) / float(M)
        phi = interaction_phi(C, gamma, alpha=alpha, beta=beta, M=M)
        gamma_new = stable_gamma_update(eta, phi)

        if keep_history:
            gamma_hist.append(gamma_new.copy())
            phi_hist.append(phi.copy())
            if Z_true is not None:
                denom = max(float(np.linalg.norm(Z_true) ** 2), np.finfo(float).eps)
                nmse_hist.append(float(np.linalg.norm(Z_true - posterior.mu) ** 2 / denom))

        delta = np.linalg.norm(gamma_new - gamma)
        gamma = gamma_new
        if delta <= tol:
            converged = True
            break

    posterior = posterior_mmv(Theta, Y, noise_var, gamma)
    return CASBLResult(
        gamma=gamma,
        mu=posterior.mu,
        iterations=k,
        converged=converged,
        gamma_history=np.asarray(gamma_hist) if keep_history else None,
        phi_history=np.asarray(phi_hist) if keep_history else None,
        nmse_history=np.asarray(nmse_hist) if (keep_history and Z_true is not None) else None,
    )


def explicit_Omega(C: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return build_Omega(C, alpha=alpha, beta=beta)
