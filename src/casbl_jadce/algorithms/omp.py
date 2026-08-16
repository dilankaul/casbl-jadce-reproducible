from __future__ import annotations
import numpy as np
from scipy.linalg import lstsq


def mmv_omp(Theta: np.ndarray, Y: np.ndarray, S: int) -> np.ndarray:
    """MMV-OMP/SOMP using row-energy selection and known joint sparsity S."""
    L, N = Theta.shape
    if not (0 < S <= min(L, N)):
        raise ValueError("S must satisfy 0 < S <= min(L, N).")
    residual = Y.copy()
    support: list[int] = []
    Xs = np.empty((0, Y.shape[1]), dtype=np.complex128)
    for _ in range(S):
        proxy = Theta.conj().T @ residual
        energy = np.sum(np.abs(proxy) ** 2, axis=1)
        if support:
            energy[np.asarray(support)] = -np.inf
        support.append(int(np.argmax(energy)))
        Xs, *_ = lstsq(Theta[:, support], Y, lapack_driver="gelsd")
        residual = Y - Theta[:, support] @ Xs
    Z_hat = np.zeros((N, Y.shape[1]), dtype=np.complex128)
    Z_hat[np.asarray(support), :] = Xs
    return Z_hat
