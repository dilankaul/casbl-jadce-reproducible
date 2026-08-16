from __future__ import annotations
import numpy as np


def support_from_gamma(gamma: np.ndarray, tau: float) -> np.ndarray:
    return np.asarray(gamma) >= float(tau)


def support_from_rows(Z_hat: np.ndarray, S: int) -> np.ndarray:
    energy = np.sum(np.abs(Z_hat) ** 2, axis=1)
    support = np.zeros(len(energy), dtype=bool)
    support[np.argpartition(energy, -S)[-S:]] = True
    return support


def detection_counts(a_true: np.ndarray, a_hat: np.ndarray) -> tuple[int, int, int, int]:
    a_true = np.asarray(a_true, dtype=bool); a_hat = np.asarray(a_hat, dtype=bool)
    tp = int(np.sum(a_true & a_hat)); fp = int(np.sum(~a_true & a_hat))
    fn = int(np.sum(a_true & ~a_hat)); tn = int(np.sum(~a_true & ~a_hat))
    return tp, fp, fn, tn


def precision_recall_f1(a_true: np.ndarray, a_hat: np.ndarray) -> tuple[float, float, float]:
    tp, fp, fn, _ = detection_counts(a_true, a_hat)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def nmse(Z_true: np.ndarray, Z_hat: np.ndarray) -> float:
    denom = float(np.linalg.norm(Z_true) ** 2)
    if denom == 0.0:
        return 0.0 if np.linalg.norm(Z_hat) == 0.0 else float("inf")
    return float(np.linalg.norm(Z_true - Z_hat) ** 2 / denom)
