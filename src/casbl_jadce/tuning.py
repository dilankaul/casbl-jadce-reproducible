from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .metrics import precision_recall_f1, support_from_gamma


@dataclass(frozen=True)
class ThresholdChoice:
    tau: float
    mean_precision: float
    mean_recall: float
    mean_f1: float


def threshold_candidates(gammas: np.ndarray, num_thresholds: int) -> np.ndarray:
    values = np.asarray(gammas, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([0.0])
    vmax = float(np.max(values))
    if vmax <= 0:
        return np.array([0.0])
    # Quantile candidates adapt automatically now that gamma is not clipped to 1.
    q = np.linspace(0.0, 1.0, int(num_thresholds))
    candidates = np.unique(np.quantile(values, q))
    return np.unique(np.concatenate(([0.0], candidates, [np.nextafter(vmax, np.inf)])))


def choose_gamma_threshold(a_true: np.ndarray, gammas: np.ndarray, num_thresholds: int = 80) -> ThresholdChoice:
    a_true = np.asarray(a_true, dtype=bool)
    gammas = np.asarray(gammas, dtype=float)
    if a_true.shape != gammas.shape:
        raise ValueError("a_true and gammas must have matching (samples, N) shapes.")
    best: ThresholdChoice | None = None
    for tau in threshold_candidates(gammas, num_thresholds):
        metrics = [precision_recall_f1(a_true[i], support_from_gamma(gammas[i], float(tau))) for i in range(len(a_true))]
        p, r, f = np.mean(metrics, axis=0)
        choice = ThresholdChoice(float(tau), float(p), float(r), float(f))
        if best is None or (choice.mean_f1, choice.mean_recall, choice.mean_precision) > (best.mean_f1, best.mean_recall, best.mean_precision):
            best = choice
    assert best is not None
    return best
