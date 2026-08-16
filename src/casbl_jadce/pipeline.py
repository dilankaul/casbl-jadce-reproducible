from __future__ import annotations

import platform
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from .activity_model import generate_activity_dataset
from .algorithms.casbl import casbl
from .algorithms.cosamp import mmv_cosamp
from .algorithms.omp import mmv_omp
from .algorithms.sbl import sbl
from .config import load_config
from .correlation import build_C, build_Omega
from .dataset import load_activity_samples, save_activity_samples
from .io import ensure_dir, save_csv, save_json, save_npz, save_yaml
from .metrics import nmse, precision_recall_f1, support_from_gamma, support_from_rows
from .realizations import communication_realization
from .tuning import choose_gamma_threshold


def output_dir(cfg: dict[str, Any]) -> Path:
    return ensure_dir(cfg["run"]["output_dir"])


def generate_activity_stage(cfg: dict[str, Any]) -> None:
    out = output_dir(cfg)
    sys = cfg["system"]; act = cfg["activity"]; run = cfg["run"]
    common = dict(N=sys["N"], V=act["V"], R=sys["R"], kappa=act["kappa"], D=act["D"], S=sys["S"], exact_sparsity=act["exact_sparsity"], max_attempts=act.get("max_attempts", 100000))
    tune = generate_activity_dataset(seed=run["activity_seed"], num_samples=run["num_tuning_samples"], **common)
    # Separate seed stream for final evaluation prevents tuning/test leakage.
    evaluation = generate_activity_dataset(seed=run["activity_seed"] + 1, num_samples=run["num_evaluation_samples"], **common)
    save_activity_samples(out / "activity" / "tuning_activity.npz", tune)
    save_activity_samples(out / "activity" / "evaluation_activity.npz", evaluation)
    save_json(out / "activity" / "summary.json", {
        "tuning_samples": len(tune), "evaluation_samples": len(evaluation),
        "N": sys["N"], "S": sys["S"], "V": act["V"], "R": sys["R"], "kappa": act["kappa"], "D": act["D"],
    })


def communication_stage(cfg: dict[str, Any]) -> None:
    """Save a compact reproducibility manifest and one numerical preview.

    Full Y arrays are intentionally regenerated deterministically from seeds instead of
    storing hundreds of MB in Git.
    """
    out = output_dir(cfg); sys = cfg["system"]; run = cfg["run"]; tune = cfg["tuning"]
    activities = load_activity_samples(out / "activity" / "tuning_activity.npz")
    sample = communication_realization(activities[0], run["communication_seed"], 0, sys["M"], tune["reference_pilot_length"], tune["reference_snr_db"])
    save_npz(out / "communication" / "preview.npz", a=sample.a, H=sample.H, Z=sample.Z, Theta=sample.Theta, W=sample.W, Y=sample.Y, noise_var=sample.noise_var)
    save_json(out / "communication" / "manifest.json", {
        "communication_seed": run["communication_seed"], "M": sys["M"], "pilot_lengths": sys["pilot_lengths"], "snr_db": sys["snr_db"],
        "rule": "H keyed by sample; Theta keyed by sample and L; W keyed by sample, L and SNR",
    })


def correlation_stage(cfg: dict[str, Any]) -> None:
    out = output_dir(cfg); corr = cfg["correlation"]
    activities = load_activity_samples(out / "activity" / "tuning_activity.npz")
    C = build_C(activities[0].device_locations, rho=corr["rho"], U=corr["U"])
    save_npz(out / "correlation" / "preview_C.npz", C=C)
    save_json(out / "correlation" / "summary.json", {
        "rho": corr["rho"], "U": corr["U"], "C_min": float(C.min()), "C_max": float(C.max()), "C_mean": float(C.mean())
    })


def _tune_pair(args: tuple[Any, ...]) -> dict[str, Any]:
    activities, master_seed, M, L, snr_db, rho, U, alpha, beta, alg = args
    gammas = [];
    nmse_values = [];
    iterations = []
    for i, activity in enumerate(activities):
        r = communication_realization(activity, master_seed, i, M, L, snr_db)
        C = build_C(activity.device_locations, rho=rho, U=U)
        result = casbl(r.Theta, r.Y, r.noise_var, C, alpha=alpha, beta=beta, gamma_init=alg["gamma_init"], max_iter=alg["sbl_max_iter"], tol=alg["sbl_tol"])
        gammas.append(result.gamma); nmse_values.append(nmse(r.Z, result.mu)); iterations.append(result.iterations)
    gammas = np.asarray(gammas)
    a_true = np.stack([a.a for a in activities])
    threshold = choose_gamma_threshold(a_true, gammas, num_thresholds=alg["num_thresholds"])
    return {
        "alpha": float(alpha), "beta": float(beta), "tau": threshold.tau,
        "precision": threshold.mean_precision, "recall": threshold.mean_recall, "f1": threshold.mean_f1,
        "nmse": float(np.mean(nmse_values)), "iterations": float(np.mean(iterations)),
    }


def tune_stage(cfg: dict[str, Any], workers: int = 1) -> None:
    out = output_dir(cfg); sys = cfg["system"]; corr = cfg["correlation"]; tune = cfg["tuning"]; run = cfg["run"]; algcfg = cfg["algorithms"].copy()
    algcfg["num_thresholds"] = tune["num_thresholds"]
    activities = load_activity_samples(out / "activity" / "tuning_activity.npz")
    tasks = [
        (activities, run["communication_seed"], sys["M"], tune["reference_pilot_length"], tune["reference_snr_db"], corr["rho"], corr["U"], alpha, beta, algcfg)
        for alpha in tune["alpha_values"] for beta in tune["beta_values"]
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex: rows = list(ex.map(_tune_pair, tasks))
    else:
        rows = [_tune_pair(t) for t in tasks]
    rows.sort(key=lambda x: (x["f1"], -x["nmse"], x["recall"]), reverse=True)
    save_csv(out / "tuning" / "alpha_beta.csv", rows)
    best = rows[0]

    # Tune SBL's threshold independently on the same tuning realization set.
    sbl_gammas = []
    for i, activity in enumerate(activities):
        r = communication_realization(activity, run["communication_seed"], i, sys["M"], tune["reference_pilot_length"], tune["reference_snr_db"])
        result = sbl(r.Theta, r.Y, r.noise_var, gamma_init=algcfg["gamma_init"], max_iter=algcfg["sbl_max_iter"], tol=algcfg["sbl_tol"])
        sbl_gammas.append(result.gamma)
    sbl_threshold = choose_gamma_threshold(np.stack([a.a for a in activities]), np.asarray(sbl_gammas), tune["num_thresholds"])

    selected = {
        "casbl": {"alpha": best["alpha"], "beta": best["beta"], "tau": best["tau"], "tuning_f1": best["f1"], "tuning_nmse": best["nmse"]},
        "sbl": {"tau": sbl_threshold.tau, "tuning_f1": sbl_threshold.mean_f1},
        "reference_snr_db": tune["reference_snr_db"], "reference_pilot_length": tune["reference_pilot_length"],
    }
    save_json(out / "tuning" / "selected.json", selected)


def evaluation_conditions(cfg: dict[str, Any]) -> list[tuple[int, float, str]]:
    sys = cfg["system"]; tune = cfg["tuning"]
    L0 = int(tune["reference_pilot_length"]); snr0 = float(tune["reference_snr_db"])
    conditions: list[tuple[int, float, str]] = []
    for snr in sys["snr_db"]: conditions.append((L0, float(snr), "snr_sweep"))
    for L in sys["pilot_lengths"]:
        # Keep the reference point in both sweeps so each figure has a complete axis.
        conditions.append((int(L), snr0, "pilot_sweep"))
    return conditions


def _evaluate_sample(args: tuple[Any, ...]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray] | None]:
    sample_index, activity, master_seed, sys, corr, alg, selected, conditions, save_gamma = args
    rows: list[dict[str, Any]] = []
    gamma_save: dict[str, np.ndarray] | None = {} if save_gamma else None
    for L, snr_db, sweep in conditions:
        r = communication_realization(activity, master_seed, sample_index, sys["M"], L, snr_db)
        C = build_C(activity.device_locations, rho=corr["rho"], U=corr["U"])

        t0 = time.perf_counter()
        ca = casbl(r.Theta, r.Y, r.noise_var, C, alpha=selected["casbl"]["alpha"], beta=selected["casbl"]["beta"], gamma_init=alg["gamma_init"], max_iter=alg["sbl_max_iter"], tol=alg["sbl_tol"])
        ca_runtime = time.perf_counter() - t0
        ca_support = support_from_gamma(ca.gamma, selected["casbl"]["tau"])
        p, rec, f = precision_recall_f1(r.a, ca_support)
        rows.append({"sample": sample_index, "algorithm": "CA-SBL-ANC", "L": L, "snr_db": snr_db, "sweep": sweep, "precision": p, "recall": rec, "f1": f, "nmse": nmse(r.Z, ca.mu), "iterations": ca.iterations, "runtime_s": ca_runtime})

        t0 = time.perf_counter()
        sb = sbl(r.Theta, r.Y, r.noise_var, gamma_init=alg["gamma_init"], max_iter=alg["sbl_max_iter"], tol=alg["sbl_tol"])
        sb_runtime = time.perf_counter() - t0
        sb_support = support_from_gamma(sb.gamma, selected["sbl"]["tau"])
        p, rec, f = precision_recall_f1(r.a, sb_support)
        rows.append({"sample": sample_index, "algorithm": "SBL", "L": L, "snr_db": snr_db, "sweep": sweep, "precision": p, "recall": rec, "f1": f, "nmse": nmse(r.Z, sb.mu), "iterations": sb.iterations, "runtime_s": sb_runtime})

        t0 = time.perf_counter(); omp = mmv_omp(r.Theta, r.Y, sys["S"]); omp_runtime = time.perf_counter() - t0
        p, rec, f = precision_recall_f1(r.a, support_from_rows(omp, sys["S"]))
        rows.append({"sample": sample_index, "algorithm": "MMV-OMP", "L": L, "snr_db": snr_db, "sweep": sweep, "precision": p, "recall": rec, "f1": f, "nmse": nmse(r.Z, omp), "iterations": sys["S"], "runtime_s": omp_runtime})

        t0 = time.perf_counter(); co = mmv_cosamp(r.Theta, r.Y, sys["S"], max_iter=alg["cosamp_max_iter"], tol=alg["cosamp_tol"]); co_runtime = time.perf_counter() - t0
        p, rec, f = precision_recall_f1(r.a, support_from_rows(co, sys["S"]))
        rows.append({"sample": sample_index, "algorithm": "MMV-CoSaMP", "L": L, "snr_db": snr_db, "sweep": sweep, "precision": p, "recall": rec, "f1": f, "nmse": nmse(r.Z, co), "iterations": np.nan, "runtime_s": co_runtime})

        if save_gamma and L == selected["reference_pilot_length"] and float(snr_db) == float(selected["reference_snr_db"]):
            gamma_save = {"a": r.a.astype(np.uint8), "casbl_gamma": ca.gamma, "sbl_gamma": sb.gamma}
    return rows, gamma_save


def evaluate_stage(cfg: dict[str, Any], workers: int = 1) -> None:
    import pandas as pd
    out = output_dir(cfg); sys = cfg["system"]; corr = cfg["correlation"]; alg = cfg["algorithms"]; run = cfg["run"]
    activities = load_activity_samples(out / "activity" / "evaluation_activity.npz")
    import json
    with (out / "tuning" / "selected.json").open("r", encoding="utf-8") as f: selected = json.load(f)
    selected["reference_snr_db"] = cfg["tuning"]["reference_snr_db"]
    selected["reference_pilot_length"] = cfg["tuning"]["reference_pilot_length"]
    conditions = evaluation_conditions(cfg)
    n_gamma = min(int(cfg["evaluation"].get("gamma_distribution_samples", 0)), len(activities))
    tasks = [(i, a, run["communication_seed"] + 100000, sys, corr, alg, selected, conditions, i < n_gamma) for i, a in enumerate(activities)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex: outputs = list(ex.map(_evaluate_sample, tasks))
    else:
        outputs = [_evaluate_sample(t) for t in tasks]
    rows = [row for sample_rows, _ in outputs for row in sample_rows]
    save_csv(out / "evaluation" / "per_sample.csv", rows)
    df = pd.DataFrame(rows)
    aggregate = df.groupby(["algorithm", "L", "snr_db", "sweep"], as_index=False)[["precision", "recall", "f1", "nmse", "runtime_s"]].mean()
    ensure_dir(out / "evaluation"); aggregate.to_csv(out / "evaluation" / "aggregate.csv", index=False)

    gamma_outputs = [g for _, g in outputs if g is not None]
    if gamma_outputs:
        save_npz(out / "evaluation" / "gamma_distribution.npz",
                 a=np.stack([g["a"] for g in gamma_outputs]),
                 casbl_gamma=np.stack([g["casbl_gamma"] for g in gamma_outputs]),
                 sbl_gamma=np.stack([g["sbl_gamma"] for g in gamma_outputs]))


def convergence_stage(cfg: dict[str, Any]) -> None:
    import json
    out = output_dir(cfg); sys = cfg["system"]; corr = cfg["correlation"]; alg = cfg["algorithms"]; run = cfg["run"]; tune = cfg["tuning"]
    activities = load_activity_samples(out / "activity" / "evaluation_activity.npz")
    idx = int(cfg["evaluation"].get("convergence_sample_offset", 0))
    activity = activities[idx]
    with (out / "tuning" / "selected.json").open("r", encoding="utf-8") as f: selected = json.load(f)
    r = communication_realization(activity, run["communication_seed"] + 100000, idx, sys["M"], tune["reference_pilot_length"], tune["reference_snr_db"])
    C = build_C(activity.device_locations, rho=corr["rho"], U=corr["U"])
    ca = casbl(r.Theta, r.Y, r.noise_var, C, selected["casbl"]["alpha"], selected["casbl"]["beta"], gamma_init=alg["gamma_init"], max_iter=alg["sbl_max_iter"], tol=alg["sbl_tol"], keep_history=True, Z_true=r.Z)
    sb = sbl(r.Theta, r.Y, r.noise_var, gamma_init=alg["gamma_init"], max_iter=alg["sbl_max_iter"], tol=alg["sbl_tol"], keep_history=True, Z_true=r.Z)
    save_npz(out / "convergence" / "convergence.npz", casbl_gamma_history=ca.gamma_history, casbl_phi_history=ca.phi_history, casbl_nmse_history=ca.nmse_history, sbl_gamma_history=sb.gamma_history, sbl_nmse_history=sb.nmse_history)


def manifest_stage(cfg: dict[str, Any]) -> None:
    out = output_dir(cfg)
    save_yaml(out / "config.yaml", cfg)
    save_json(out / "manifest.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "algorithm_theory": "complex-MMV: -M sum(log gamma + eta/gamma), phi=(Omega gamma)/M",
    })


def run_all(config_path: str, workers: int = 1) -> None:
    cfg = load_config(config_path)
    manifest_stage(cfg)
    generate_activity_stage(cfg)
    communication_stage(cfg)
    correlation_stage(cfg)
    tune_stage(cfg, workers=workers)
    evaluate_stage(cfg, workers=workers)
    convergence_stage(cfg)
