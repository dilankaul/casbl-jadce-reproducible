from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .activity_model import ActivitySample, event_activation_probabilities
from .dataset import load_activity_samples
from .io import ensure_dir


def figure_formats(mode: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(mode, str):
        mode = mode.lower()
        if mode == "none": return ()
        if mode == "both": return ("png", "pdf")
        if mode in {"png", "pdf"}: return (mode,)
        raise ValueError("figure format must be none, png, pdf, or both")
    return tuple(mode)


def _save(fig: plt.Figure, stem: Path, formats: str | Iterable[str] = "both") -> list[Path]:
    paths: list[Path] = []
    ensure_dir(stem.parent)
    for ext in figure_formats(formats):
        path = Path(f"{stem}.{ext}")
        kwargs = {"bbox_inches": "tight"}
        if ext == "png": kwargs["dpi"] = 250
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def activity_probability_grid(
    event_locations: np.ndarray,
    R: float,
    D: float,
    kappa: float,
    resolution: int = 301,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    """Evaluate the paper's combined activation probability over the cell.

    At every spatial grid point x, the plotted field is exactly
    P(x) = 1 - prod_v(1 - P_v(x)). Points outside the BS cell are masked.
    """
    resolution = max(51, int(resolution))
    x = np.linspace(-float(R), float(R), resolution)
    y = np.linspace(-float(R), float(R), resolution)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack((X.ravel(), Y.ravel()))
    P = event_activation_probabilities(points, np.asarray(event_locations), kappa=float(kappa), D=float(D)).reshape(X.shape)
    outside = X * X + Y * Y > float(R) ** 2
    return X, Y, np.ma.array(P, mask=outside)


def plot_activity_sample(
    sample: ActivitySample,
    R: float,
    D: float,
    stem: Path,
    formats: str = "both",
    show_event_radius: bool = True,
    title: str | None = None,
    kappa: float | None = None,
    show_probability_field: bool = False,
    probability_resolution: int = 301,
    probability_alpha: float = 0.55,
    probability_levels: int = 128,
) -> list[Path]:
    """Plot one activity sample without requiring Task 02 output files."""
    loc = sample.device_locations
    active = sample.a.astype(bool)

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    if show_probability_field:
        if kappa is None:
            raise ValueError("kappa is required when show_probability_field=True.")
        X, Y, P = activity_probability_grid(
            sample.event_locations, R=R, D=D, kappa=kappa, resolution=probability_resolution
        )

        # Plot the calculated probability field without modifying P. The only
        # visualization mapping is the colormap: P=0 is pure white and P=1 is
        # dark red. The mask in P applies only outside the physical BS cell.
        n_levels = max(8, int(probability_levels))
        levels = np.linspace(0.0, 1.0, n_levels)
        probability_cmap = LinearSegmentedColormap.from_list(
            "activation_probability_white_red",
            [
                (0.00, "#ffffff"),
                (0.20, "#fee5d9"),
                (0.40, "#fcae91"),
                (0.60, "#fb6a4a"),
                (0.80, "#cb181d"),
                (1.00, "#67000d"),
            ],
            N=256,
        )
        probability_norm = Normalize(vmin=0.0, vmax=1.0)
        image = ax.contourf(
            X, Y, P, levels=levels, cmap=probability_cmap, norm=probability_norm,
            alpha=float(probability_alpha), antialiased=True, zorder=0,
        )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=r"Combined activation probability $P(\mathbf{x})$")

    if np.any(~active):
        ax.scatter(loc[~active, 0], loc[~active, 1], marker="o", s=18, alpha=0.75, label="Inactive MTD", zorder=3)
    if np.any(active):
        ax.scatter(loc[active, 0], loc[active, 1], marker="o", s=52, label="Active MTD", zorder=4)
    ax.scatter(sample.event_locations[:, 0], sample.event_locations[:, 1], marker="*", s=150, label="Event", zorder=5)
    ax.scatter([0.0], [0.0], marker="x", s=90, label="BS", zorder=5)
    ax.add_patch(Circle((0.0, 0.0), R, fill=False, linestyle="--", linewidth=1.2, zorder=6))
    if show_event_radius:
        first = True
        for xy in sample.event_locations:
            ax.add_patch(Circle(tuple(xy), D, fill=False, linestyle="--", linewidth=0.8, alpha=0.75, zorder=2, label="Event cutoff radius D" if first else None))
            first = False
    ax.set_aspect("equal", adjustable="box")
    pad = 1.08 * R
    ax.set_xlim(-pad, pad); ax.set_ylim(-pad, pad)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.2); ax.legend(loc="best")
    fig.tight_layout()
    return _save(fig, stem, formats)


def _saved_activity_parameters(cfg: dict) -> tuple[float, float, float]:
    """Use parameters recorded by Task 02 when available.

    This prevents a misleading plot if the config is edited after activity data
    were generated. Old files without a summary fall back to the config.
    """
    out = Path(cfg["run"]["output_dir"])
    summary_path = out / "activity" / "summary.json"
    if summary_path.exists():
        import json
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        return (
            float(summary.get("R", cfg["system"]["R"])),
            float(summary.get("D", cfg["activity"]["D"])),
            float(summary.get("kappa", cfg["activity"]["kappa"])),
        )
    return float(cfg["system"]["R"]), float(cfg["activity"]["D"]), float(cfg["activity"]["kappa"])


def plot_activity_realization(
    cfg: dict,
    split: str = "tuning",
    sample_index: int = 0,
    formats: str = "both",
    show_event_radius: bool = True,
    show_probability_field: bool | None = None,
) -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    if split not in {"tuning", "evaluation"}:
        raise ValueError("split must be 'tuning' or 'evaluation'.")
    samples = load_activity_samples(out / "activity" / f"{split}_activity.npz")
    if not 0 <= sample_index < len(samples):
        raise IndexError(f"sample_index must be in [0, {len(samples)-1}].")
    sample = samples[sample_index]
    R, D, kappa = _saved_activity_parameters(cfg)
    fig_cfg = cfg.get("figures", {})
    if show_probability_field is None:
        show_probability_field = bool(fig_cfg.get("show_probability_field", False))
    suffix = "_probability_field" if show_probability_field else ""
    return plot_activity_sample(
        sample,
        R=R,
        D=D,
        kappa=kappa,
        stem=out / "figures" / "tasks" / f"02_activity_realization_{split}_{sample_index:04d}{suffix}",
        formats=formats,
        show_event_radius=show_event_radius,
        show_probability_field=show_probability_field,
        probability_resolution=int(fig_cfg.get("probability_resolution", 301)),
        probability_alpha=float(fig_cfg.get("probability_alpha", 0.55)),
        probability_levels=int(fig_cfg.get("probability_levels", 128)),
        title=f"Selected {split} realization #{sample_index}: {int(np.sum(sample.a))} active MTDs",
    )



def plot_activity_s_distribution(cfg: dict, formats: str = "both") -> list[Path]:
    """Plot the realized support-size distribution using saved Task 02 data."""
    out = Path(cfg["run"]["output_dir"])
    tuning_path = out / "activity" / "tuning_activity.npz"
    evaluation_path = out / "activity" / "evaluation_activity.npz"
    if not tuning_path.exists() or not evaluation_path.exists():
        raise FileNotFoundError("Task 02 tuning/evaluation activity files are required.")
    tuning = load_activity_samples(tuning_path)
    evaluation = load_activity_samples(evaluation_path)
    tuning_s = np.asarray([np.sum(sample.a) for sample in tuning], dtype=int)
    evaluation_s = np.asarray([np.sum(sample.a) for sample in evaluation], dtype=int)
    all_counts = np.concatenate((tuning_s, evaluation_s))
    lo = int(np.min(all_counts)); hi = int(np.max(all_counts))
    bins = np.arange(lo - 0.5, hi + 1.5, 1.0)
    target = float(cfg["system"]["S"])

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.hist(tuning_s, bins=bins, alpha=0.55, density=True, label=f"Tuning (mean={np.mean(tuning_s):.2f})")
    ax.hist(evaluation_s, bins=bins, alpha=0.55, density=True, label=f"Evaluation (mean={np.mean(evaluation_s):.2f})")
    ax.axvline(target, linestyle="--", linewidth=1.0, label=f"Target mean S={target:g}")
    ax.set_xlabel("Realized active MTD count S")
    ax.set_ylabel("Relative frequency")
    ax.set_title("Distribution of realized event-driven activity")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return _save(fig, out / "figures" / "tasks" / "02_activity_realized_s_distribution", formats)

def plot_activity_calibration(cfg: dict, formats: str = "both") -> list[Path]:
    """Plot mean active count versus kappa from the Task 01 calibration files."""
    import json
    out = Path(cfg["run"]["output_dir"])
    path = out / "activity_calibration" / "kappa_sweep.csv"
    summary_path = out / "activity_calibration" / "summary.json"
    df = pd.read_csv(path).sort_values("kappa")
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        target = float(summary["target_mean_active"])
        D = float(summary["D"])
    else:
        target = float(cfg["system"]["S"])
        D = float(cfg["activity"]["D"])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(df["kappa"], df["mean_expected_active"], marker="o", label=r"Mean $\sum_i P_i$")
    ax.plot(df["kappa"], df["mean_realized_active"], marker="s", markersize=4, label="Mean realized active MTDs")
    ax.axhline(target, linestyle="--", linewidth=1.0, label=f"Target mean S={target:g}")
    best = df.iloc[int(np.argmin(np.abs(df["mean_expected_active"].to_numpy() - target)))]
    ax.axvline(best["kappa"], linestyle=":", linewidth=1.0, label=fr"Closest grid $\kappa$={best['kappa']:.4g}")
    ax.set_xlabel(r"Decay parameter $\kappa$")
    ax.set_ylabel("Average active MTD count")
    ax.set_title(fr"Activity calibration for $D={D:g}$ m")
    ax.grid(True, alpha=0.2); ax.legend()
    fig.tight_layout()
    return _save(fig, out / "figures" / "tasks" / "01_activity_kappa_calibration", formats)


def plot_communication_preview(cfg: dict, formats: str = "both") -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    data = np.load(out / "communication" / "preview.npz")
    Theta = data["Theta"]
    gram = np.abs(Theta.conj().T @ Theta)
    np.fill_diagonal(gram, 0.0)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    image = ax.imshow(gram, aspect="auto", interpolation="nearest")
    fig.colorbar(image, ax=ax, label=r"$|\theta_i^H\theta_j|$")
    ax.set_xlabel("Pilot index j"); ax.set_ylabel("Pilot index i")
    ax.set_title("Pilot cross-correlation magnitude (diagonal removed)")
    fig.tight_layout()
    return _save(fig, out / "figures" / "tasks" / "03_pilot_cross_correlation", formats)


def plot_correlation_preview(cfg: dict, formats: str = "both") -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    C = np.load(out / "correlation" / "preview_C.npz")["C"]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    image = ax.imshow(C, aspect="auto", interpolation="nearest")
    fig.colorbar(image, ax=ax, label=r"$C_{ij}$")
    ax.set_xlabel("MTD j"); ax.set_ylabel("MTD i"); ax.set_title("Spatial correlation matrix C")
    fig.tight_layout()
    return _save(fig, out / "figures" / "tasks" / "04_correlation_matrix", formats)


def plot_alpha_beta_tuning(cfg: dict, formats: str = "both") -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    df = pd.read_csv(out / "tuning" / "alpha_beta.csv")
    pivot = df.pivot(index="alpha", columns="beta", values="f1").sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    image = ax.imshow(pivot.values, aspect="auto", origin="lower")
    fig.colorbar(image, ax=ax, label="Mean tuning F1")
    ax.set_xticks(np.arange(len(pivot.columns)), [f"{x:g}" for x in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), [f"{x:g}" for x in pivot.index])
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$\alpha$"); ax.set_title(r"CA-SBL $\alpha$-$\beta$ tuning")
    fig.tight_layout()
    return _save(fig, out / "figures" / "tasks" / "05_alpha_beta_f1", formats)


def plot_threshold_tuning(cfg: dict, formats: str = "both") -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    paths: list[Path] = []
    for filename, label, stem in [
        ("casbl_thresholds.csv", "CA-SBL-ANC", "06_casbl_threshold"),
        ("sbl_thresholds.csv", "SBL", "06_sbl_threshold"),
    ]:
        df = pd.read_csv(out / "tuning" / filename).sort_values("tau")
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(df["tau"], df["f1"], marker="o", markersize=3, label="F1")
        ax.plot(df["tau"], df["precision"], label="Precision")
        ax.plot(df["tau"], df["recall"], label="Recall")
        best = df.iloc[int(np.argmax(df["f1"].to_numpy()))]
        ax.axvline(best["tau"], linestyle="--", linewidth=1.0, label=f"selected tau={best['tau']:.3g}")
        ax.set_xlabel(r"Threshold $\tau$"); ax.set_ylabel("Score"); ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{label} threshold tuning"); ax.grid(True, alpha=0.2); ax.legend()
        fig.tight_layout(); paths.extend(_save(fig, out / "figures" / "tasks" / stem, formats))
    return paths


def plot_gamma_distribution(cfg: dict, formats: str = "both") -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    data = np.load(out / "evaluation" / "gamma_distribution.npz")
    a = data["a"].astype(bool)
    paths: list[Path] = []
    for key, label, stem in [("casbl_gamma", "CA-SBL-ANC", "07_casbl_gamma_distribution"), ("sbl_gamma", "SBL", "07_sbl_gamma_distribution")]:
        gamma = data[key]
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.hist(gamma[~a], bins=50, alpha=0.55, density=True, label="Inactive MTD")
        ax.hist(gamma[a], bins=50, alpha=0.55, density=True, label="Active MTD")
        ax.set_xlabel(r"$\gamma_i$"); ax.set_ylabel("Density"); ax.set_title(f"{label} gamma distribution")
        ax.legend(); ax.grid(True, alpha=0.2); fig.tight_layout()
        paths.extend(_save(fig, out / "figures" / "tasks" / stem, formats))
    return paths


def plot_convergence(cfg: dict, formats: str = "both") -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    conv = np.load(out / "convergence" / "convergence.npz")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(np.arange(1, len(conv["casbl_nmse_history"]) + 1), conv["casbl_nmse_history"], label="CA-SBL-ANC")
    ax.plot(np.arange(1, len(conv["sbl_nmse_history"]) + 1), conv["sbl_nmse_history"], label="SBL")
    ax.set_xlabel("Iteration"); ax.set_ylabel("NMSE"); ax.set_yscale("log")
    ax.grid(True, alpha=0.2); ax.legend(); ax.set_title("Convergence at reference condition")
    fig.tight_layout()
    return _save(fig, out / "figures" / "tasks" / "08_convergence_nmse", formats)


def _line_plot(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, stem: Path, formats: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for algorithm, group in df.groupby("algorithm"):
        group = group.sort_values(x)
        ax.plot(group[x], group[y], marker="o", label=algorithm)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.25); ax.legend()
    fig.tight_layout(); return _save(fig, stem, formats)


def make_performance_figures(cfg: dict, formats: str = "both", show_progress: bool = True) -> list[Path]:
    out = Path(cfg["run"]["output_dir"]); figdir = ensure_dir(out / "figures")
    df = pd.read_csv(out / "evaluation" / "aggregate.csv")
    snr = df[df["sweep"] == "snr_sweep"]
    pilot = df[df["sweep"] == "pilot_sweep"]
    jobs = [
        (snr, "snr_db", "f1", "Activity Detection vs SNR", "SNR (dB)", "F1 score", figdir / "tasks" / "09_f1_vs_snr"),
        (snr, "snr_db", "nmse", "Channel Estimation vs SNR", "SNR (dB)", "NMSE", figdir / "tasks" / "09_nmse_vs_snr"),
        (pilot, "L", "f1", "Activity Detection vs Pilot Length", "Pilot length L", "F1 score", figdir / "tasks" / "09_f1_vs_pilot_length"),
        (pilot, "L", "nmse", "Channel Estimation vs Pilot Length", "Pilot length L", "NMSE", figdir / "tasks" / "09_nmse_vs_pilot_length"),
        (snr, "snr_db", "runtime_s", "Runtime vs SNR", "SNR (dB)", "Mean runtime (s)", figdir / "tasks" / "09_runtime_vs_snr"),
    ]
    paths: list[Path] = []
    for job in tqdm(jobs, desc="Performance figures", unit="figure", disable=not show_progress):
        paths.extend(_line_plot(*job, formats=formats))
    return paths


def make_figures(cfg: dict, formats: str = "both", show_progress: bool = True, selected_sample: int | None = None) -> list[Path]:
    """Generate all figures whose prerequisite task outputs exist."""
    out = Path(cfg["run"]["output_dir"])
    paths: list[Path] = []
    sample_index = int(cfg.get("figures", {}).get("selected_sample", 0) if selected_sample is None else selected_sample)
    if (out / "activity" / "tuning_activity.npz").exists():
        paths.extend(plot_activity_realization(
            cfg, "tuning", sample_index, formats=formats,
            show_event_radius=cfg.get("figures", {}).get("show_event_radius", True),
            show_probability_field=cfg.get("figures", {}).get("show_probability_field", False),
        ))
        if (out / "activity" / "evaluation_activity.npz").exists():
            paths.extend(plot_activity_s_distribution(cfg, formats=formats))
    if (out / "communication" / "preview.npz").exists(): paths.extend(plot_communication_preview(cfg, formats))
    if (out / "correlation" / "preview_C.npz").exists(): paths.extend(plot_correlation_preview(cfg, formats))
    if (out / "tuning" / "alpha_beta.csv").exists(): paths.extend(plot_alpha_beta_tuning(cfg, formats))
    if (out / "tuning" / "casbl_thresholds.csv").exists(): paths.extend(plot_threshold_tuning(cfg, formats))
    if (out / "evaluation" / "gamma_distribution.npz").exists(): paths.extend(plot_gamma_distribution(cfg, formats))
    if (out / "convergence" / "convergence.npz").exists(): paths.extend(plot_convergence(cfg, formats))
    if (out / "evaluation" / "aggregate.csv").exists(): paths.extend(make_performance_figures(cfg, formats, show_progress=show_progress))
    print("Generated figures:")
    for path in paths: print(f"  - {path}")
    return paths
