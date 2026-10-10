"""Task09 performance figures."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
from casbl_jadce.paths import result_path, figure_path
from casbl_jadce.figures.common import _apply_figure_style, _save


def _line_plot(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, stem: Path, formats: str) -> list[Path]:
    figure_style = {"axes.titlesize": 12, "axes.labelsize": 12,
                    "xtick.labelsize": 10, "ytick.labelsize": 10,
                    "legend.fontsize": 10.5, "colorbar.labelsize": 12,
                    "colorbar.ticksize": 10}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for algorithm, group in df.groupby("algorithm"):
        group = group.sort_values(x)
        ax.plot(group[x], group[y], marker="o", label=algorithm)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.25); ax.legend()
    _apply_figure_style(fig, figure_style); fig.tight_layout(); return _save(fig, stem, formats)



def make_performance_figures(cfg: dict, formats: str = "png", show_progress: bool = True) -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    df = pd.read_csv(result_path(out, "evaluation", "aggregate.csv"))
    snr = df[df["sweep"] == "snr_sweep"]
    pilot = df[df["sweep"] == "pilot_sweep"]
    jobs = [
        (snr, "snr_db", "f1", r"Activity Detection vs $\mathrm{SNR}$", r"$\mathrm{SNR}$ (dB)", r"$F_1$ score", figure_path(out, "09_f1_vs_snr")),
        (snr, "snr_db", "nmse", r"Channel Estimation vs $\mathrm{SNR}$", r"$\mathrm{SNR}$ (dB)", r"$\mathrm{NMSE}$", figure_path(out, "09_nmse_vs_snr")),
        (pilot, "L", "f1", r"Activity Detection vs Pilot Length $L$", r"Pilot length $L$", r"$F_1$ score", figure_path(out, "09_f1_vs_pilot_length")),
        (pilot, "L", "nmse", r"Channel Estimation vs Pilot Length $L$", r"Pilot length $L$", r"$\mathrm{NMSE}$", figure_path(out, "09_nmse_vs_pilot_length")),
        (snr, "snr_db", "runtime_s", r"Runtime vs $\mathrm{SNR}$", r"$\mathrm{SNR}$ (dB)", "Mean runtime (s)", figure_path(out, "09_runtime_vs_snr")),
    ]
    paths: list[Path] = []
    for job in tqdm(jobs, desc="Performance figures", unit="figure", disable=not show_progress):
        paths.extend(_line_plot(*job, formats=formats))
    return paths
