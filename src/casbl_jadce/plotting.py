from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .io import ensure_dir


def _line_plot(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for algorithm, group in df.groupby("algorithm"):
        group = group.sort_values(x)
        ax.plot(group[x], group[y], marker="o", label=algorithm)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def make_figures(output_dir: str | Path) -> None:
    out = Path(output_dir); figdir = ensure_dir(out / "figures")
    df = pd.read_csv(out / "evaluation" / "aggregate.csv")
    snr = df[df["sweep"] == "snr_sweep"]
    pilot = df[df["sweep"] == "pilot_sweep"]
    _line_plot(snr, "snr_db", "f1", "Activity Detection vs SNR", "SNR (dB)", "F1 score", figdir / "f1_vs_snr.png")
    _line_plot(snr, "snr_db", "nmse", "Channel Estimation vs SNR", "SNR (dB)", "NMSE", figdir / "nmse_vs_snr.png")
    _line_plot(pilot, "L", "f1", "Activity Detection vs Pilot Length", "Pilot length L", "F1 score", figdir / "f1_vs_pilot_length.png")
    _line_plot(pilot, "L", "nmse", "Channel Estimation vs Pilot Length", "Pilot length L", "NMSE", figdir / "nmse_vs_pilot_length.png")
    _line_plot(snr, "snr_db", "runtime_s", "Runtime vs SNR", "SNR (dB)", "Mean runtime (s)", figdir / "runtime_vs_snr.png")

    conv = np.load(out / "convergence" / "convergence.npz")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(np.arange(1, len(conv["casbl_nmse_history"]) + 1), conv["casbl_nmse_history"], label="CA-SBL-ANC")
    ax.plot(np.arange(1, len(conv["sbl_nmse_history"]) + 1), conv["sbl_nmse_history"], label="SBL")
    ax.set_xlabel("Iteration"); ax.set_ylabel("NMSE"); ax.set_yscale("log"); ax.grid(True, alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(figdir / "convergence_nmse.png", dpi=200); plt.close(fig)
