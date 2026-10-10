"""Task05 alpha beta tuning figures."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from casbl_jadce.paths import result_path, figure_path
from casbl_jadce.figures.common import _apply_figure_style, _save


def plot_alpha_beta_tuning(cfg: dict, formats: str = "png") -> list[Path]:
    out = Path(cfg["run"]["output_dir"])
    df = pd.read_csv(result_path(out, "tuning", "alpha_beta.csv"))
    pivot = df.pivot(index="alpha", columns="beta", values="f1").sort_index().sort_index(axis=1)
    figure_style = {"axes.titlesize": 12, "axes.labelsize": 12,
                    "xtick.labelsize": 10, "ytick.labelsize": 10,
                    "legend.fontsize": 10.5, "colorbar.labelsize": 12,
                    "colorbar.ticksize": 10}
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    image = ax.imshow(pivot.values, aspect="auto", origin="lower")
    fig.colorbar(image, ax=ax, label=r"Mean tuning $F_1$")
    ax.set_xticks(np.arange(len(pivot.columns)), [f"{x:g}" for x in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), [f"{x:g}" for x in pivot.index])
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$\alpha$"); ax.set_title(r"CA-SBL $\alpha$-$\beta$ tuning")
    _apply_figure_style(fig, figure_style); fig.tight_layout()
    return _save(fig, figure_path(out, "05_alpha_beta_f1"), formats)
