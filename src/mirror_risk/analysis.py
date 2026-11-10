"""Small analysis helpers for completed or partial experiments."""

from __future__ import annotations

import csv
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

PredictFunction = Callable[[float, float, float, float], dict[str, object]]


def load_theory_predictor(path: str | Path) -> PredictFunction:
    """Load the validated ``predict`` function from an external theory file."""
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("mirror_risk_external_theory", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load theory module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "predict"):
        raise AttributeError(f"{path} does not define predict(k, alpha, sigma, r)")
    return module.predict


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def plot_risk_comparison(
    experiment_directory: str | Path,
    *,
    theory_predict: PredictFunction | None = None,
    converged_only: bool = False,
    output: str | Path | None = None,
):
    """Plot finite-N risk means and optional thermodynamic predictions."""
    import matplotlib.pyplot as plt

    root = Path(experiment_directory)
    with (root / "experiment_config.json").open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    rows = _read_csv(root / "summary_by_alpha.csv")

    mean_key = "risk_converged_mean" if converged_only else "risk_all_mean"
    low_key = "risk_converged_ci95_low" if converged_only else "risk_all_ci95_low"
    high_key = "risk_converged_ci95_high" if converged_only else "risk_all_ci95_high"

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    methods = list(dict.fromkeys(row["method"] for row in rows))
    for method in methods:
        selected = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: float(row["alpha"]),
        )
        alpha = np.asarray([float(row["alpha"]) for row in selected])
        mean = np.asarray([float(row[mean_key]) for row in selected])
        low = np.asarray([float(row[low_key]) for row in selected])
        high = np.asarray([float(row[high_key]) for row in selected])
        valid = np.isfinite(mean)
        lower_error = np.maximum(0.0, mean - low)
        upper_error = np.maximum(0.0, high - mean)
        ax.errorbar(
            alpha[valid],
            mean[valid],
            yerr=np.vstack([lower_error[valid], upper_error[valid]]),
            marker="o",
            linewidth=1.3,
            capsize=2.5,
            label=f"{method}, N={config['N']}",
        )

    if theory_predict is not None:
        alpha = np.asarray(sorted({float(row["alpha"]) for row in rows}))
        theory = np.asarray(
            [
                theory_predict(
                    float(config["kappa"]),
                    value,
                    float(config["sigma"]),
                    float(config["r"]),
                )["risk"]
                for value in alpha
            ]
        )
        ax.plot(
            alpha,
            theory,
            color="black",
            linestyle="--",
            label="thermodynamic prediction",
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha=N/P$")
    ax.set_ylabel(r"$\mathcal{R}$")
    ax.legend(frameon=False)
    fig.tight_layout()

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
    return fig, ax
