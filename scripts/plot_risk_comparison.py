"""Plot finite-N mean risks against thermodynamic-limit predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from mirror_risk.analysis import load_theory_predictor, plot_risk_comparison
from resistor_theory import predict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_directory")
    parser.add_argument(
        "--theory-file",
        default=None,
        help="Optional path to an external thermodynamic_predictions.py file",
    )
    parser.add_argument("--converged-only", action="store_true")
    parser.add_argument("--output", default=None, help="Optional PDF/PNG output path")
    args = parser.parse_args()

    predictor = load_theory_predictor(args.theory_file) if args.theory_file else predict
    plot_risk_comparison(
        Path(args.experiment_directory),
        theory_predict=predictor,
        converged_only=args.converged_only,
        output=args.output,
    )
    if args.output is None:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
