"""Command-line entry point for new and resumed experiments."""

from __future__ import annotations

import argparse

from .experiment import (
    ExperimentConfig,
    prepare_problems,
    run_experiment,
    write_summaries,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or resume a finite-N regression experiment."
    )
    parser.add_argument("config", help="Path to an experiment JSON file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue existing incomplete checkpoints; converged runs remain untouched",
    )
    parser.add_argument(
        "--additional-steps",
        type=int,
        default=None,
        help="Additional budget per incomplete run (default: steps_per_round from config)",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--summary-only",
        action="store_true",
        help="Rebuild summary files from checkpoints without advancing any run",
    )
    modes.add_argument(
        "--prepare-only",
        action="store_true",
        help="Save all datasets without starting dynamics",
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=("mirror", "gate", "linear_gd"),
        help="Advance only this configured method; repeat to select several",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig.from_json(args.config)
    if args.prepare_only:
        report = prepare_problems(config)
        print(f"Dataset directory: {report['output_directory'] / 'problems'}")
        print(f"Datasets:          {report['problems']}")
        print(f"New datasets:      {report['created']}")
        return 0
    if args.summary_only:
        paths = write_summaries(config)
        print(f"Summary: {paths['text']}")
        return 0
    report = run_experiment(
        config,
        resume=args.resume,
        additional_steps=args.additional_steps,
        methods=tuple(args.method) if args.method else None,
        verbose=True,
    )
    print(f"Output directory: {report['output_directory']}")
    print(f"Started runs:    {report['started_runs']}")
    print(f"Converged runs:  {report['converged_runs']}")
    print(f"Advanced steps:  {report['advanced_steps']}")
    print(f"Summary:         {report['summary_paths']['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
