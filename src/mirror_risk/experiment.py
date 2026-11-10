"""Resumable finite-N disorder experiments over an alpha grid."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from .dynamics import ConvergenceConfig, DynamicsConfig, DynamicsRun
from .problem import LinearRegressionProblem, ProblemSpec
from .storage import canonical_json


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete, serializable definition of a finite-N experiment."""

    name: str
    output_directory: str
    N: int
    alpha_values: tuple[float, ...]
    kappa: float
    sigma: float
    r: float
    seeds: tuple[int, ...]
    dynamics: tuple[DynamicsConfig, ...]
    convergence: ConvergenceConfig
    steps_per_round: int = 10_000
    checkpoint_every: int = 1_000

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Experiment name cannot be empty")
        if int(self.N) != self.N or self.N <= 0:
            raise ValueError("N must be a positive integer")
        if not self.alpha_values or any(
            not np.isfinite(a) or a <= 0 for a in self.alpha_values
        ):
            raise ValueError("alpha_values must be nonempty, positive, and finite")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if any(int(seed) != seed for seed in self.seeds):
            raise ValueError("Every seed must be an integer")
        if not np.isfinite(self.kappa) or self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if not np.isfinite(self.sigma) or self.sigma < 0:
            raise ValueError("sigma must be nonnegative")
        if not np.isfinite(self.r) or self.r < 0:
            raise ValueError("r must be nonnegative")
        if self.steps_per_round <= 0 or self.checkpoint_every <= 0:
            raise ValueError("Step and checkpoint budgets must be positive")
        if not self.dynamics:
            raise ValueError("At least one dynamics must be configured")
        methods = [item.method for item in self.dynamics]
        if len(set(methods)) != len(methods):
            raise ValueError("Each dynamics method may appear only once")
        if any(not np.isclose(item.kappa, self.kappa) for item in self.dynamics):
            raise ValueError(
                "Every dynamics configuration must use the experiment kappa"
            )

        sample_counts = [max(1, round(self.N / alpha)) for alpha in self.alpha_values]
        if len(set(sample_counts)) != len(sample_counts):
            raise ValueError(
                "At this finite N, at least two requested alpha values round to the same P. "
                "Choose the grid through distinct integer sample counts."
            )

    @property
    def expected_run_count(self) -> int:
        return len(self.alpha_values) * len(self.seeds) * len(self.dynamics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "output_directory": self.output_directory,
            "N": int(self.N),
            "alpha_values": list(self.alpha_values),
            "kappa": self.kappa,
            "sigma": self.sigma,
            "r": self.r,
            "seeds": list(self.seeds),
            "dynamics": [item.to_dict() for item in self.dynamics],
            "convergence": self.convergence.to_dict(),
            "steps_per_round": int(self.steps_per_round),
            "checkpoint_every": int(self.checkpoint_every),
        }

    def scientific_signature(self) -> dict[str, Any]:
        """Fields that must not change when an experiment is resumed."""
        data = self.to_dict()
        data.pop("steps_per_round")
        data.pop("checkpoint_every")
        data.pop("output_directory")
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        kappa = float(data["kappa"])
        dynamics = []
        for item in data.get("dynamics", [{"method": "mirror"}]):
            item = dict(item)
            item.setdefault("kappa", kappa)
            dynamics.append(DynamicsConfig.from_dict(item))

        if "seeds" in data:
            seeds = tuple(int(seed) for seed in data["seeds"])
        else:
            n_realizations = int(data.get("n_realizations", 1))
            first_seed = int(data.get("first_seed", 0))
            seeds = tuple(range(first_seed, first_seed + n_realizations))

        return cls(
            name=str(data["name"]),
            output_directory=str(data["output_directory"]),
            N=int(data["N"]),
            alpha_values=tuple(float(alpha) for alpha in data["alpha_values"]),
            kappa=kappa,
            sigma=float(data.get("sigma", 0.0)),
            r=float(data.get("r", 1.0)),
            seeds=seeds,
            dynamics=tuple(dynamics),
            convergence=ConvergenceConfig.from_dict(data.get("convergence", {})),
            steps_per_round=int(data.get("steps_per_round", 10_000)),
            checkpoint_every=int(data.get("checkpoint_every", 1_000)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        path = Path(path)
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        config = cls.from_dict(data)
        output = Path(config.output_directory)
        if not output.is_absolute():
            output = (path.parent / output).resolve()
        data = config.to_dict()
        data["output_directory"] = str(output)
        return cls.from_dict(data)


def _alpha_key(spec: ProblemSpec) -> str:
    return f"P_{spec.P:06d}_alpha_{spec.realized_alpha:.12g}"


def _problem_path(root: Path, spec: ProblemSpec) -> Path:
    return root / "problems" / _alpha_key(spec) / f"seed_{spec.seed:06d}.npz"


def _checkpoint_path(root: Path, spec: ProblemSpec, method: str) -> Path:
    return root / "runs" / method / _alpha_key(spec) / f"seed_{spec.seed:06d}.npz"


def _load_or_create_problem(root: Path, spec: ProblemSpec) -> LinearRegressionProblem:
    path = _problem_path(root, spec)
    if path.exists():
        problem = LinearRegressionProblem.load(path)
        if problem.spec != spec:
            raise ValueError(
                f"Stored problem metadata does not match requested spec: {path}"
            )
        return problem
    problem = LinearRegressionProblem.generate(spec)
    problem.save(path)
    return problem


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _prepare_output(config: ExperimentConfig) -> Path:
    root = Path(config.output_directory)
    root.mkdir(parents=True, exist_ok=True)
    snapshot_path = root / "experiment_config.json"

    if snapshot_path.exists():
        with snapshot_path.open("r", encoding="utf-8") as stream:
            previous = ExperimentConfig.from_dict(json.load(stream))
        if canonical_json(previous.scientific_signature()) != canonical_json(
            config.scientific_signature()
        ):
            raise ValueError(
                "The output directory already contains a scientifically different experiment. "
                "Use another output_directory or restore the original configuration."
            )
    else:
        _write_json_atomic(snapshot_path, config.to_dict())
    return root


def prepare_problems(config: ExperimentConfig) -> dict[str, Any]:
    """Save the configured datasets without starting either learning dynamics."""
    root = _prepare_output(config)
    created = 0
    for alpha in config.alpha_values:
        for seed in config.seeds:
            spec = ProblemSpec(config.N, alpha, config.sigma, config.r, seed)
            path = _problem_path(root, spec)
            existed = path.exists()
            _load_or_create_problem(root, spec)
            created += not existed
    return {
        "output_directory": root,
        "problems": len(config.alpha_values) * len(config.seeds),
        "created": created,
    }


def _run_row(
    spec: ProblemSpec,
    method: str,
    checkpoint: Path,
    problem: LinearRegressionProblem,
) -> dict[str, Any]:
    run = DynamicsRun.load_checkpoint(problem, checkpoint)
    values = run.final_observables()
    return {
        "N": spec.N,
        "P": spec.P,
        "alpha_requested": spec.alpha,
        "alpha": spec.realized_alpha,
        "seed": spec.seed,
        "method": method,
        "started": 1,
        "converged": int(run.converged),
        "termination_reason": run.termination_reason,
        "iteration": run.iteration,
        "time": run.time,
        "step_size": run.step_size,
        "risk": values["risk"],
        "test_mse": values["test_mse"],
        "loss": values["loss"],
        "train_mse": values["train_mse"],
        "m": values["m"],
        "q": values["q"],
        "M": values["M"],
        "epsilon": None if method == "linear_gd" else values["epsilon"],
        "active_fraction": values["active_fraction"],
        "gradient_rms": values["gradient_rms"],
        "projected_gradient_rms": values["projected_gradient_rms"],
        "coordinate_rms": values["coordinate_rms"],
        "relative_weight_change": values["relative_weight_change"],
        "relative_loss_change": values["relative_loss_change"],
        "mirror_lambda": values["mirror_lambda"],
        "sign_crossings": run.sign_crossings if method == "gate" else None,
        "crossed_fraction": (
            float(np.mean(run.crossed_mask)) if method == "gate" else None
        ),
        "teacher_mean": float(np.mean(problem.w_star)),
        "teacher_q": float(np.mean(problem.w_star**2)),
        "training_noise_mse": float(np.mean(problem.noise**2)),
        "problem_fingerprint": problem.fingerprint(),
        "checkpoint": str(checkpoint),
    }


def collect_run_rows(config: ExperimentConfig) -> list[dict[str, Any]]:
    root = Path(config.output_directory)
    rows: list[dict[str, Any]] = []
    for alpha in config.alpha_values:
        for seed in config.seeds:
            spec = ProblemSpec(config.N, alpha, config.sigma, config.r, seed)
            problem_path = _problem_path(root, spec)
            problem = (
                LinearRegressionProblem.load(problem_path)
                if problem_path.exists()
                else None
            )
            for dynamics in config.dynamics:
                checkpoint = _checkpoint_path(root, spec, dynamics.method)
                if problem is not None and checkpoint.exists():
                    rows.append(_run_row(spec, dynamics.method, checkpoint, problem))
                else:
                    rows.append(
                        {
                            "N": spec.N,
                            "P": spec.P,
                            "alpha_requested": spec.alpha,
                            "alpha": spec.realized_alpha,
                            "seed": spec.seed,
                            "method": dynamics.method,
                            "started": 0,
                            "converged": 0,
                            "termination_reason": "missing",
                            "iteration": 0,
                        }
                    )
    return rows


def _sample_statistics(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {key: np.nan for key in ("mean", "std", "sem", "ci95_low", "ci95_high")}
    mean = float(np.mean(array))
    if array.size == 1:
        return {
            "mean": mean,
            "std": np.nan,
            "sem": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }
    std = float(np.std(array, ddof=1))
    sem = std / np.sqrt(array.size)
    half_width = float(student_t.ppf(0.975, array.size - 1) * sem)
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def aggregate_run_rows(
    config: ExperimentConfig,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for alpha in config.alpha_values:
        P = max(1, round(config.N / alpha))
        realized_alpha = config.N / P
        for dynamics in config.dynamics:
            selected = [
                row
                for row in rows
                if row["method"] == dynamics.method and row["P"] == P
            ]
            started = [row for row in selected if row["started"]]
            converged = [row for row in started if row["converged"]]
            aggregate: dict[str, Any] = {
                "N": config.N,
                "P": P,
                "alpha_requested": alpha,
                "alpha": realized_alpha,
                "method": dynamics.method,
                "expected_runs": len(config.seeds),
                "started_runs": len(started),
                "converged_runs": len(converged),
                "converged_fraction": len(converged) / len(config.seeds),
                "median_iterations": float(
                    np.median([row["iteration"] for row in started])
                )
                if started
                else np.nan,
                "maximum_iterations": max(
                    (row["iteration"] for row in started), default=0
                ),
            }

            # Keep enough aggregate information for later theory comparisons
            # without having to rerun the trajectories.  The one-row-per-run
            # CSV remains the authoritative source if a new statistic is
            # desired later.
            quantities = (
                "risk",
                "test_mse",
                "loss",
                "train_mse",
                "m",
                "q",
                "M",
                "epsilon",
                "active_fraction",
                "gradient_rms",
                "projected_gradient_rms",
                "coordinate_rms",
                "relative_weight_change",
                "relative_loss_change",
                "crossed_fraction",
                "teacher_mean",
                "teacher_q",
                "training_noise_mse",
            )
            for quantity in quantities:
                all_statistics = _sample_statistics(
                    row[quantity] for row in started if row[quantity] is not None
                )
                converged_statistics = _sample_statistics(
                    row[quantity] for row in converged if row[quantity] is not None
                )
                aggregate.update(
                    {
                        f"{quantity}_all_{key}": value
                        for key, value in all_statistics.items()
                    }
                )
                aggregate.update(
                    {
                        f"{quantity}_converged_{key}": value
                        for key, value in converged_statistics.items()
                    }
                )
            aggregates.append(aggregate)
    return aggregates


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _format_number(value: Any, width: int = 11) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "--".rjust(width)
    return f"{value:{width}.4g}"


def _write_text_summary(
    path: Path,
    config: ExperimentConfig,
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
) -> None:
    started = sum(int(row["started"]) for row in rows)
    converged = sum(int(row["converged"]) for row in rows)
    generated = datetime.now(timezone.utc).isoformat()
    physical = any(item.method in ("mirror", "gate") for item in config.dynamics)

    lines = [
        "MIRROR-RISK FINITE-N EXPERIMENT SUMMARY"
        if physical else "LINEAR GD FINITE-N EXPERIMENT SUMMARY",
        "=" * 80,
        f"Generated (UTC): {generated}",
        f"Experiment:      {config.name}",
        f"Output:          {config.output_directory}",
        f"N:               {config.N}",
        f"kappa:           {config.kappa:g}"
        if physical else "kappa:           not applicable to linear GD",
        f"sigma:           {config.sigma:g}",
        f"r:               {config.r:g}",
        f"Methods:         {', '.join(item.method for item in config.dynamics)}",
        f"Seeds:           {len(config.seeds)} ({min(config.seeds)} ... {max(config.seeds)})",
        f"Expected runs:   {config.expected_run_count}",
        f"Started runs:    {started}",
        f"Converged runs:  {converged}",
        "",
        "Convergence rule",
        "-" * 80,
        (
            "A run must satisfy a first-order residual, relative weight-change, and "
            "relative loss-change tolerances for the requested patience. The residual "
            "is the raw gradient for linear_gd and the projected gradient otherwise."
        ),
        (
            f"check_every={config.convergence.check_every}, "
            f"min_steps={config.convergence.min_steps}, "
            f"patience={config.convergence.patience}"
        ),
        (
            f"projected_gradient_tol={config.convergence.projected_gradient_tol:g}, "
            f"relative_weight_tol={config.convergence.relative_weight_tol:g}, "
            f"relative_loss_tol={config.convergence.relative_loss_tol:g}"
        ),
        "",
        "Statistics by alpha and method",
        "-" * 116,
        (
            " method       alpha       P   started  converged   fraction    "
            "risk(all)   SEM(all)   risk(conv)  loss(all)   max steps"
        ),
        "-" * 116,
    ]

    for row in aggregates:
        lines.append(
            f" {row['method']:<10}"
            f" {_format_number(row['alpha'], 10)}"
            f" {int(row['P']):7d}"
            f" {int(row['started_runs']):9d}"
            f" {int(row['converged_runs']):10d}"
            f" {_format_number(row['converged_fraction'], 10)}"
            f" {_format_number(row['risk_all_mean'])}"
            f" {_format_number(row['risk_all_sem'])}"
            f" {_format_number(row['risk_converged_mean'])}"
            f" {_format_number(row['loss_all_mean'])}"
            f" {int(row['maximum_iterations']):11d}"
        )

    incomplete = [row for row in rows if row["started"] and not row["converged"]]
    lines.extend(["", "Incomplete runs", "-" * 80])
    if not incomplete:
        lines.append("None.")
    else:
        lines.append(
            " method       alpha    seed   iteration   KKT RMS         risk       loss"
        )
        for row in incomplete:
            lines.append(
                f" {row['method']:<10}"
                f" {_format_number(row['alpha'], 9)}"
                f" {int(row['seed']):7d}"
                f" {int(row['iteration']):11d}"
                f" {_format_number(row['projected_gradient_rms'], 12)}"
                f" {_format_number(row['risk'], 10)}"
                f" {_format_number(row['loss'], 10)}"
            )

    lines.extend(
        [
            "",
            "Notes",
            "-" * 80,
            "risk(all) includes the current endpoint of every started run.",
            "risk(conv) uses only runs certified by the convergence rule.",
            "Do not compare risk(conv) alone when the convergence fraction is small; the",
            "conditioning can bias the sample. Resume incomplete runs instead.",
            "summary_runs.csv contains one row per realization and checkpoint.",
            "summary_by_alpha.csv contains the aggregate values shown above.",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write("\n".join(lines))
        temporary = Path(stream.name)
    os.replace(temporary, path)


def write_summaries(config: ExperimentConfig) -> dict[str, Path]:
    root = Path(config.output_directory)
    rows = collect_run_rows(config)
    aggregates = aggregate_run_rows(config, rows)
    run_csv = root / "summary_runs.csv"
    alpha_csv = root / "summary_by_alpha.csv"
    text = root / "summary.txt"
    _write_csv(run_csv, rows)
    _write_csv(alpha_csv, aggregates)
    _write_text_summary(text, config, rows, aggregates)
    return {"runs": run_csv, "aggregate": alpha_csv, "text": text}


def run_experiment(
    config: ExperimentConfig,
    *,
    resume: bool = False,
    additional_steps: int | None = None,
    methods: tuple[str, ...] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run or resume every realization in an experiment.

    Existing converged checkpoints are always skipped.  Existing incomplete
    checkpoints are advanced only when ``resume=True``.  Missing runs are
    created in either mode.  Summaries are refreshed after every attempted
    run and in a final ``finally`` block.
    """
    requested = set(methods) if methods is not None else None
    configured = {item.method for item in config.dynamics}
    if requested is not None and (not requested or not requested <= configured):
        raise ValueError("Requested methods must be present in the experiment config")
    selected_dynamics = tuple(
        item for item in config.dynamics
        if requested is None or item.method in requested
    )
    root = _prepare_output(config)
    budget = (
        config.steps_per_round if additional_steps is None else int(additional_steps)
    )
    if budget <= 0:
        raise ValueError("additional_steps must be positive")

    attempted = 0
    advanced = 0
    skipped_converged = 0
    skipped_incomplete = 0

    try:
        for alpha in config.alpha_values:
            for seed in config.seeds:
                spec = ProblemSpec(config.N, alpha, config.sigma, config.r, seed)
                problem = _load_or_create_problem(root, spec)

                for dynamics in selected_dynamics:
                    checkpoint = _checkpoint_path(root, spec, dynamics.method)
                    attempted += 1

                    if checkpoint.exists():
                        run = DynamicsRun.load_checkpoint(problem, checkpoint)
                        if run.converged:
                            skipped_converged += 1
                            if verbose:
                                print(
                                    f"skip converged: method={dynamics.method:<6} "
                                    f"alpha={spec.realized_alpha:.6g} seed={seed} "
                                    f"iteration={run.iteration}"
                                )
                            continue
                        if not resume:
                            skipped_incomplete += 1
                            if verbose:
                                print(
                                    f"skip incomplete: method={dynamics.method:<6} "
                                    f"alpha={spec.realized_alpha:.6g} seed={seed} "
                                    f"iteration={run.iteration}; use --resume"
                                )
                            continue
                    else:
                        run = DynamicsRun(problem, dynamics, config.convergence)

                    steps_taken = run.advance(
                        budget,
                        checkpoint_path=checkpoint,
                        checkpoint_every=config.checkpoint_every,
                    )
                    advanced += steps_taken
                    if verbose:
                        print(
                            f"advanced:       method={dynamics.method:<6} "
                            f"alpha={spec.realized_alpha:.6g} seed={seed} "
                            f"steps={steps_taken} iteration={run.iteration} "
                            f"converged={run.converged}"
                        )
    finally:
        summary_paths = write_summaries(config)

    rows = collect_run_rows(config)
    return {
        "output_directory": root,
        "attempted": attempted,
        "advanced_steps": advanced,
        "skipped_converged": skipped_converged,
        "skipped_incomplete": skipped_incomplete,
        "started_runs": sum(int(row["started"]) for row in rows),
        "converged_runs": sum(int(row["converged"]) for row in rows),
        "summary_paths": summary_paths,
    }
