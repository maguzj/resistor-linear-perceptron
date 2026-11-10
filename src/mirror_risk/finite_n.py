"""Finite-N fixed points and paired comparisons with saved learning runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import LinearConstraint, linprog, minimize
from scipy.stats import binomtest

from .dynamics import DynamicsRun, projected_gradient_rms
from .experiment import (
    ExperimentConfig,
    _alpha_key,
    _checkpoint_path,
    _problem_path,
    _sample_statistics,
    _write_csv,
)
from .problem import LinearRegressionProblem, ProblemSpec
from .storage import atomic_savez


@dataclass(frozen=True)
class Endpoint:
    w: np.ndarray
    regime: str
    critical_kappa: float


@dataclass(frozen=True)
class LinearEndpoint:
    w: np.ndarray
    regime: str


def _positive_negative(w: np.ndarray) -> np.ndarray:
    return np.r_[np.maximum(w, 0.0), np.maximum(-w, 0.0)]


def _basis_pursuit(problem: LinearRegressionProblem) -> np.ndarray:
    N = problem.N
    result = linprog(
        np.ones(2 * N),
        A_eq=np.c_[problem.X, -problem.X],
        b_eq=problem.y,
        bounds=(0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Basis pursuit failed: {result.message}")
    return result.x[:N] - result.x[N:]


def _ball_least_squares(problem: LinearRegressionProblem, G: float) -> np.ndarray:
    N = problem.N
    # Scale the small near-interpolation loss and its gradient together for SLSQP.
    objective_scale = 100.0

    def objective(z: np.ndarray) -> tuple[float, np.ndarray]:
        w = z[:N] - z[N:]
        residual = problem.X @ w - problem.y
        gradient = problem.X.T @ residual / problem.P
        return (
            objective_scale * float(residual @ residual / (2 * problem.P)),
            objective_scale * np.r_[gradient, -gradient],
        )

    result = minimize(
        objective,
        np.zeros(2 * N),
        jac=True,
        method="SLSQP",
        bounds=[(0, None)] * (2 * N),
        constraints=[LinearConstraint(np.ones((1, 2 * N)), -np.inf, G)],
        options={"ftol": 1e-12, "maxiter": 5000},
    )
    if not result.success:
        raise RuntimeError(f"Constrained least squares failed: {result.message}")
    w = result.x[:N] - result.x[N:]
    support = np.abs(w) > 1e-9 * max(1e-12, G / N)
    signs = np.sign(w[support])
    H = problem.hessian[np.ix_(support, support)]
    system = np.block(
        [
            [H, signs[:, None]],
            [signs[None, :], np.zeros((1, 1))],
        ]
    )
    solution = np.linalg.solve(system, np.r_[problem.linear_term[support], G])
    polished = np.zeros(N)
    polished[support] = solution[:-1]
    multiplier = solution[-1]
    gradient = problem.gradient(polished)
    curvature = float(np.linalg.eigvalsh(problem.hessian)[-1])
    kkt = projected_gradient_rms(polished, gradient, G, curvature)
    if (
        np.any(np.sign(polished[support]) != signs)
        or multiplier < -1e-9
        or np.any(np.abs(gradient[~support]) > multiplier + 1e-8)
        or abs(np.linalg.norm(polished, 1) - G) > 1e-8 * max(1.0, G)
        or kkt > 1e-8
    ):
        raise RuntimeError(f"Constrained least squares failed verification: KKT={kkt:g}")
    return polished


def _bregman_interpolator(
    problem: LinearRegressionProblem, G: float, initial: np.ndarray
) -> np.ndarray:
    N = problem.N
    # Scale the steep near-boundary objective and its gradient together for SLSQP.
    objective_scale = 100.0

    def objective(z: np.ndarray) -> tuple[float, np.ndarray]:
        w = z[:N] - z[N:]
        epsilon = max(1.0 - z.sum() / G, 1e-12)
        squared_norm = float(w @ w)
        common = squared_norm / (2.0 * G * epsilon**2)
        gradient = w / epsilon
        return (
            squared_norm / (2.0 * N * epsilon * objective_scale),
            np.r_[gradient + common, -gradient + common] / (N * objective_scale),
        )

    constraints = [
        LinearConstraint(np.c_[problem.X, -problem.X], problem.y, problem.y),
        LinearConstraint(np.ones((1, 2 * N)), -np.inf, G * (1.0 - 1e-12)),
    ]

    def optimize(start: np.ndarray):
        return minimize(
            objective,
            _positive_negative(start),
            jac=True,
            method="SLSQP",
            bounds=[(0, None)] * (2 * N),
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 5000},
        )

    result = optimize(initial)
    if not result.success:
        minimum_norm = np.linalg.lstsq(problem.X, problem.y, rcond=None)[0]
        if np.linalg.norm(minimum_norm, 1) < G:
            result = optimize(minimum_norm)
    if not result.success:
        raise RuntimeError(f"Bregman minimization failed: {result.message}")
    w = result.x[:N] - result.x[N:]
    residual = np.linalg.norm(problem.X @ w - problem.y)
    if (
        residual > 1e-7 * max(1.0, np.linalg.norm(problem.y))
        or np.linalg.norm(w, 1) >= G
        or result.x.sum() - np.linalg.norm(w, 1) > 1e-7 * max(1.0, G)
    ):
        raise RuntimeError(f"Bregman minimization failed verification: residual={residual:g}")
    return w


def solve_endpoint(problem: LinearRegressionProblem, kappa: float) -> Endpoint:
    """Classify one realized dataset and solve its zero-initialized mirror endpoint."""
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be positive and finite")
    G = problem.N * kappa
    if problem.P >= problem.N:
        candidate = np.linalg.lstsq(problem.X, problem.y, rcond=None)[0]
        interior_regime = "ols"
    else:
        candidate = _basis_pursuit(problem)
        interior_regime = "bregman"

    critical_kappa = float(np.linalg.norm(candidate, 1) / problem.N)
    gap = G - float(np.linalg.norm(candidate, 1))
    tolerance = 1e-8 * max(1.0, G)
    if abs(gap) <= tolerance:
        regime = "critical"
        w = _ball_least_squares(problem, G)
    elif gap < 0:
        regime = "lasso"
        w = _ball_least_squares(problem, G)
    elif interior_regime == "ols":
        regime = "ols"
        w = candidate
    else:
        regime = "bregman"
        w = _bregman_interpolator(problem, G, candidate)
    return Endpoint(w=w, regime=regime, critical_kappa=critical_kappa)


def solve_linear_endpoint(problem: LinearRegressionProblem) -> LinearEndpoint:
    """Return the zero-initialized linear GD limit, independent of kappa."""
    w = np.linalg.lstsq(problem.X, problem.y, rcond=None)[0]
    interpolates = np.linalg.norm(problem.residual(w)) <= 1e-8 * max(
        1.0, np.linalg.norm(problem.y)
    )
    regime = (
        "minimum_norm_interpolator"
        if problem.P < problem.N and interpolates
        else "ols"
    )
    return LinearEndpoint(w=w, regime=regime)


def _endpoint_path(root: Path, spec: ProblemSpec) -> Path:
    return root / "solutions" / _alpha_key(spec) / f"seed_{spec.seed:06d}.npz"


def _linear_endpoint_path(root: Path, spec: ProblemSpec) -> Path:
    return root / "linear_solutions" / _alpha_key(spec) / f"seed_{spec.seed:06d}.npz"


def _load_endpoint(
    root: Path, problem: LinearRegressionProblem, spec: ProblemSpec, kappa: float
) -> tuple[np.ndarray, str]:
    with np.load(_endpoint_path(root, spec), allow_pickle=False) as saved:
        if str(saved["problem_fingerprint"].item()) != problem.fingerprint():
            raise ValueError("Endpoint belongs to a different training problem")
        if float(saved["kappa"].item()) != kappa:
            raise ValueError("Endpoint uses a different kappa")
        return saved["w"].copy(), str(saved["regime"].item())


def _load_linear_endpoint(
    root: Path, problem: LinearRegressionProblem, spec: ProblemSpec
) -> tuple[np.ndarray, str]:
    with np.load(_linear_endpoint_path(root, spec), allow_pickle=False) as saved:
        if str(saved["problem_fingerprint"].item()) != problem.fingerprint():
            raise ValueError("Linear endpoint belongs to a different training problem")
        return saved["w"].copy(), str(saved["regime"].item())


def solve_saved_endpoints(config: ExperimentConfig, output: str | Path) -> dict[str, Path]:
    """Solve all saved datasets; write new results without altering simulation files."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    physical = any(item.method in ("mirror", "gate") for item in config.dynamics)
    linear = any(item.method == "linear_gd" for item in config.dynamics)
    rows = []
    linear_rows = []
    for alpha in config.alpha_values:
        for seed in config.seeds:
            spec = ProblemSpec(config.N, alpha, config.sigma, config.r, seed)
            problem_path = _problem_path(Path(config.output_directory), spec)
            problem = LinearRegressionProblem.load(problem_path)
            fingerprint = problem.fingerprint()
            if physical:
                endpoint = solve_endpoint(problem, config.kappa)
                path = _endpoint_path(output, spec)
                atomic_savez(
                    path,
                    w=endpoint.w,
                    regime=np.array(endpoint.regime),
                    critical_kappa=np.array(endpoint.critical_kappa),
                    kappa=np.array(config.kappa),
                    problem_fingerprint=np.array(fingerprint),
                )
                rows.append(
                    {
                        "N": spec.N,
                        "P": spec.P,
                        "alpha": spec.realized_alpha,
                        "seed": seed,
                        "regime": endpoint.regime,
                        "critical_kappa": endpoint.critical_kappa,
                        "problem_fingerprint": fingerprint,
                        "solution": str(path),
                        **problem.observables(endpoint.w, config.kappa),
                    }
                )
            if linear:
                endpoint = solve_linear_endpoint(problem)
                path = _linear_endpoint_path(output, spec)
                atomic_savez(
                    path,
                    w=endpoint.w,
                    regime=np.array(endpoint.regime),
                    problem_fingerprint=np.array(fingerprint),
                )
                observables = problem.observables(endpoint.w, config.kappa)
                observables["epsilon"] = None
                linear_rows.append(
                    {
                        "N": spec.N,
                        "P": spec.P,
                        "alpha": spec.realized_alpha,
                        "seed": seed,
                        "regime": endpoint.regime,
                        "problem_fingerprint": fingerprint,
                        "solution": str(path),
                        **observables,
                    }
                )

    paths = {}
    if physical:
        aggregate = []
        for alpha in config.alpha_values:
            P = ProblemSpec(config.N, alpha, config.sigma, config.r).P
            selected = [row for row in rows if row["P"] == P]
            summary = {"N": config.N, "P": P, "alpha": config.N / P, "runs": len(selected)}
            for regime in ("ols", "lasso", "bregman", "critical"):
                count = sum(row["regime"] == regime for row in selected)
                interval = binomtest(count, len(selected)).proportion_ci(method="wilson")
                summary.update(
                    {
                        f"{regime}_count": count,
                        f"{regime}_fraction": count / len(selected),
                        f"{regime}_ci95_low": interval.low,
                        f"{regime}_ci95_high": interval.high,
                    }
                )
            for quantity in ("risk", "loss"):
                summary.update(
                    {
                        f"{quantity}_{key}": value
                        for key, value in _sample_statistics(
                            row[quantity] for row in selected
                        ).items()
                    }
                )
            aggregate.append(summary)

        paths.update(
            runs=output / "endpoints.csv",
            aggregate=output / "summary_by_alpha.csv",
        )
        _write_csv(paths["runs"], rows)
        _write_csv(paths["aggregate"], aggregate)

    if linear:
        linear_aggregate = []
        for alpha in config.alpha_values:
            P = ProblemSpec(config.N, alpha, config.sigma, config.r).P
            selected = [row for row in linear_rows if row["P"] == P]
            summary = {"N": config.N, "P": P, "alpha": config.N / P, "runs": len(selected)}
            for quantity in ("risk", "loss"):
                summary.update(
                    {
                        f"{quantity}_{key}": value
                        for key, value in _sample_statistics(
                            row[quantity] for row in selected
                        ).items()
                    }
                )
            linear_aggregate.append(summary)
        paths.update(
            linear_runs=output / "linear_endpoints.csv",
            linear_aggregate=output / "linear_summary_by_alpha.csv",
        )
        _write_csv(paths["linear_runs"], linear_rows)
        _write_csv(paths["linear_aggregate"], linear_aggregate)
    return paths


def compare_saved_runs(
    config: ExperimentConfig,
    endpoints: str | Path,
    output: str | Path,
    methods: tuple[str, ...] | None = None,
) -> dict[str, Path]:
    """Pair each dynamics checkpoint with the endpoint of its exact dataset."""
    requested = set(methods) if methods is not None else None
    configured = {item.method for item in config.dynamics}
    if requested is not None and (not requested or not requested <= configured):
        raise ValueError("Requested methods must be present in the experiment config")
    selected_dynamics = tuple(
        item for item in config.dynamics
        if requested is None or item.method in requested
    )
    endpoints, output = Path(endpoints), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for alpha in config.alpha_values:
        for seed in config.seeds:
            spec = ProblemSpec(config.N, alpha, config.sigma, config.r, seed)
            problem_path = _problem_path(Path(config.output_directory), spec)
            problem = LinearRegressionProblem.load(problem_path)
            for dynamics in selected_dynamics:
                checkpoint = _checkpoint_path(
                    Path(config.output_directory), spec, dynamics.method
                )
                if not checkpoint.exists():
                    continue
                if dynamics.method == "linear_gd":
                    reference, regime = _load_linear_endpoint(endpoints, problem, spec)
                else:
                    reference, regime = _load_endpoint(
                        endpoints, problem, spec, config.kappa
                    )
                target = problem.observables(reference, config.kappa)
                run = DynamicsRun.load_checkpoint(problem, checkpoint)
                if run.config.method != dynamics.method or run.config.kappa != config.kappa:
                    raise ValueError("Checkpoint method or kappa does not match experiment")
                observed = run.final_observables()
                row = {
                    "N": spec.N,
                    "P": spec.P,
                    "alpha": spec.realized_alpha,
                    "seed": seed,
                    "method": dynamics.method,
                    "regime": regime,
                    "converged": int(run.converged),
                    "iteration": run.iteration,
                    "time": run.time,
                    "weight_rms_error": float(
                        np.linalg.norm(run.w - reference) / np.sqrt(spec.N)
                    ),
                    "projected_gradient_rms": observed["projected_gradient_rms"],
                }
                for quantity in ("risk", "loss", "m", "q", "M", "epsilon"):
                    if quantity == "epsilon" and dynamics.method == "linear_gd":
                        row[f"{quantity}_run"] = None
                        row[f"{quantity}_endpoint"] = None
                        row[f"{quantity}_error"] = None
                    else:
                        row[f"{quantity}_run"] = observed[quantity]
                        row[f"{quantity}_endpoint"] = target[quantity]
                        row[f"{quantity}_error"] = observed[quantity] - target[quantity]
                rows.append(row)

    aggregate = []
    for alpha in config.alpha_values:
        P = ProblemSpec(config.N, alpha, config.sigma, config.r).P
        for dynamics in selected_dynamics:
            matching = [
                row
                for row in rows
                if row["P"] == P and row["method"] == dynamics.method
            ]
            for regime in ("all", *sorted({row["regime"] for row in matching})):
                selected = (
                    matching
                    if regime == "all"
                    else [row for row in matching if row["regime"] == regime]
                )
                summary = {
                    "N": config.N,
                    "P": P,
                    "alpha": config.N / P,
                    "method": dynamics.method,
                    "regime": regime,
                    "runs": len(selected),
                    "converged_runs": sum(row["converged"] for row in selected),
                }
                for quantity in ("risk_error", "loss_error", "weight_rms_error"):
                    summary.update(
                        {
                            f"{quantity}_{key}": value
                            for key, value in _sample_statistics(
                                row[quantity] for row in selected
                            ).items()
                        }
                    )
                aggregate.append(summary)

    paths = {
        "runs": output / "comparison_runs.csv",
        "aggregate": output / "comparison_by_alpha.csv",
    }
    _write_csv(paths["runs"], rows)
    _write_csv(paths["aggregate"], aggregate)
    return paths


def compare_saved_trajectory(
    config: ExperimentConfig,
    endpoints: str | Path,
    alpha: float,
    seed: int,
    method: str,
) -> dict[str, np.ndarray | str]:
    """Return saved observable histories relative to one paired endpoint."""
    if method not in {item.method for item in config.dynamics}:
        raise ValueError(f"Method {method!r} is not in the experiment")
    spec = ProblemSpec(config.N, alpha, config.sigma, config.r, seed)
    problem = LinearRegressionProblem.load(
        _problem_path(Path(config.output_directory), spec)
    )
    if method == "linear_gd":
        reference, regime = _load_linear_endpoint(Path(endpoints), problem, spec)
    else:
        reference, regime = _load_endpoint(
            Path(endpoints), problem, spec, config.kappa
        )
    checkpoint = _checkpoint_path(Path(config.output_directory), spec, method)
    run = DynamicsRun.load_checkpoint(problem, checkpoint)
    if run.config.method != method or run.config.kappa != config.kappa:
        raise ValueError("Checkpoint method or kappa does not match experiment")
    target = problem.observables(reference, config.kappa)
    history: dict[str, np.ndarray | str] = {
        "regime": regime,
        "iteration": np.asarray(run.history["iteration"]),
        "time": np.asarray(run.history["time"]),
    }
    for quantity in ("risk", "loss", "epsilon"):
        history[f"{quantity}_error"] = (
            np.full(len(run.history[quantity]), np.nan)
            if quantity == "epsilon" and method == "linear_gd"
            else np.asarray(run.history[quantity]) - target[quantity]
        )
    return history


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Solve finite-N endpoints or compare saved dynamics"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    solve = commands.add_parser("solve")
    solve.add_argument("config")
    solve.add_argument("output")
    compare = commands.add_parser("compare")
    compare.add_argument("config")
    compare.add_argument("endpoints")
    compare.add_argument("output")
    compare.add_argument(
        "--method",
        action="append",
        choices=("mirror", "gate", "linear_gd"),
        help="Compare only this configured method; repeat to select several",
    )
    args = parser.parse_args(argv)
    config = ExperimentConfig.from_json(args.config)
    paths = (
        solve_saved_endpoints(config, args.output)
        if args.command == "solve"
        else compare_saved_runs(
            config,
            args.endpoints,
            args.output,
            methods=tuple(args.method) if args.method else None,
        )
    )
    print(paths["aggregate"] if "aggregate" in paths else paths["linear_aggregate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
