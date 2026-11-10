"""Finite-step mirror, direct-gate, and unconstrained linear GD dynamics.

Mirror and gate methods evolve the same effective weights inside
``||w||_1 < G``, where ``G = N * kappa``.

Mirror descent evolves ``theta in partial Psi(w)`` with

    Psi(w) = ||w||_2^2 / (2 * (1 - ||w||_1/G)).

Direct-gate descent evolves the dimensionless gate ``u=(G/g0)g`` and uses
the declared convention ``sign(0)=0``.  The saved algorithmic time is the
dimensionless time of the corresponding finite-step update, not physical
circuit time.

Linear GD evolves the regression weights directly from zero initialization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.optimize import brentq

from .problem import LinearRegressionProblem
from .storage import atomic_savez, canonical_json, parse_json_scalar

Method = Literal["mirror", "gate", "linear_gd"]


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def _mirror_threshold(theta: np.ndarray, G: float) -> float:
    """Solve lambda=||S_lambda(theta)||^2/(2G).

    The active-set quadratic formula is normally used.  The scalar Brent
    solve is retained as a defensive fallback for roundoff at tied kinks.
    """
    magnitudes = np.sort(np.abs(theta))[::-1]
    if magnitudes.size == 0 or magnitudes[0] == 0.0:
        return 0.0

    sum_1 = np.cumsum(magnitudes)
    sum_2 = np.cumsum(magnitudes**2)
    n = magnitudes.size

    for index in range(n):
        active = index + 1
        a = sum_1[index] + G
        discriminant = max(0.0, a * a - active * sum_2[index])
        denominator = a + sqrt(discriminant)
        lam = sum_2[index] / denominator

        largest_inactive = magnitudes[index + 1] if index + 1 < n else 0.0
        if lam <= magnitudes[index] + 1e-14 and lam >= largest_inactive - 1e-14:
            return float(max(0.0, lam))

    upper = float(magnitudes[0])

    def equation(lam: float) -> float:
        active_vector = soft_threshold(theta, lam)
        return float(active_vector @ active_vector) / (2.0 * G) - lam

    return float(brentq(equation, 0.0, upper, xtol=1e-14, rtol=1e-12))


def inverse_mirror_map(
    theta: np.ndarray, G: float
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return ``(w, lambda, u)`` from the mirror dual coordinate.

    ``u=S_lambda(theta)`` is the dimensionless physical gate and

        w = G u / (G + ||u||_1).
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1 or G <= 0:
        raise ValueError("theta must be one-dimensional and G must be positive")

    lam = _mirror_threshold(theta, G)
    u = soft_threshold(theta, lam)
    w = G * u / (G + np.sum(np.abs(u)))
    return w, lam, u


def effective_weights_from_gate(u: np.ndarray, G: float) -> tuple[np.ndarray, float]:
    """Map the dimensionless gate to effective weights and headroom."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 1 or G <= 0:
        raise ValueError("u must be one-dimensional and G must be positive")
    denominator = G + float(np.sum(np.abs(u)))
    epsilon = G / denominator
    return epsilon * u, epsilon


def project_l1_ball(vector: np.ndarray, radius: float) -> np.ndarray:
    """Euclidean projection onto ``{x: ||x||_1 <= radius}``."""
    vector = np.asarray(vector, dtype=float)
    if vector.ndim != 1 or radius < 0:
        raise ValueError("vector must be one-dimensional and radius nonnegative")
    if np.sum(np.abs(vector)) <= radius:
        return vector.copy()
    if radius == 0:
        return np.zeros_like(vector)

    sorted_abs = np.sort(np.abs(vector))[::-1]
    cumulative = np.cumsum(sorted_abs)
    indices = np.arange(1, vector.size + 1)
    positive = sorted_abs - (cumulative - radius) / indices > 0
    rho = int(np.nonzero(positive)[0][-1])
    threshold = (cumulative[rho] - radius) / (rho + 1)
    return soft_threshold(vector, float(threshold))


def projected_gradient_rms(
    w: np.ndarray,
    gradient: np.ndarray,
    radius: float,
    largest_curvature: float,
) -> float:
    """Mobility-independent KKT residual for the l1-constrained problem.

    This quantity vanishes exactly at constrained least-squares stationary
    points, including interior OLS/interpolating solutions and boundary LASSO
    solutions.
    """
    if largest_curvature <= 0:
        raise ValueError("largest_curvature must be positive")
    step = 1.0 / largest_curvature
    projected = project_l1_ball(w - step * gradient, radius)
    mapping = (w - projected) / step
    return float(np.linalg.norm(mapping) / np.sqrt(w.size))


@dataclass(frozen=True)
class DynamicsConfig:
    method: Method
    kappa: float
    step_size: float | None = None
    step_factor: float | None = None
    record_every: int = 100

    def __post_init__(self) -> None:
        if self.method not in ("mirror", "gate", "linear_gd"):
            raise ValueError("method must be 'mirror', 'gate', or 'linear_gd'")
        if not np.isfinite(self.kappa) or self.kappa <= 0:
            raise ValueError("kappa must be positive and finite")
        if self.step_size is not None and (
            not np.isfinite(self.step_size) or self.step_size <= 0
        ):
            raise ValueError("step_size must be positive and finite")
        if self.step_factor is not None and (
            not np.isfinite(self.step_factor) or self.step_factor <= 0
        ):
            raise ValueError("step_factor must be positive and finite")
        if self.record_every <= 0:
            raise ValueError("record_every must be positive")

    @property
    def default_step_factor(self) -> float:
        return {"mirror": 0.25, "gate": 0.10, "linear_gd": 1.0}[self.method]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DynamicsConfig:
        return cls(
            method=str(data["method"]),
            kappa=float(data["kappa"]),
            step_size=None
            if data.get("step_size") is None
            else float(data["step_size"]),
            step_factor=None
            if data.get("step_factor") is None
            else float(data["step_factor"]),
            record_every=int(data.get("record_every", 100)),
        )


@dataclass(frozen=True)
class ConvergenceConfig:
    """Convergence rule evaluated only every ``check_every`` steps.

    A run passes one check when the projected-gradient KKT residual, relative
    weight change, and relative loss change are all below tolerance.  It is
    declared converged after ``patience`` consecutive passing checks.
    """

    check_every: int = 100
    min_steps: int = 500
    patience: int = 3
    projected_gradient_tol: float = 1e-7
    relative_weight_tol: float = 1e-7
    relative_loss_tol: float = 1e-9

    def __post_init__(self) -> None:
        if self.check_every <= 0 or self.min_steps < 0 or self.patience <= 0:
            raise ValueError("Invalid convergence schedule")
        tolerances = (
            self.projected_gradient_tol,
            self.relative_weight_tol,
            self.relative_loss_tol,
        )
        if any(not np.isfinite(x) or x < 0 for x in tolerances):
            raise ValueError("Convergence tolerances must be finite and nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConvergenceConfig:
        return cls(**{key: data[key] for key in asdict(cls()) if key in data})


HISTORY_NAMES = (
    "iteration",
    "time",
    "time_over_N",
    "m",
    "q",
    "M",
    "risk",
    "test_mse",
    "loss",
    "train_mse",
    "epsilon",
    "active_fraction",
    "gradient_rms",
    "projected_gradient_rms",
    "coordinate_rms",
    "relative_weight_change",
    "relative_loss_change",
    "mirror_lambda",
    "sign_crossings",
    "crossed_fraction",
)


class DynamicsRun:
    """Mutable state of one finite-step learning trajectory."""

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        problem: LinearRegressionProblem,
        config: DynamicsConfig,
        convergence: ConvergenceConfig | None = None,
    ) -> None:
        self.problem = problem
        self.config = config
        self.convergence = convergence or ConvergenceConfig()
        self.G = problem.N * config.kappa

        self._H = problem.hessian
        self._b = problem.linear_term
        self.largest_curvature = float(np.linalg.eigvalsh(self._H)[-1])
        if self.largest_curvature <= 0:
            raise ValueError("The loss has zero curvature in every direction")

        if config.step_size is not None:
            self.step_size = float(config.step_size)
        else:
            factor = (
                config.default_step_factor
                if config.step_factor is None
                else config.step_factor
            )
            self.step_size = float(factor / self.largest_curvature)

        self.coordinate = np.zeros(problem.N, dtype=float)
        self.w, self.mirror_lambda = self._weights_from_coordinate(self.coordinate)
        self.iteration = 0
        self.time = 0.0
        self.converged = False
        self.termination_reason = "not_started"
        self.consecutive_passes = 0
        self.last_check_w = self.w.copy()
        self.last_check_loss = problem.loss(self.w)
        self.sign_crossings = 0
        self.crossed_mask = np.zeros(problem.N, dtype=bool)
        self.history: dict[str, list[float]] = {name: [] for name in HISTORY_NAMES}
        self._record(
            self._diagnostics(self.last_check_w, self.last_check_loss, initial=True)
        )

    def _weights_from_coordinate(
        self, coordinate: np.ndarray
    ) -> tuple[np.ndarray, float]:
        if self.config.method == "mirror":
            w, lam, _ = inverse_mirror_map(coordinate, self.G)
            return w, lam
        if self.config.method == "gate":
            w, _ = effective_weights_from_gate(coordinate, self.G)
            return w, np.nan
        return coordinate.copy(), np.nan

    def _gradient(self) -> np.ndarray:
        return self._H @ self.w - self._b

    def _step_once(self) -> None:
        gradient = self._gradient()

        if self.config.method == "mirror":
            self.coordinate = self.coordinate - self.step_size * gradient
        elif self.config.method == "gate":
            old_sign = np.sign(self.coordinate)
            _, epsilon = effective_weights_from_gate(self.coordinate, self.G)
            z = old_sign  # NumPy declares sign(0)=0.
            gate_gradient = epsilon * (
                gradient - z * (float(self.w @ gradient) / self.G)
            )
            new_coordinate = self.coordinate - self.step_size * gate_gradient
            new_sign = np.sign(new_coordinate)
            crossings = old_sign * new_sign < 0
            self.sign_crossings += int(np.count_nonzero(crossings))
            self.crossed_mask |= crossings
            self.coordinate = new_coordinate
        else:
            self.coordinate = self.coordinate - self.step_size * gradient

        self.w, self.mirror_lambda = self._weights_from_coordinate(self.coordinate)
        self.iteration += 1
        self.time += self.step_size

    def _diagnostics(
        self,
        reference_w: np.ndarray,
        reference_loss: float,
        *,
        initial: bool = False,
    ) -> dict[str, float]:
        values = self.problem.observables(self.w, self.config.kappa)
        gradient = self._gradient()
        if self.config.method == "linear_gd":
            values["epsilon"] = np.nan
            values["projected_gradient_rms"] = float(
                np.linalg.norm(gradient) / np.sqrt(self.problem.N)
            )
        else:
            values["projected_gradient_rms"] = projected_gradient_rms(
                self.w,
                gradient,
                self.G,
                self.largest_curvature,
            )
        values["coordinate_rms"] = float(
            np.linalg.norm(self.coordinate) / np.sqrt(self.problem.N)
        )

        if initial:
            relative_weight_change = np.inf
            relative_loss_change = np.inf
        else:
            rms_change = np.linalg.norm(self.w - reference_w) / np.sqrt(self.problem.N)
            rms_scale = max(1.0, np.linalg.norm(self.w) / np.sqrt(self.problem.N))
            relative_weight_change = float(rms_change / rms_scale)
            relative_loss_change = float(
                abs(values["loss"] - reference_loss) / max(1.0, abs(values["loss"]))
            )

        values.update(
            {
                "iteration": float(self.iteration),
                "time": self.time,
                "time_over_N": self.time / self.problem.N,
                "relative_weight_change": relative_weight_change,
                "relative_loss_change": relative_loss_change,
                "mirror_lambda": float(self.mirror_lambda),
                "sign_crossings": float(self.sign_crossings),
                "crossed_fraction": float(np.mean(self.crossed_mask)),
            }
        )
        return values

    def _record(self, diagnostics: dict[str, float]) -> None:
        same_iteration = bool(
            self.history["iteration"]
            and int(self.history["iteration"][-1]) == self.iteration
        )
        for name in HISTORY_NAMES:
            value = float(diagnostics[name])
            if same_iteration:
                self.history[name][-1] = value
            else:
                self.history[name].append(value)
        self.latest_diagnostics = dict(diagnostics)

    def _check_convergence(self) -> None:
        diagnostics = self._diagnostics(self.last_check_w, self.last_check_loss)
        eligible = self.iteration >= self.convergence.min_steps
        passed = eligible and (
            diagnostics["projected_gradient_rms"]
            <= self.convergence.projected_gradient_tol
            and diagnostics["relative_weight_change"]
            <= self.convergence.relative_weight_tol
            and diagnostics["relative_loss_change"]
            <= self.convergence.relative_loss_tol
        )

        self.consecutive_passes = self.consecutive_passes + 1 if passed else 0
        self._record(diagnostics)
        self.last_check_w = self.w.copy()
        self.last_check_loss = diagnostics["loss"]

        if self.consecutive_passes >= self.convergence.patience:
            self.converged = True
            self.termination_reason = (
                "gradient_and_stability"
                if self.config.method == "linear_gd"
                else "projected_kkt_and_stability"
            )

    def advance(
        self,
        additional_steps: int,
        *,
        checkpoint_path: str | Path | None = None,
        checkpoint_every: int | None = None,
    ) -> int:
        """Advance by at most ``additional_steps`` and optionally checkpoint.

        A converged run is left unchanged.  The return value is the number of
        steps actually taken.
        """
        if int(additional_steps) != additional_steps or additional_steps < 0:
            raise ValueError("additional_steps must be a nonnegative integer")
        if checkpoint_every is not None and checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        if self.converged or additional_steps == 0:
            if checkpoint_path is not None:
                self.save_checkpoint(checkpoint_path)
            return 0

        start_iteration = self.iteration
        self.termination_reason = "running"

        try:
            for _ in range(int(additional_steps)):
                self._step_once()

                checked = self.iteration % self.convergence.check_every == 0
                if checked:
                    self._check_convergence()
                elif self.iteration % self.config.record_every == 0:
                    self._record(
                        self._diagnostics(self.last_check_w, self.last_check_loss)
                    )

                if (
                    checkpoint_path is not None
                    and checkpoint_every is not None
                    and self.iteration % checkpoint_every == 0
                ):
                    self.save_checkpoint(checkpoint_path)

                if self.converged:
                    break
        except KeyboardInterrupt:
            self.termination_reason = "interrupted"
            if (
                not self.history["iteration"]
                or int(self.history["iteration"][-1]) != self.iteration
            ):
                self._record(self._diagnostics(self.last_check_w, self.last_check_loss))
            if checkpoint_path is not None:
                self.save_checkpoint(checkpoint_path)
            raise

        if not self.converged:
            self.termination_reason = "step_budget_exhausted"

        if (
            not self.history["iteration"]
            or int(self.history["iteration"][-1]) != self.iteration
        ):
            self._record(self._diagnostics(self.last_check_w, self.last_check_loss))
        if checkpoint_path is not None:
            self.save_checkpoint(checkpoint_path)
        return self.iteration - start_iteration

    def final_observables(self) -> dict[str, float]:
        values = self._diagnostics(self.last_check_w, self.last_check_loss)
        if (
            self.history["iteration"]
            and int(self.history["iteration"][-1]) == self.iteration
        ):
            values["relative_weight_change"] = self.history["relative_weight_change"][
                -1
            ]
            values["relative_loss_change"] = self.history["relative_loss_change"][-1]
        return values

    def save_checkpoint(self, path: str | Path) -> None:
        metadata = {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "problem_fingerprint": self.problem.fingerprint(),
            "config": self.config.to_dict(),
            "convergence": self.convergence.to_dict(),
            "iteration": self.iteration,
            "time": self.time,
            "step_size": self.step_size,
            "largest_curvature": self.largest_curvature,
            "converged": self.converged,
            "termination_reason": self.termination_reason,
            "consecutive_passes": self.consecutive_passes,
            "last_check_loss": self.last_check_loss,
            # JSON metadata stays standards-compliant.  Other methods have no
            # mirror threshold, so represent that absence by ``null`` rather
            # than the non-standard JSON token ``NaN``.
            "mirror_lambda": (
                float(self.mirror_lambda) if np.isfinite(self.mirror_lambda) else None
            ),
            "sign_crossings": self.sign_crossings,
        }
        arrays: dict[str, Any] = {
            "metadata": np.array(canonical_json(metadata)),
            "coordinate": self.coordinate,
            "w": self.w,
            "last_check_w": self.last_check_w,
            "crossed_mask": self.crossed_mask.astype(np.uint8),
        }
        arrays.update(
            {
                f"history__{name}": np.asarray(values, dtype=float)
                for name, values in self.history.items()
            }
        )
        atomic_savez(path, **arrays)

    @classmethod
    def load_checkpoint(
        cls,
        problem: LinearRegressionProblem,
        path: str | Path,
    ) -> DynamicsRun:
        with np.load(path, allow_pickle=False) as archive:
            metadata = parse_json_scalar(archive["metadata"])
            if metadata["problem_fingerprint"] != problem.fingerprint():
                raise ValueError("Checkpoint belongs to a different training problem")
            if int(metadata["checkpoint_version"]) != cls.CHECKPOINT_VERSION:
                raise ValueError("Unsupported checkpoint version")

            run = cls(
                problem=problem,
                config=DynamicsConfig.from_dict(metadata["config"]),
                convergence=ConvergenceConfig.from_dict(metadata["convergence"]),
            )
            run.coordinate = archive["coordinate"].copy()
            run.w = archive["w"].copy()
            run.last_check_w = archive["last_check_w"].copy()
            run.crossed_mask = archive["crossed_mask"].astype(bool)
            run.history = {
                name: archive[f"history__{name}"].astype(float).tolist()
                for name in HISTORY_NAMES
            }

        run.iteration = int(metadata["iteration"])
        run.time = float(metadata["time"])
        run.step_size = float(metadata["step_size"])
        run.largest_curvature = float(metadata["largest_curvature"])
        run.converged = bool(metadata["converged"])
        run.termination_reason = str(metadata["termination_reason"])
        run.consecutive_passes = int(metadata["consecutive_passes"])
        run.last_check_loss = float(metadata["last_check_loss"])
        run.mirror_lambda = (
            np.nan
            if metadata["mirror_lambda"] is None
            else float(metadata["mirror_lambda"])
        )
        run.sign_crossings = int(metadata["sign_crossings"])

        reconstructed_w, reconstructed_lambda = run._weights_from_coordinate(
            run.coordinate
        )
        if not np.allclose(reconstructed_w, run.w, rtol=1e-11, atol=1e-12):
            raise ValueError(
                "Checkpoint coordinate and effective weights are inconsistent"
            )
        run.w = reconstructed_w
        run.mirror_lambda = reconstructed_lambda
        run.latest_diagnostics = {
            name: run.history[name][-1] for name in HISTORY_NAMES if run.history[name]
        }
        return run
