"""Definition and persistence of one teacher--student regression problem.

The problem object is immutable in practice: generate it once, save it, and
pass the same object to every learning dynamics that should be compared.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .storage import atomic_savez, canonical_json, parse_json_scalar


@dataclass(frozen=True)
class ProblemSpec:
    """Parameters defining one random Gaussian teacher--student problem.

    The convention throughout the repository is ``alpha = N / P`` and
    ``X[mu, i] ~ Normal(0, 1/N)``.
    """

    N: int
    alpha: float
    sigma: float = 0.0
    r: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if int(self.N) != self.N or self.N <= 0:
            raise ValueError("N must be a positive integer")
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be positive and finite")
        if not np.isfinite(self.sigma) or self.sigma < 0:
            raise ValueError("sigma must be nonnegative and finite")
        if not np.isfinite(self.r) or self.r < 0:
            raise ValueError("r must be nonnegative and finite")
        if int(self.seed) != self.seed:
            raise ValueError("seed must be an integer")

    @property
    def P(self) -> int:
        return max(1, round(self.N / self.alpha))

    @property
    def realized_alpha(self) -> float:
        return self.N / self.P

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["N"] = int(self.N)
        result["seed"] = int(self.seed)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProblemSpec:
        return cls(
            N=int(data["N"]),
            alpha=float(data["alpha"]),
            sigma=float(data.get("sigma", 0.0)),
            r=float(data.get("r", 1.0)),
            seed=int(data.get("seed", 0)),
        )


@dataclass
class LinearRegressionProblem:
    """A fixed training realization shared by one or more dynamics."""

    spec: ProblemSpec
    X: np.ndarray
    y: np.ndarray
    w_star: np.ndarray
    noise: np.ndarray

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        self.w_star = np.asarray(self.w_star, dtype=float)
        self.noise = np.asarray(self.noise, dtype=float)

        expected_shape = (self.spec.P, self.spec.N)
        if self.X.shape != expected_shape:
            raise ValueError(f"X has shape {self.X.shape}; expected {expected_shape}")
        if self.y.shape != (self.spec.P,):
            raise ValueError("y has the wrong shape")
        if self.w_star.shape != (self.spec.N,):
            raise ValueError("w_star has the wrong shape")
        if self.noise.shape != (self.spec.P,):
            raise ValueError("noise has the wrong shape")
        if not all(
            np.all(np.isfinite(x)) for x in (self.X, self.y, self.w_star, self.noise)
        ):
            raise ValueError("Problem arrays must contain only finite values")

    @classmethod
    def generate(cls, spec: ProblemSpec) -> LinearRegressionProblem:
        rng = np.random.default_rng(spec.seed)
        X = rng.normal(size=(spec.P, spec.N)) / np.sqrt(spec.N)
        w_star = rng.normal(scale=spec.r, size=spec.N)
        noise = rng.normal(scale=spec.sigma, size=spec.P)
        y = X @ w_star + noise
        return cls(spec=spec, X=X, y=y, w_star=w_star, noise=noise)

    @property
    def N(self) -> int:
        return self.spec.N

    @property
    def P(self) -> int:
        return self.spec.P

    @property
    def alpha(self) -> float:
        return self.spec.realized_alpha

    @property
    def hessian(self) -> np.ndarray:
        return self.X.T @ self.X / self.P

    @property
    def linear_term(self) -> np.ndarray:
        return self.X.T @ self.y / self.P

    def residual(self, w: np.ndarray) -> np.ndarray:
        return self.X @ np.asarray(w, dtype=float) - self.y

    def gradient(self, w: np.ndarray) -> np.ndarray:
        residual = self.residual(w)
        return self.X.T @ residual / self.P

    def loss(self, w: np.ndarray) -> float:
        residual = self.residual(w)
        return 0.5 * float(residual @ residual) / self.P

    def observables(self, w: np.ndarray, kappa: float) -> dict[str, float]:
        """Finite-size counterparts of the thermodynamic observables."""
        w = np.asarray(w, dtype=float)
        if w.shape != (self.N,):
            raise ValueError("w has the wrong shape")
        if kappa <= 0:
            raise ValueError("kappa must be positive")

        residual = self.residual(w)
        gradient = self.X.T @ residual / self.P
        m = float(np.mean(np.abs(w)))
        risk = float(np.mean((w - self.w_star) ** 2))
        train_mse = float(np.mean(residual**2))

        return {
            "m": m,
            "q": float(np.mean(w**2)),
            "M": float(np.mean(w * self.w_star)),
            "risk": risk,
            "test_mse": risk + self.spec.sigma**2,
            "loss": 0.5 * train_mse,
            "train_mse": train_mse,
            "epsilon": 1.0 - m / kappa,
            "active_fraction": float(np.mean(np.abs(w) > 1e-12)),
            "gradient_rms": float(np.linalg.norm(gradient) / np.sqrt(self.N)),
        }

    def fingerprint(self) -> str:
        """Content hash used to prevent resuming on the wrong dataset."""
        digest = hashlib.sha256(canonical_json(self.spec.to_dict()).encode())
        for array in (self.X, self.y, self.w_star, self.noise):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.shape).encode())
            digest.update(contiguous.dtype.str.encode())
            digest.update(contiguous.view(np.uint8))
        return digest.hexdigest()

    def save(self, path: str | Path) -> None:
        atomic_savez(
            path,
            metadata=np.array(canonical_json(self.spec.to_dict())),
            fingerprint=np.array(self.fingerprint()),
            X=self.X,
            y=self.y,
            w_star=self.w_star,
            noise=self.noise,
        )

    @classmethod
    def load(cls, path: str | Path) -> LinearRegressionProblem:
        with np.load(path, allow_pickle=False) as archive:
            spec = ProblemSpec.from_dict(parse_json_scalar(archive["metadata"]))
            problem = cls(
                spec=spec,
                X=archive["X"].copy(),
                y=archive["y"].copy(),
                w_star=archive["w_star"].copy(),
                noise=archive["noise"].copy(),
            )
            stored_fingerprint = str(archive["fingerprint"].item())

        if problem.fingerprint() != stored_fingerprint:
            raise ValueError(f"Dataset fingerprint mismatch in {path}")
        return problem
