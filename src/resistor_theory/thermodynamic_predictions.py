"""Thermodynamic predictions for the Gaussian teacher--student model.

Conventions
-----------
alpha = N / P
X[mu, i] ~ Normal(0, 1/N)
W_star[i] ~ Normal(0, r**2)
noise[mu] ~ Normal(0, sigma**2)

The returned ``risk`` is the parameter-estimation error

    ||w_hat - w_star||_2**2 / N,

whereas ``test_mse = risk + sigma**2`` includes fresh test-label noise.
The training loss is normalized as ||X w_hat - y||_2**2 / (2 P).

This is intentionally a small scalar module: no classes and no simulation code.
"""

from math import erf, erfc, exp, inf, isfinite, pi, sqrt

from scipy.optimize import brentq
from scipy.special import erfcinv


SQRT_2 = sqrt(2.0)
SQRT_2PI = sqrt(2.0 * pi)


# -----------------------------------------------------------------------------
# Gaussian functions
# -----------------------------------------------------------------------------

def phi(t):
    """Standard-normal density."""
    return exp(-0.5 * t * t) / SQRT_2PI


def Q(t):
    """Standard-normal upper-tail probability."""
    return 0.5 * erfc(t / SQRT_2)


def CC(t):
    """Integral of (z-t)^2 phi(z) over z >= t."""
    return (1.0 + t * t) * Q(t) - t * phi(t)


def t_alpha(alpha):
    """Solve 2 alpha Q(t_alpha) = 1; defined for alpha >= 1."""
    if alpha < 1.0:
        raise ValueError("t_alpha is defined only for alpha >= 1")
    return SQRT_2 * erfcinv(1.0 / alpha)


# -----------------------------------------------------------------------------
# LASSO boundary solution
# -----------------------------------------------------------------------------

def s2_lasso(t, alpha, sigma, r):
    """Variance s^2 of the scalar Gaussian field in the LASSO branch."""
    # At this single point the original expression is 0/0, with limit r^2.
    if t == 0.0 and alpha == 1.0:
        return r * r if sigma == 0.0 else inf

    # These are algebraically identical to the Mathematica numerator and
    # denominator, but avoid cancellation near alpha=1 and t=0.
    e = erf(t / SQRT_2)
    numerator = (
        (1.0 - alpha) * r * r
        + alpha * sigma * sigma
        + 2.0 * alpha * r * r * e
    )
    denominator = (
        (1.0 - alpha)
        - alpha * t * t
        + alpha * (1.0 + t * t) * e
        + 2.0 * alpha * t * phi(t)
    )
    return numerator / denominator


def q_lasso(t, alpha, sigma, r):
    return 2.0 * s2_lasso(t, alpha, sigma, r) * CC(t)


def m_lasso(t, alpha, sigma, r):
    s2 = s2_lasso(t, alpha, sigma, r)
    return 2.0 * sqrt(s2) * (phi(t) - t * Q(t))


def M_lasso(t, alpha, sigma, r):
    return 2.0 * r * r * Q(t)


def risk_lasso(t, alpha, sigma, r):
    return q_lasso(t, alpha, sigma, r) + r * r * (1.0 - 4.0 * Q(t))


def loss_lasso(t, alpha, sigma, r):
    risk = risk_lasso(t, alpha, sigma, r)
    return 0.5 * (1.0 - 2.0 * alpha * Q(t)) ** 2 * (
        sigma * sigma + risk
    )


# -----------------------------------------------------------------------------
# Interior Bregman solution (alpha > 1)
# -----------------------------------------------------------------------------

def s2_bregman(t, alpha, sigma, r):
    qt = Q(t)
    numerator = (
        2.0
        * alpha
        * qt**2
        * (r * r * (alpha - 1.0) + alpha * sigma * sigma)
    )
    denominator = 2.0 * alpha * qt**2 - CC(t)
    return numerator / denominator


def q_bregman(t, alpha, sigma, r):
    qt = Q(t)
    return s2_bregman(t, alpha, sigma, r) * CC(t) / (
        2.0 * alpha**2 * qt**2
    )


def m_bregman(t, alpha, sigma, r):
    qt = Q(t)
    s2 = s2_bregman(t, alpha, sigma, r)
    return sqrt(s2) * (phi(t) - t * qt) / (alpha * qt)


def M_bregman(t, alpha, sigma, r):
    return r * r / alpha


def risk_bregman(t, alpha, sigma, r):
    return q_bregman(t, alpha, sigma, r) + r * r * (1.0 - 2.0 / alpha)


def kappa_bregman(t, alpha, sigma, r):
    if t <= 0.0:
        raise ValueError("kappa_bregman requires t > 0")
    s2 = s2_bregman(t, alpha, sigma, r)
    q = q_bregman(t, alpha, sigma, r)
    return m_bregman(t, alpha, sigma, r) + (
        (2.0 * alpha * Q(t) - 1.0) * q / (2.0 * sqrt(s2) * t)
    )


# -----------------------------------------------------------------------------
# OLS solution (alpha < 1)
# -----------------------------------------------------------------------------

def q_ols(alpha, sigma, r):
    return r * r + sigma * sigma * alpha / (1.0 - alpha)


def m_ols(alpha, sigma, r):
    return sqrt(2.0 / pi) * sqrt(q_ols(alpha, sigma, r))


def M_ols(alpha, sigma, r):
    return r * r


def risk_ols(alpha, sigma, r):
    return sigma * sigma * alpha / (1.0 - alpha)


def loss_ols(alpha, sigma, r):
    return 0.5 * sigma * sigma * (1.0 - alpha)


# -----------------------------------------------------------------------------
# Branch selection and scalar root finding
# -----------------------------------------------------------------------------

def critical_kappa(alpha, sigma, r):
    """Radius separating the two relevant regimes."""
    if alpha < 1.0:
        return m_lasso(0.0, alpha, sigma, r)
    if alpha > 1.0:
        tc = t_alpha(alpha)
        return m_lasso(tc, alpha, sigma, r)
    return r * sqrt(2.0 / pi) if sigma == 0.0 else inf


def t_boundary(k, alpha, sigma, r):
    """Solve m_lasso(t)=k on the physical LASSO branch."""
    if k <= 0.0:
        raise ValueError("t_boundary requires k > 0")

    lower = t_alpha(alpha) if alpha > 1.0 else 0.0

    def f(t):
        return m_lasso(t, alpha, sigma, r) - k

    upper = max(1.0, 2.0 * lower)
    for _ in range(100):
        if f(upper) <= 0.0:
            break
        upper *= 2.0
    else:
        raise RuntimeError("Could not bracket the LASSO root")

    return brentq(f, lower, upper, xtol=1e-12, rtol=1e-12)


def t_mirror(k, alpha, sigma, r):
    """Solve kappa_bregman(t)=k on 0 < t < t_alpha."""
    if alpha <= 1.0:
        raise ValueError("t_mirror requires alpha > 1")

    tc = t_alpha(alpha)

    def f(t):
        return kappa_bregman(t, alpha, sigma, r) - k

    lower = 0.5 * tc
    for _ in range(100):
        if f(lower) >= 0.0:
            break
        lower *= 0.5
    else:
        raise RuntimeError("Could not bracket the Bregman root")

    return brentq(f, lower, tc, xtol=1e-12, rtol=1e-12)


def _result(regime, t, kappa_c, s2, m, q, M, risk, loss, sigma):
    """Use a dictionary so results are easy to inspect or store in a table."""
    return {
        "regime": regime,
        "t": t,
        "kappa_c": kappa_c,
        "s2": s2,
        "m": m,
        "q": q,
        "M": M,
        "risk": risk,
        "test_mse": risk + sigma * sigma,
        "loss": loss,
    }


def predict(k, alpha, sigma, r):
    """Return all thermodynamic predictions for one parameter set.

    Parameters are scalars.  In particular, ``k`` is the normalized l1-ball
    radius and ``alpha = N/P``.  ``k=inf`` gives unconstrained minimum-norm
    linear regression.
    """
    k, alpha, sigma, r = map(float, (k, alpha, sigma, r))
    if not (isfinite(k) or k == inf) or not all(isfinite(x) for x in (alpha, sigma, r)):
        raise ValueError("Inputs must be finite except for k=inf")
    if k < 0.0 or alpha <= 0.0 or sigma < 0.0 or r < 0.0:
        raise ValueError("Require k>=0, alpha>0, sigma>=0 and r>=0")

    kc = critical_kappa(alpha, sigma, r)

    if k == inf:
        if alpha < 1.0:
            q = q_ols(alpha, sigma, r)
            return _result(
                "ols", 0.0, kc, q, m_ols(alpha, sigma, r), q,
                M_ols(alpha, sigma, r), risk_ols(alpha, sigma, r),
                loss_ols(alpha, sigma, r), sigma
            )
        if alpha == 1.0:
            q = r * r if sigma == 0.0 else inf
            return _result(
                "interpolation_threshold", 0.0, kc, q,
                sqrt(2.0 / pi) * sqrt(q), q, r * r,
                0.0 if sigma == 0.0 else inf, 0.0, sigma
            )
        q = r * r / alpha + sigma * sigma / (alpha - 1.0)
        return _result(
            "minimum_norm_interpolator", 0.0, kc, q,
            sqrt(2.0 / pi) * sqrt(q), q, r * r / alpha,
            r * r * (1.0 - 1.0 / alpha) + sigma * sigma / (alpha - 1.0),
            0.0, sigma
        )

    # At k=0, the estimator is exactly zero and no finite threshold exists.
    if k == 0.0:
        risk = r * r
        return _result(
            "zero", inf, kc, (1.0 + alpha) * r * r + alpha * sigma * sigma,
            0.0, 0.0, 0.0, risk, 0.5 * (r * r + sigma * sigma), sigma
        )

    if alpha < 1.0:
        if k >= kc:
            q = q_ols(alpha, sigma, r)
            risk = risk_ols(alpha, sigma, r)
            return _result(
                "ols", 0.0, kc, q, m_ols(alpha, sigma, r), q,
                M_ols(alpha, sigma, r), risk,
                loss_ols(alpha, sigma, r), sigma
            )
        t = t_boundary(k, alpha, sigma, r)
        return _lasso_result(t, kc, alpha, sigma, r, "lasso_boundary")

    if alpha > 1.0:
        tc = t_alpha(alpha)
        if k > kc:
            t = t_mirror(k, alpha, sigma, r)
            return _bregman_result(t, kc, alpha, sigma, r)
        if k < kc:
            t = t_boundary(k, alpha, sigma, r)
            return _lasso_result(t, kc, alpha, sigma, r, "lasso_boundary")
        return _lasso_result(tc, kc, alpha, sigma, r, "critical")

    # alpha == 1: the noiseless square problem has the unique solution w_star.
    if sigma == 0.0 and k >= kc:
        return _result(
            "exact_fit", 0.0, kc, r * r, r * sqrt(2.0 / pi),
            r * r, r * r, 0.0, 0.0, sigma
        )

    t = t_boundary(k, alpha, sigma, r)
    return _lasso_result(t, kc, alpha, sigma, r, "lasso_boundary")


def _lasso_result(t, kc, alpha, sigma, r, regime):
    s2 = s2_lasso(t, alpha, sigma, r)
    q = q_lasso(t, alpha, sigma, r)
    M = M_lasso(t, alpha, sigma, r)
    risk = q + r * r - 2.0 * M
    return _result(
        regime, t, kc, s2, m_lasso(t, alpha, sigma, r), q, M,
        risk, loss_lasso(t, alpha, sigma, r), sigma
    )


def _bregman_result(t, kc, alpha, sigma, r):
    s2 = s2_bregman(t, alpha, sigma, r)
    q = q_bregman(t, alpha, sigma, r)
    M = M_bregman(t, alpha, sigma, r)
    risk = q + r * r - 2.0 * M
    return _result(
        "bregman", t, kc, s2, m_bregman(t, alpha, sigma, r),
        q, M, risk, 0.0, sigma
    )
