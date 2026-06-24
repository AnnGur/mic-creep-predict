"""
Tobit regression for left- and right-censored MIC outcomes.

Standard Tobit model (Tobin 1958):
  latent:   y* = Xβ + ε,   ε ~ N(0, σ²)
  observed: y = y*  if  L < y* < U
            y = L   if  y* ≤ L  (left-censored — MIC at panel floor)
            y = U   if  y* ≥ U  (right-censored — MIC above panel ceiling)

Log-likelihood:
  ℓ(β, σ) = Σ_uncensored  log φ((y − Xβ)/σ) − log σ
           + Σ_left        log Φ((L − Xβ)/σ)
           + Σ_right       log Φ((Xβ − U)/σ)

Gradients are supplied analytically for efficient L-BFGS-B convergence.

Prediction:  E[y*] = Xβ  (latent mean — use for trend surveillance)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression


class TobitRegressor(BaseEstimator, RegressorMixin):
    """
    Maximum-likelihood Tobit regression for doubly-censored continuous outcomes.

    Parameters
    ----------
    lower : float or None
        Left censoring limit in the target space (e.g. log2(0.06) = -4.059).
        Observations flagged left-censored contribute log Φ((L - Xβ)/σ) to ℓ.
    upper : float or None
        Right censoring limit (e.g. log2(32) = 5.0).
    alpha : float
        L2 regularisation on β (ridge penalty). Helps stability with many OHE columns.
    max_iter : int
        Maximum L-BFGS-B iterations.
    """

    def __init__(
        self,
        lower: float | None = None,
        upper: float | None = None,
        alpha: float = 1e-4,
        max_iter: int = 500,
    ):
        self.lower = lower
        self.upper = upper
        self.alpha = alpha
        self.max_iter = max_iter

    # ------------------------------------------------------------------
    # Internal: negative log-likelihood + analytical gradient
    # ------------------------------------------------------------------

    def _nll_and_grad(
        self,
        params: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        beta = params[:-1]
        log_sigma = params[-1]
        sigma = np.exp(log_sigma)

        mu = X @ beta
        uncensored = ~left_mask & ~right_mask

        nll = 0.0
        grad_beta = np.zeros_like(beta)
        grad_log_sigma = 0.0

        # --- Uncensored observations ---
        if uncensored.any():
            z = (y[uncensored] - mu[uncensored]) / sigma
            nll += (-norm.logpdf(z) + log_sigma).sum()
            # ∂NLL/∂β = (1/σ²) Xᵀ(μ − y)
            grad_beta += (X[uncensored].T @ ((mu[uncensored] - y[uncensored]) / sigma**2))
            # ∂NLL/∂log_σ = Σ (1 − z²)
            grad_log_sigma += (1.0 - z**2).sum()

        # --- Left-censored: contribute log Φ((L − μ)/σ) ---
        if left_mask.any() and self.lower is not None:
            z_L = (self.lower - mu[left_mask]) / sigma
            nll -= norm.logcdf(z_L).sum()
            mills_L = norm.pdf(z_L) / np.maximum(norm.cdf(z_L), 1e-300)
            # ∂NLL/∂β = (1/σ) Xᵀ mills_L
            grad_beta += (1.0 / sigma) * (X[left_mask].T @ mills_L)
            # ∂NLL/∂log_σ = Σ z_L · mills_L
            grad_log_sigma += (z_L * mills_L).sum()

        # --- Right-censored: contribute log(1 − Φ((U − μ)/σ)) ---
        if right_mask.any() and self.upper is not None:
            z_U = (self.upper - mu[right_mask]) / sigma
            nll -= norm.logsf(z_U).sum()
            mills_U = norm.pdf(z_U) / np.maximum(norm.sf(z_U), 1e-300)
            # ∂NLL/∂β = −(1/σ) Xᵀ mills_U
            grad_beta -= (1.0 / sigma) * (X[right_mask].T @ mills_U)
            # ∂NLL/∂log_σ = −Σ z_U · mills_U
            grad_log_sigma -= (z_U * mills_U).sum()

        # L2 regularisation on β (not on log_sigma)
        if self.alpha > 0:
            nll += 0.5 * self.alpha * (beta**2).sum()
            grad_beta += self.alpha * beta

        return nll, np.append(grad_beta, grad_log_sigma)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        left_censored_mask: np.ndarray | None = None,
        right_censored_mask: np.ndarray | None = None,
    ) -> "TobitRegressor":
        """
        Fit the Tobit model via MLE (L-BFGS-B).

        Parameters
        ----------
        X : (n, p) feature matrix — no intercept column needed, added internally.
        y : (n,)   observed / imputed target values (log2 MIC).
        left_censored_mask  : bool (n,) — True where y* ≤ lower.
        right_censored_mask : bool (n,) — True where y* ≥ upper.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = len(y)

        # Prepend intercept
        X_int = np.column_stack([np.ones(n), X])

        left_mask = (
            np.zeros(n, dtype=bool)
            if left_censored_mask is None
            else np.asarray(left_censored_mask, dtype=bool)
        )
        right_mask = (
            np.zeros(n, dtype=bool)
            if right_censored_mask is None
            else np.asarray(right_censored_mask, dtype=bool)
        )

        # Initialise with OLS (gives good β starting point)
        ols = LinearRegression(fit_intercept=False).fit(X_int, y)
        beta_init = ols.coef_
        sigma_init = max(np.std(y - X_int @ beta_init), 0.1)
        params_init = np.append(beta_init, np.log(sigma_init))

        result = minimize(
            self._nll_and_grad,
            params_init,
            args=(X_int, y, left_mask, right_mask),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter, "ftol": 1e-10, "gtol": 1e-6},
        )

        self.coef_ = result.x[1:-1]       # slope coefficients (without intercept)
        self.intercept_ = result.x[0]
        self.sigma_ = np.exp(result.x[-1])
        self.converged_ = result.success
        self.nll_ = result.fun
        self._X_int = X_int                # kept for diagnostics only
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return latent mean E[y*] = intercept + Xβ."""
        X = np.asarray(X, dtype=np.float64)
        return self.intercept_ + X @ self.coef_
