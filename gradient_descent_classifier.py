"""
Custom full-batch gradient-descent logistic classifier (sklearn compatible).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class GradientDescentLogisticClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 500,
        tol: float = 1e-4,
        alpha: float = 0.0,
        fit_intercept: bool = True,
        verbose: int = 0,
        random_state: int | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.random_state = random_state

    def _initialize(self, n_features: int) -> None:
        rng = np.random.default_rng(self.random_state)
        self.coef_ = rng.normal(0, 0.01, size=n_features)
        self.intercept_ = 0.0
        self.loss_history_: list[float] = []
        self.classes_ = np.array([0, 1])

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self, X: ArrayLike, y: ArrayLike) -> "GradientDescentLogisticClassifier":
        X, y = check_X_y(X, y)
        if np.unique(y).size < 2:
            raise ValueError("GradientDescentLogisticClassifier requires two classes.")
        n_samples, n_features = X.shape
        self._initialize(n_features)

        weights = self.coef_.copy()
        bias = self.intercept_

        for iteration in range(self.max_iter):
            logits = X @ weights + (bias if self.fit_intercept else 0.0)
            probs = 1 / (1 + np.exp(-logits))
            residual = probs - y
            grad_w = (X.T @ residual) / n_samples + self.alpha * weights
            grad_b = residual.mean() if self.fit_intercept else 0.0

            weights -= self.learning_rate * grad_w
            if self.fit_intercept:
                bias -= self.learning_rate * grad_b

            loss = -np.mean(y * np.log(probs + 1e-12) + (1 - y) * np.log(1 - probs + 1e-12))
            loss += (self.alpha / 2.0) * np.sum(weights ** 2)
            self.loss_history_.append(float(loss))

            grad_norm = np.linalg.norm(grad_w)
            if self.verbose and iteration % 50 == 0:
                print(f"[gd] iter={iteration} loss={loss:.4f} grad_norm={grad_norm:.5f}")
            if grad_norm < self.tol:
                break

        self.coef_ = weights
        self.intercept_ = bias if self.fit_intercept else 0.0
        self.n_features_in_ = n_features
        return self

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "coef_")
        X = check_array(X)
        return X @ self.coef_ + (self.intercept_ if self.fit_intercept else 0.0)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        scores = self.decision_function(X)
        probs = 1 / (1 + np.exp(-scores))
        return np.vstack([1 - probs, probs]).T

    def predict(self, X: ArrayLike) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X))


__all__ = ["GradientDescentLogisticClassifier"]
