"""Trace-scaled ridge regression for readout and compound-route updates."""
import numpy as np


def solve_ridge(design, residual, strength, stabilizer=1e-8):
    """Solve the ridge normal equations using the smaller Gram matrix.

    The penalty is strength * trace(X.T @ X) / feature_count + stabilizer.
    Both a residual vector and a matrix of response residuals are supported.
    """
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(residual, dtype=np.float64)
    samples, features = x.shape
    mean_energy = float(np.sum(np.square(x))) / max(features, 1)
    penalty = float(strength) * mean_energy + stabilizer
    dual = samples <= features
    gram = x @ x.T if dual else x.T @ x
    system = gram + penalty * np.eye(gram.shape[0])
    right = y if dual else x.T @ y
    try:
        solution = np.linalg.solve(system, right)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(system, right, rcond=None)[0]
    return x.T @ solution if dual else solution
