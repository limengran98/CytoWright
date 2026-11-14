"""Estimate low-rank compound features from response residuals."""
import numpy as np

def fit_basis(chemical, residual, rank, seed):
    X = np.asarray(chemical, dtype=np.float64)
    Y = np.asarray(residual, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError('chemical and residual must be 2-D')
    n, d = X.shape
    n_y, g = Y.shape
    if n_y != n:
        raise ValueError('chemical and residual must have the same number of rows')
    if n <= 0 or d <= 0 or g <= 0:
        raise ValueError('inputs must have positive dimensions')
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise ValueError('inputs must be finite')
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
        raise ValueError('rank must be an integer')
    rank = int(rank)
    if rank <= 0:
        raise ValueError('rank must be positive')
    if rank > min(d, g):
        raise ValueError('rank must not exceed min(d, g)')
    if seed is None:
        raise ValueError('seed must be provided for deterministic sketching')
    C = (X.T @ Y) / float(n)
    sketch_dim = min(rank + 8, d, g)
    rng = np.random.default_rng(seed)
    Omega = rng.standard_normal((g, sketch_dim))
    Q, _ = np.linalg.qr(C @ Omega, mode='reduced')
    for _ in range(2):
        Z = C.T @ Q
        Q, _ = np.linalg.qr(C @ Z, mode='reduced')
    B = Q.T @ C
    U, _, _ = np.linalg.svd(B, full_matrices=False)
    return Q @ U[:, :rank]
