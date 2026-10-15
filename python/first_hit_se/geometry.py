"""Stationary SE full-gradient reconstruction for signed first-zero crossings."""
import numpy as np


def mean_gradient(profile, q):
    features = profile["features"].cpu().numpy()
    ends = np.cumsum(np.exp(features[:, 4]))
    i = min(int(np.searchsorted(ends, q)), len(ends)-1)
    lo = 0. if i == 0 else ends[i-1]
    u = np.clip((q-lo)/(ends[i]-lo), 0., 1.)
    coefficients = np.asarray(profile["geometry"]["mean_gradient_polynomials"])[i]
    return (coefficients[:, 0]*u+coefficients[:, 1])*u+coefficients[:, 2]


def sample_gradient(geometry, mean, signed_derivative, rng):
    direction = np.asarray(geometry["direction"], dtype=float)
    if not np.isclose(direction @ direction, 1., atol=1e-10):
        raise ValueError("gradient reconstruction requires a unit direction")
    sigma, ell = geometry["sigma"], geometry["ell"]
    C = np.asarray(geometry.get("gradient_covariance", (sigma/ell)**2*np.eye(3)))
    cw = C @ direction
    variance = direction @ cw
    if variance <= 0 or not np.all(np.isfinite(C)):
        raise ValueError("invalid directional gradient covariance")
    regression = cw/variance
    helper = np.eye(3)[np.argmin(np.abs(direction))]
    e1 = np.cross(direction, helper)
    e1 /= np.linalg.norm(e1)
    basis = np.column_stack((e1, np.cross(direction, e1)))
    L = np.linalg.cholesky(basis.T @ (C-np.outer(cw, cw)/variance) @ basis)
    longitudinal = np.asarray(mean)+regression*(signed_derivative-direction @ mean)
    return signed_derivative*direction+basis @ (basis.T @ longitudinal+L @ rng.normal(size=2))
