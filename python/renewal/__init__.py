"""Renewal+ sequence models with explicit, validated covariance conventions."""

FORMAT_VERSION = 1
KERNEL = {"type": "matern_3_2", "parameterization": "unit_decay", "beta": 1.0}
SE_KERNEL = {"type": "squared_exponential", "parameterization": "unit_length", "beta": 1.0}


def validate_kernel(kernel):
    """SE uses rho(x)=exp(-x*x/2), x=distance/ell; beta=-rho''(0)=1."""
    if kernel not in (KERNEL, SE_KERNEL):
        raise ValueError("unsupported kernel convention; expected unit-decay Matern-3/2 or unit-length SE")
    return dict(kernel)
