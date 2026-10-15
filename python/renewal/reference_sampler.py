"""The fixed reference process is part of a dataset/model's identity, not a network input."""
import math


FUNCTION_SPACE = "gpis_function_space"
_CONVENTIONS = {"version": 1, "history_derivatives": "latest", "grid_rule": "fixed_from_entry",
                "derivative_observations": "analytic"}
_SLOPE_METHODS = {"tungsten_linear": "total_field_linear_secant",
                  "hermite_residual_cellwise_mean": "analytic_derivative_hermite_cellwise_mean"}
_DEFAULTS = {"type": FUNCTION_SPACE, "history_points": 1, "batch_size": 8,
             "step_size": 1/64, "rank_tolerance": 1e-10, "crossing_rule": "tungsten_linear", **_CONVENTIONS}


def reference_sampler(spec=None):
    """None represents pre-existing full-history/state-space references for backward compatibility."""
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError("reference_sampler must be an object")
    if not spec:
        return None
    if spec.get("type") in ("collision_state_auto", "exact_grid_circulant"):
        return None
    if spec.get("type") != FUNCTION_SPACE or set(spec)-set(_DEFAULTS):
        raise ValueError("unknown reference_sampler type or setting")
    result = {**_DEFAULTS, **spec}
    for key, value in _CONVENTIONS.items():
        if type(result[key]) is not type(value) or result[key] != value:
            raise ValueError(f"unsupported GPIS convention: {key}")
    if not isinstance(result["crossing_rule"], str) or result["crossing_rule"] not in _SLOPE_METHODS:
        raise ValueError("unsupported GPIS crossing_rule")
    for key in ("history_points", "batch_size"):
        if type(result[key]) is not int or not 1 <= result[key] <= 128:
            raise ValueError(f"GPIS {key} must be an integer in [1,128]")
    for key in ("step_size", "rank_tolerance"):
        if type(result[key]) not in (int, float) or not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"GPIS {key} must be finite and positive")
    if not 1e-14 <= result["rank_tolerance"] <= 1e-4:
        raise ValueError("GPIS rank_tolerance must be in [1e-14,1e-4]")
    return result


def require_same_sampler(expected, actual, context="reference data"):
    expected, actual = reference_sampler(expected), reference_sampler(actual)
    if expected != actual:
        raise ValueError(f"{context}: reference_sampler mismatch; use a separate dataset/model "
                         f"for each H, batch_size, step_size, crossing_rule and numerical convention ({expected} != {actual})")


def validate_sampler_row(sampler, row):
    """Do not relabel old Hermite CSVs by changing their source metadata to linear."""
    if sampler is not None and row.get("crossing_slope_method") != _SLOPE_METHODS[sampler["crossing_rule"]]:
        raise ValueError("GPIS crossing_slope_method does not match reference_sampler.crossing_rule")


def sampler_from_source(config):
    sampler = reference_sampler(config.get("sampler"))
    if sampler is not None:
        if any(k["type"] != "squared_exponential" for k in config["kernels"]):
            raise ValueError("GPIS function space requires an SE kernel")
        steps = config["grid"]["step_sizes"]
        if len(steps) != 1 or not math.isclose(steps[0], sampler["step_size"], rel_tol=1e-12):
            raise ValueError("GPIS reference must use exactly its fixed step_size")
    return sampler


def configure_sampler(config, settings):
    """Apply an optional collection override; never rescale a GPIS step to fit the horizon."""
    if "reference_sampler" in settings:
        sampler = reference_sampler(settings["reference_sampler"])
        if sampler is None:
            raise ValueError("omit reference_sampler to use the existing reference backend")
        config["sampler"] = sampler
    sampler = reference_sampler(config.get("sampler"))
    if sampler is not None:
        if "reference_steps" in settings and settings["reference_steps"] != [sampler["step_size"]]:
            raise ValueError("reference_steps conflicts with fixed GPIS step_size")
        config["grid"]["step_sizes"] = [sampler["step_size"]]
        sampler_from_source(config)
    return sampler
