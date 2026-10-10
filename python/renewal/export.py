"""Export the full trained model or its hazard path for the Eigen C++ runtime."""
from pathlib import Path

from .collect import digest, write_json
from .evaluate import load_model


def export_model(checkpoint: Path, output: Path, *, hazard_only: bool = False):
    model, bundle = load_model(checkpoint, "cpu")
    weights = {}
    for name, tensor in model.state_dict().items():
        if hazard_only and name.startswith("mixture_head."):
            continue
        if not tensor.is_floating_point() or not tensor.isfinite().all():
            raise ValueError(f"invalid model tensor: {name}")
        value = tensor.detach().cpu().float().contiguous()
        weights[name] = {"shape": list(value.shape), "values": value.reshape(-1).tolist()}
    result = {"format": "macrofacet.renewal_hazard" if hazard_only else "macrofacet.renewal", "version": 1, "kernel": bundle["kernel"],
              "activation_dtype": "float32", "gru_convention": "pytorch_rzn_reset_after",
              "feature_transform": "asinh_first_four_log_dx_identity",
              "initial_transform": "asinh_mode_b0_known_z0_known_d0",
              "checkpoint_sha256": digest(checkpoint), "selected_epoch": bundle.get("epoch"),
              "model_config": model.configuration(), "weights": weights}
    if not hazard_only:
        result["mixture_convention"] = "positive_truncated_components_residual_mean"
    write_json(output, result)
    return {"bundle": str(output), "checkpoint_sha256": result["checkpoint_sha256"],
            "tensors": len(weights), "parameters": sum(len(w["values"]) for w in weights.values())}


def export_hazard(checkpoint: Path, output: Path):
    return export_model(checkpoint, output, hazard_only=True)
