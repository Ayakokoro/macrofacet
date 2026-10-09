import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from renewal import KERNEL
from renewal.model import ModelConfig, RenewalNetwork
from renewal.verify_cpp import verify_cpp


@unittest.skipUnless(os.environ.get("MACROFACET_RENEWAL_EXE"), "C++ executable is supplied by CTest")
class CppParityTests(unittest.TestCase):
    def test_random_weight_gru_and_cumulative_in_both_start_modes(self):
        torch.manual_seed(9847)
        model = RenewalNetwork(ModelConfig(embedding=7, hidden=9, hazard_width=6, mixture_width=8, components=2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            checkpoint = path / "network.pt"
            torch.save({"format_version": 1, "kernel": KERNEL, "epoch": 0,
                        "model_config": model.configuration(), "model_state": model.state_dict()}, checkpoint)
            for mode in ("positive_exterior", "collision_state"):
                ray = {"id": "curve", "origin": [0.13, 0.2, 0.83], "direction": [1, 0.2, 0.1]}
                if mode == "collision_state":
                    ray["gradient"] = [0.3, 0.2, 0.7]
                config = {"schema_version": 1, "seed": 891, "first_passage": {
                    "process": {"mean_field": {"mean_type": "sphere", "sphere_center": [0, 0, 0], "sphere_radius": 0.8}},
                    "initial_condition": {"type": mode, "rays": [ray]},
                    "grid": {"max_time": 4, "step_sizes": [0.015625]},
                    "profile": {"maximum_step": 0.125},
                    "kernels": [{"id": "m32", **KERNEL, "variance": 0.01, "length_scale": 0.23}]}}
                source = path / f"{mode}.json"
                source.write_text(json.dumps(config), encoding="utf-8")
                result = verify_cpp(checkpoint, Path(os.environ["MACROFACET_RENEWAL_EXE"]),
                                    source, path/mode, trials=4096)
                self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
