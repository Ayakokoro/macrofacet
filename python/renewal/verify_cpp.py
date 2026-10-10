"""Compare exported C++ inference/inversion with PyTorch on identical ray segments."""
from __future__ import annotations

import bisect
import json
import math
from pathlib import Path
import subprocess

import torch
from torch.nn import functional as F

from .collect import ROOT, write_json
from .evaluate import load_model
from .export import export_model
from .model import cumulative, mixture_cdf


@torch.no_grad()
def verify_cpp(checkpoint: Path, executable: Path, config: Path, output: Path,
               trials: int = 16384, limit_rays: int = 4):
    torch.set_num_threads(2)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model, bundle_metadata = load_model(checkpoint, "cpu")
    model.eval()
    bundle = output / "renewal_model.json"
    export_model(checkpoint, bundle)
    root = json.loads(config.read_text(encoding="utf-8"))
    settings = root["first_passage"]
    for kernel in settings["kernels"]:
        if (kernel["type"] != bundle_metadata["kernel"]["type"] or
                kernel.get("parameterization", bundle_metadata["kernel"]["parameterization"]) !=
                bundle_metadata["kernel"]["parameterization"]):
            raise ValueError("reference/model kernel mismatch")
    initial = settings["initial_condition"]
    for key in ("rays", "profiles"):
        if key in initial:
            initial[key] = initial[key][:limit_rays]
    maximum = settings["grid"]["max_time"]
    root["output_directory"] = str(output / "cpp")
    root["renewal_query"] = {
        "normalized_distances": [maximum*f for f in (0, 0.0001, 0.13, 0.25, 0.5, 0.61, 0.9999, 1)],
        "survival_intervals": [[0, maximum], [0.13*maximum, 0.61*maximum], [maximum, maximum]],
        "speed_sample_count": 4096,
        "optical_depths": [0.0, 1e-7, 0.01, 0.1, 0.3, 0.7, 1.5, 3.0, 10.0, 1000.0]}
    request = output / "request.json"
    write_json(request, root)
    completed = subprocess.run([str(executable.resolve()), "renewal-query", "--config", str(request),
                                "--model", str(bundle), "--trials", str(trials)],
                               cwd=ROOT, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)
    results = json.loads((output / "cpp" / "renewal_query.json").read_text(encoding="utf-8"))
    report = {"rays": len(results["rays"]), "max_rate_error": 0.0, "max_cumulative_error": 0.0,
              "max_transmittance_error": 0.0, "max_inverse_optical_depth_error": 0.0,
              "max_sampled_survival_error": 0.0, "max_mixture_parameter_error": 0.0,
              "max_sampled_speed_cdf_error": 0.0, "checkpoint_sha256": results["checkpoint_sha256"]}
    for ray in results["rays"]:
        features = torch.tensor([s["features"] for s in ray["segments"]], dtype=torch.double)
        known = torch.tensor([ray["known_initial"]], dtype=torch.double)
        context, logits = model.encode(features.unsqueeze(0), known, torch.tensor([len(features)]))
        logits = logits[0]
        rates = F.softplus(logits)
        cpp_rates = torch.tensor([s["rates"] for s in ray["segments"]], dtype=torch.double)
        report["max_rate_error"] = max(report["max_rate_error"], float((rates.double()-cpp_rates).abs().max()))
        torch.testing.assert_close(cpp_rates, rates.double(), rtol=5e-5, atol=3e-6)
        ends = [s["end"] for s in ray["segments"]]
        starts = [s["begin"] for s in ray["segments"]]
        dx = features[:, 4].exp()
        totals = dx * rates.sum(-1)/4
        prefix = F.pad(totals.cumsum(-1), (1, 0))

        def H(x):
            index = min(bisect.bisect_right(ends, x), len(ends)-1)
            u = min(1., max(0., (x-starts[index])/(ends[index]-starts[index])))
            return float(prefix[index] + cumulative(logits[index], dx[index], torch.tensor(u, dtype=torch.double)))

        def close(actual, expected, name):
            if not math.isclose(actual, expected, rel_tol=5e-5, abs_tol=4e-6):
                raise AssertionError(f"{name}: C++ {actual}, PyTorch {expected}")

        for query in ray["queries"]:
            expected_h = H(query["x"])
            expected_t = math.exp(-expected_h)
            report["max_cumulative_error"] = max(report["max_cumulative_error"], abs(query["cumulative_hazard"]-expected_h))
            report["max_transmittance_error"] = max(report["max_transmittance_error"], abs(query["transmittance"]-expected_t))
            close(query["cumulative_hazard"], expected_h, "cumulative hazard")
            close(query["transmittance"], expected_t, "transmittance")
            error = abs(query["sampled_survival"]-query["transmittance"])
            report["max_sampled_survival_error"] = max(report["max_sampled_survival_error"], error)
            if error > 6*math.sqrt(expected_t*(1-expected_t)/trials)+2/trials:
                raise AssertionError("sampled survival disagrees with model transmittance")
            index = min(bisect.bisect_right(ends, query["x"]), len(ends)-1)
            u = (query["x"]-starts[index])/(ends[index]-starts[index])
            mixture = model.mixture(context[0, index], features[index], torch.tensor(u, dtype=torch.double))
            for name, expected in zip(("weights", "means", "scales"), (mixture[0].exp(), mixture[1], mixture[2])):
                actual = torch.tensor(query["speed_mixture"][name], dtype=torch.double)
                report["max_mixture_parameter_error"] = max(report["max_mixture_parameter_error"], float((actual-expected.double()).abs().max()))
                torch.testing.assert_close(actual, expected.double(), rtol=5e-5, atol=4e-6)
            n = query["speed_sample_count"]
            # Float32 log_softmax can sum to 1+O(1e-7) after conversion to
            # double. Normalize the reference weights before binomial checks.
            log_weights = mixture[0].double()
            cdf_mixture = (log_weights-torch.logsumexp(log_weights, -1), mixture[1], mixture[2])
            for check in query["speed_cdf"]:
                probability = float(mixture_cdf(*cdf_mixture, torch.tensor(check["w"], dtype=torch.double)))
                probability = min(1., max(0., probability))
                error = abs(check["empirical"]-probability)
                report["max_sampled_speed_cdf_error"] = max(report["max_sampled_speed_cdf_error"], error)
                if error > 6*math.sqrt(probability*(1-probability)/n)+2/n:
                    raise AssertionError("sampled speed CDF disagrees with the truncated mixture")
        for interval in ray["intervals"]:
            start_h = H(interval["begin_x"])
            total = H(interval["end_x"])-start_h
            close(interval["transmittance"], math.exp(-total), "conditional transmittance")
            for inverse in interval["inversions"]:
                depth = inverse["optical_depth"]
                if inverse["hit"]:
                    residual = abs(H(inverse["x"])-start_h-depth)
                    report["max_inverse_optical_depth_error"] = max(report["max_inverse_optical_depth_error"], residual)
                    close(H(inverse["x"])-start_h, depth, "inverse optical depth")
                    close(inverse["distance"], inverse["x"]*ray["ell"], "physical distance")
                elif depth < total-4e-6:
                    raise AssertionError("C++ incorrectly censored an event inside the interval")
    report["passed"] = True
    write_json(output / "parity_report.json", report)
    return report
