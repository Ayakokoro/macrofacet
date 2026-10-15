"""Compare exported C++ inference and samples to the same Python model and cubic."""
from pathlib import Path
import subprocess

import torch

from renewal.collect import write_json
from renewal.reference_sampler import require_same_sampler
from .data import load_data, read_json
from .train import load_model, export_model


@torch.no_grad()
def verify_cpp(checkpoint, dataset, executable, output, limit=3, trials=512):
    torch.set_num_threads(4)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    model, _ = load_model(checkpoint)
    export_model(checkpoint, output/"model.json")
    data = load_data(dataset)
    require_same_sampler(model.reference_sampler, data["reference_sampler"], "C++ verification")
    selected = [p for p in data["profiles"] if p["split"] == "test"][:limit]
    if not selected:
        raise ValueError("no test profiles to verify")
    records = []
    for i,p in enumerate(selected):
        q = torch.linspace(0,p["horizon"],65,dtype=torch.float64)
        python = model.query(p["features"],q)
        request, result_path = output/f"profile_{i}.json", output/f"result_{i}.json"
        write_json(request, {"features":p["features"].tolist(),"queries":q.tolist()})
        subprocess.run([str(executable),str(output/"model.json"),str(request),str(result_path),str(trials)],check=True)
        result = read_json(result_path)
        errors = {}
        for key in ("H","log_hazard","hit_branch_probability"):
            cpp = torch.tensor([row[key] for row in result["predictions"]],dtype=torch.float64).T
            errors[key] = float((cpp-python[key]).abs().max())
        errors["log_T"] = float((torch.tensor([row["log_T"] for row in result["predictions"]],dtype=torch.float64)-python["log_T"]).abs().max())
        errors["T"] = float((torch.tensor([row["log_T"] for row in result["predictions"]],dtype=torch.float64).exp()-python["T"]).abs().max())
        for name in ("weights","means","scales"):
            cpp = torch.tensor([[m[name] for m in row["mixtures"]] for row in result["predictions"]],dtype=torch.float64).transpose(0,1)
            expected = python["log_weights"].exp() if name=="weights" else python[name]
            errors[name] = float((cpp-expected).abs().max())
        tolerances = {key: 2e-4 for key in errors}
        if model.config.architecture == "spline":
            # Moving narrow learned knots by float32 GEMM roundoff can change
            # pointwise log density more than cumulative probability. Require
            # <=0.1% relative hazard error AND a tighter absolute CDF check;
            # retain the original parameter/other-output limits.
            tolerances.update(log_hazard=1e-3, T=1e-5)
        if any(error > tolerances[key] for key, error in errors.items()):
            raise AssertionError(f"C++/Python mismatch: {errors}")
        sample_error = None
        if trials:
            observed = torch.tensor([sum((not s["hit"]) or s["q"]>x for s in result["samples"])/trials for x in q.tolist()])
            sample_error = float((observed-python["T"]).abs().max())
            if sample_error > 2/(trials**.5)+.002:
                raise AssertionError(f"sampled CDF differs from predicted CDF: {sample_error}")
        records.append({"profile":p["id"],"maximum_errors":errors,"absolute_tolerances":tolerances,
                        "sample_curve_max_error":sample_error,"trials":trials})
    report={"architecture":model.config.architecture,"profiles":records}
    write_json(output/"verification.json",report)
    return report
