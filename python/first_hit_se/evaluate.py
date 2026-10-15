"""Paired GRU/RNO likelihood, curve and exact-subdivision diagnostics."""
from __future__ import annotations

import csv
import html
from pathlib import Path
import time

import torch

from renewal.collect import digest, write_json
from renewal.profiles import subdivide_features
from renewal.model import mixture_cdf
from .data import load_data, profile_to_device, read_json
from .train import load_model, metrics


@torch.no_grad()
def compare(checkpoints, dataset, output, split="test", device="cpu", points=129, factors=(1, 2, 4, 8), limit=None):
    if points < 2 or not factors or any(n < 1 for n in factors) or (limit is not None and limit < 1):
        raise ValueError("invalid query grid or subdivision factors")
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = load_data(dataset)
    selected = [p for p in data["profiles"] if p["split"] == split]
    if not selected:
        raise ValueError("empty comparison split")
    models = [(load_model(path, device)[0], Path(path)) for path in checkpoints]
    names = [f"{model.config.architecture}_{path.parent.name.rsplit('_', 1)[-1]}" for model, path in models]
    if len(set(names)) != len(names):
        names = [f"{model.config.architecture}_{i}" for i, (model, _) in enumerate(models)]
    report = {"event": data["event"], "dataset_sha256": digest(dataset), "split": split,
              "reference_sampler": data["reference_sampler"],
              "split_unit": data["split_unit"], "models": {}, "profiles": [],
              "note": "Monte Carlo intervals exclude remaining GP discretization error. Curve and subdivision diagnostics may cover a requested subset; likelihood covers the entire split."}
    for name, (model, path) in zip(names, models):
        report["models"][name] = {"checkpoint": str(path), "checkpoint_sha256": digest(path),
            "model_config": model.configuration(), "likelihood": metrics(model, selected, device, batch_size=8)}
        training_report = path.parent/"run_report.json"
        if training_report.exists():
            training = read_json(training_report)
            report["models"][name]["training"] = {k: training[k] for k in
                ("selected_epoch", "completed_epochs", "optimizer_updates", "stop_reason", "training_seconds") if k in training}
    records, plots, speed_records, speed_plots = [], [], [], []
    for original in selected[:limit]:
        p = profile_to_device(original, device)
        q = torch.linspace(0, p["horizon"], points, dtype=torch.float64, device=device)
        observed = ((~p["hit"][:, None]) | (p["distance"][:, None] > q)).double().mean(0)
        n = len(p["hit"])
        z = 1.959963984540054
        center = (observed+z*z/(2*n))/(1+z*z/n)
        half = z*torch.sqrt(observed*(1-observed)/n+z*z/(4*n*n))/(1+z*z/n)
        entry = {"id": p["id"], "source": p["source"], "samples": n,
                 "initial_positive_fraction": float((p["initial_sign"] > 0).double().mean()),
                 "models": {}}
        curves = {"GP": observed.cpu().tolist()}
        predictions = {}
        for name, (model, _) in zip(names, models):
            if str(device).startswith("cuda"): torch.cuda.synchronize()
            began = time.perf_counter()
            base = model.query(p["features"], q, with_mixture=False)
            if str(device).startswith("cuda"): torch.cuda.synchronize()
            seconds = time.perf_counter()-began
            predictions[name] = base["T"].cpu().tolist()
            curves[name] = predictions[name]
            diagnostics = {}
            for factor in factors:
                if factor == 1:
                    changed = base
                else:
                    features = subdivide_features(p["features"], torch.full((len(p["features"]),), factor, dtype=torch.long, device=device))
                    changed = model.query(features, q, with_mixture=False)
                diagnostics[str(factor)] = {"max_abs_T": float((changed["T"]-base["T"]).abs().max()),
                    "max_abs_H": float((changed["log_T"]-base["log_T"]).abs().max())}
            refinement = None
            generator = torch.Generator().manual_seed(20261013)
            counts = torch.randint(1, 9, (len(p["features"]),), generator=generator).to(device)
            random_profile = subdivide_features(p["features"], counts)
            random_prediction = model.query(random_profile, q, with_mixture=False)
            diagnostics["random_1_to_8"] = {"max_abs_T": float((random_prediction["T"]-base["T"]).abs().max()),
                "max_abs_H": float((random_prediction["log_T"]-base["log_T"]).abs().max())}
            if model.config.architecture == "rno":
                finer = model.query(p["features"], q, with_mixture=False, maximum_step=model.config.maximum_step/2)
                refinement = {"half_step_max_abs_T": float((finer["T"]-base["T"]).abs().max()),
                              "half_step_max_abs_H": float((finer["log_T"]-base["log_T"]).abs().max())}
            entry["models"][name] = {"max_abs_T_reference": float((base["T"]-observed).abs().max()),
                "rmse_T_reference": float(((base["T"]-observed)**2).mean().sqrt()),
                "exact_subdivision": diagnostics, "solver_refinement": refinement,
                "curve_query_seconds": seconds, "curve_query_points": points, "device": str(device)}
        # Isolate the learned gradient head by holding event positions/signs at
        # their reference values. This is not the model's unconditional speed law.
        if p["hit"].any():
            speeds = p["speed"][p["hit"]]
            grid = torch.linspace(0, float(speeds.quantile(.995))*1.3, 97, dtype=torch.float64, device=device)
            empirical = (speeds[:, None] <= grid).double().mean(0)
            speed_curves = {"GP": empirical.cpu().tolist()}
            for name, (model, _) in zip(names, models):
                pred = model.query(p["features"], p["distance"][p["hit"]])
                branch = (p["initial_sign"][p["hit"]] < 0).long()
                rows = torch.arange(len(speeds), device=device)
                mix = tuple(pred[k][branch, rows] for k in ("log_weights", "means", "scales"))
                values = torch.stack([mixture_cdf(*mix, x).mean() for x in grid])
                speed_curves[name] = values.cpu().tolist()
                entry["models"][name]["reference_event_weighted_speed_cdf_max_error"] = float((values-empirical).abs().max())
            for i, x in enumerate(grid.cpu().tolist()):
                speed_records.append({"profile": p["id"], "W": x, **{name: values[i] for name, values in speed_curves.items()}})
            speed_plots.append((p["id"], float(grid[-1]), speed_curves))
        for i, x in enumerate(q.cpu().tolist()):
            records.append({"profile": p["id"], "q": x, "distance_from_entry": x*p["geometry"]["ell"],
                "GP": float(observed[i]), "GP_lower": float(center[i]-half[i]), "GP_upper": float(center[i]+half[i]),
                **{name: values[i] for name, values in predictions.items()}})
        report["profiles"].append(entry)
        plots.append((p["id"], p["horizon"], curves))
        print(f"compared {len(report['profiles'])}/{len(selected[:limit])}: {p['id']}", flush=True)
    with (output/"curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    if speed_records:
        with (output/"conditional_speed_cdf.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(speed_records[0]))
            writer.writeheader(); writer.writerows(speed_records)
    for name in names:
        report["models"][name]["mean_profile_max_abs_T"] = sum(p["models"][name]["max_abs_T_reference"] for p in report["profiles"])/len(report["profiles"])
        report["models"][name]["mean_profile_rmse_T"] = sum(p["models"][name]["rmse_T_reference"] for p in report["profiles"])/len(report["profiles"])
    write_json(output/"comparison_report.json", report)
    colors = ["#39464c", "#db9f66", "#8aaee2", "#bf382b", "#244fb3", "#7946ac"]
    content = ['<!doctype html><meta charset="utf-8"><title>SE first zero: RNO / GRU</title>',
        '<style>body{font:16px system-ui;max-width:1080px;margin:32px auto;padding:0 20px;color:#192331}section{margin:24px 0}svg{width:100%;height:auto;background:#f6f8fa}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}code{font-size:13px}</style>',
        '<h1>Unconditioned SE first zero — RNO / GRU</h1>',
        '<p>Distances start at medium entry. Both initial signs are retained. All models use the same cubic mean and held-out labels. Finite GP resolution and Monte Carlo error remain.</p>',
        '<table><tr><th>Model</th><th>Joint NLL</th><th>Distance NLL</th><th>Speed PIT KS</th><th>Mean maximum curve error</th></tr>']
    for name in names:
        info = report["models"][name]
        pit = info["likelihood"]["speed_pit_ks"]
        pit_text = "n/a" if pit is None else f"{pit:.5f}"
        content.append(f'<tr><td>{html.escape(name)}</td><td>{info["likelihood"]["joint_nll"]:.5f}</td><td>{info["likelihood"]["distance_nll"]:.5f}</td><td>{pit_text}</td><td>{info["mean_profile_max_abs_T"]:.5f}</td></tr>')
    content.append('</table><p><a href="comparison_report.json">Full metrics and subdivision diagnostics</a> · <a href="curves.csv">Curve data</a></p>')
    content.append('<details><summary>Checkpoints and training</summary><ul>')
    for name in names:
        info = report["models"][name]
        training = info.get("training", {})
        details = "; ".join(f"{k}={v}" for k, v in training.items())
        content.append(f'<li><b>{html.escape(name)}</b>: <code>{html.escape(info["checkpoint"])}</code><br>{html.escape(details)}</li>')
    content.append('</ul></details>')
    content.append('<h2>Partition sensitivity</h2><p>The polynomial is unchanged. H below denotes −log(T) after marginalizing the entry sign.</p><table><tr><th>Profile</th><th>Model</th><th>8× subdivision: max |ΔT|</th><th>8× subdivision: max |ΔH|</th><th>Random subdivision: max |ΔT|</th><th>Halved solver step: max |ΔT|</th></tr>')
    for p in report["profiles"]:
        for name, info in p["models"].items():
            eight = info["exact_subdivision"].get("8")
            a, b = (f'{eight["max_abs_T"]:.3g}', f'{eight["max_abs_H"]:.3g}') if eight else ("n/a", "n/a")
            half = info["solver_refinement"]
            half_text = f'{half["half_step_max_abs_T"]:.3g}' if half else "n/a"
            content.append(f'<tr><td>{html.escape(p["id"])}</td><td>{html.escape(name)}</td><td>{a}</td><td>{b}</td><td>{info["exact_subdivision"]["random_1_to_8"]["max_abs_T"]:.3g}</td><td>{half_text}</td></tr>')
    content.append('</table><h2>Transmittance from medium entry</h2>')
    def draw(identifier, xmax, curves, xlabel):
        content.append(f'<section><h3>{html.escape(identifier)}</h3><svg viewBox="0 0 1000 285" role="img" aria-label="{html.escape(xlabel)}"><path d="M50 20V220H980" fill="none" stroke="#999"/>')
        for j, (name, values) in enumerate(curves.items()):
            vertices = " ".join(f"{50+930*i/(len(values)-1):.2f},{220-190*y:.2f}" for i,y in enumerate(values))
            color = colors[j % len(colors)]
            content.append(f'<polyline points="{vertices}" fill="none" stroke="{color}" stroke-width="2"/><text x="{70+j*170}" y="250" fill="{color}">{html.escape(name)}</text>')
        content.append(f'<text x="5" y="30">1</text><text x="5" y="220">0</text><text x="850" y="250">{xmax:.4g}</text><text x="50" y="275">{html.escape(xlabel)}</text></svg></section>')
    for identifier, xmax, curves in plots:
        draw(identifier, xmax, curves, "q = distance from entry / ell")
    content.append('<h2>Conditional crossing-speed calibration</h2><p>W = −initial_sign × ell/sigma × dot(direction, gradient). Model CDFs are averaged over reference hit positions and signs to assess the gradient head separately from the distance prediction. These are not unconditional model-generated speed CDFs. Transverse gradients use the analytic stationary-SE conditional Gaussian.</p>')
    for identifier, xmax, curves in speed_plots:
        draw(identifier, xmax, curves, "W (positive normalized crossing speed); vertical axis: CDF")
    (output/"comparison.html").write_text("\n".join(content), encoding="utf-8")
    return {"report": str(output/"comparison_report.json"), "html": str(output/"comparison.html"), "models": report["models"]}
