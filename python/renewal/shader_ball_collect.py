"""Collect the fixed shader-ball scene with stratified 3-sigma surface starts.

Run from the repository root: python -m python.renewal.shader_ball_collect
    --config python/configs/renewal_shader_ball_sigma002_r05_collect.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from . import FORMAT_VERSION, SE_KERNEL
from .collect import ROOT, collision_seeds, digest, resolve, write_json

MODES = ("positive_exterior", "collision_state")
SPLITS = ("train", "validation", "test")


def stratified_residuals(mean_slopes, limit, seed):
    """Uniform coverage of each feasible residual interval, not a GP marginal draw."""
    slopes = np.asarray(mean_slopes, dtype=float)
    if not math.isfinite(limit) or limit <= 0 or not np.isfinite(slopes).all():
        raise ValueError("finite slopes and a positive residual limit are required")
    lower = np.maximum(-limit, -slopes)
    if (lower >= limit).any():
        raise ValueError("a surface ray has no outward derivative within the residual limit")
    rng = np.random.default_rng(seed)
    quantiles = (rng.permutation(len(slopes)) + rng.uniform(1e-6, 1-1e-6, len(slopes))) / len(slopes)
    residual = lower + (limit-lower) * quantiles
    if not ((np.abs(residual) <= limit).all() and (residual+slopes > 0).all()):
        raise ValueError("invalid surface derivative after reconstruction")
    return residual, quantiles


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def status(output, stage, **fields):
    path = output / "generation_status.json"
    previous = read(path) if path.exists() else {}
    previous.update(stage=stage, updated_at_utc=datetime.now(timezone.utc).isoformat(),
                    controller_pid=os.getpid(), **fields)
    temporary = path.with_suffix(".tmp")
    write_json(temporary, previous)
    # Windows readers may briefly deny deletion/rename of the old status file.
    # Progress reporting must not abort an otherwise valid sampling job.
    for attempt in range(40):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


def make_plan(config_path):
    settings = read(config_path)
    if settings.get("schema_version") != FORMAT_VERSION:
        raise ValueError("unsupported collection schema")
    output = resolve(settings["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    saved = output / "collection_config.json"
    if saved.exists() and read(saved) != settings:
        raise ValueError("collection settings changed; use a new output directory")
    write_json(saved, settings)
    geometry_path = resolve(settings["geometry"])
    geometry = read(geometry_path)
    scene = read(resolve(settings["render_config"]))
    sigma, ell = geometry["sigma"], geometry["ell"]
    if not math.isclose(sigma, settings["sigma"], rel_tol=1e-6) or geometry["roughness"] != settings["roughness"]:
        raise ValueError("geometry sigma/roughness differs from the collection settings")
    limit = settings["surface_residual_limit"]
    conditions = {}
    for split_index, split in enumerate(SPLITS):
        selected = [p for p in geometry["profiles"] if p["mode"] == "collision_state" and p["split"] == split]
        residuals, quantiles = stratified_residuals(
            [p["entry_mean_derivative"] for p in selected], limit,
            np.random.SeedSequence([settings["seed"], split_index, 73]))
        for p, z, u in zip(selected, residuals, quantiles):
            beta = p["entry_mean_derivative"] + float(z)
            conditions[p["id"]] = {
                "standardized_derivative_residual": float(z), "residual_divided_by_limit": float(z/limit),
                "beta_g": beta, "physical_outward_derivative": sigma/ell*beta,
                "mean_derivative_dimensionless": p["entry_mean_derivative"], "interval_quantile": float(u)}
    sampling = {"residual_limit": limit, "variable": "z0 = beta_g - b_prime(0)",
                "proposal": "stratified_uniform_over_each_feasible_interval",
                "constraints": "abs(z0) <= residual_limit and beta_g > 0",
                "scope": "surface start conditions only; conditional trajectories and hit labels remain untruncated",
                "network_input": "existing standardized residual and asinh encoding; no model change"}
    executable = resolve(settings["executable"])
    generator_hash = digest(executable)
    common = {"schema_version": FORMAT_VERSION, "kernel": SE_KERNEL, "split_unit": "ray",
              "split_description": "One shader-ball scene, disjoint cameras and physical rays across splits",
              "seed_scheme": "first_passage_collision_v1", "surface_derivative_sampling": sampling,
              "collection_config_sha256": digest(saved), "geometry_sha256": digest(geometry_path),
              "generator_sha256": generator_hash,
              "reference_description": "Full-ray SE circulant embedding; central-difference start derivative; cellwise mean and Hermite residual"}
    jobs, sources, identities, seeds = [], [], set(), set()
    for p in geometry["profiles"]:
        identity = tuple(round(v, 12) for v in p["origin"] + p["direction"])
        if identity in identities:
            raise ValueError("duplicate physical ray")
        identities.add(identity)
        if p["split"] not in SPLITS or p["mode"] not in MODES:
            raise ValueError("unknown geometry split/mode")
        n = int(settings["realizations_per_profile"][p["split"]])
        run = output / "raw" / p["mode"] / p["split"] / p["id"]
        ray = {"id": p["id"], "origin": p["origin"], "direction": p["direction"]}
        if p["mode"] == "collision_state":
            d = conditions[p["id"]]["physical_outward_derivative"]
            # For an isotropic kernel, only this projection affects the 1D ray.
            ray["gradient"] = [d*v for v in p["direction"]]
        seed = int.from_bytes(hashlib.sha256(f"{settings['seed']}:{p['id']}".encode()).digest()[:8], "little")
        if seed in seeds:
            raise ValueError("duplicate source seed")
        seeds.add(seed)
        coarse = p["horizon"] / math.ceil(p["horizon"] / settings["coarse_maximum_step"])
        config = {"schema_version": 1, "seed": seed, "output_directory": str(run), "first_passage": {
            "process": {"mean_field": scene["field"]}, "initial_condition": {"type": p["mode"], "rays": [ray]},
            "grid": {"max_time": p["horizon"], "step_sizes": [coarse, coarse/2]},
            "profile": {"maximum_step": settings["profile_maximum_step"]}, "curve": {"bins": settings["curve_bins"]},
            "monte_carlo": {"trajectories": n, "thread_count": 1, "write_raw_samples": True},
            "sampler": {"type": "collision_state_auto", "minimum_step": 1/1024,
                        "crossing_tolerance": 1e-9, "bridge_sigma_margin": 6, "max_refinement_depth": 16},
            "kernels": [{"id": "se", "type": "squared_exponential", "parameterization": "unit_length",
                         "variance": sigma*sigma, "length_scale": ell}]}}
        input_path = run / "input.json"
        write_json(input_path, config)
        source = {"scene_id": "shader_ball_full_0.02", "split": p["split"], "mode": p["mode"],
                  "directory": str(run), "geometry": scene["field"], "profile_id": p["id"],
                  "camera_id": p["camera_id"], "sigma": sigma, "ell": ell, "seed": seed,
                  "trajectories": n, "input_sha256": digest(input_path)}
        if p["id"] in conditions:
            source["start_condition"] = conditions[p["id"]]
        sources.append(source)
        jobs.append({"id": p["id"], "input": str(input_path), "stamp": str(run / "complete.json"),
                     "fingerprint": {"input": source["input_sha256"], "executable": generator_hash},
                     "finest_samples": n, "estimated_cost": n*p["horizon"]})
    for mode in MODES:
        write_json(output / mode / "manifest.json", {**common, "sources": [s for s in sources if s["mode"] == mode]})
    write_json(output / "manifest.json", {**common, "sources": sources})
    write_json(output / "start_conditions.json", {"sampling": sampling, "conditions": conditions})
    # A longest-first schedule reduces the serial tail of the parallel collector.
    jobs.sort(key=lambda j: j["estimated_cost"], reverse=True)
    write_json(output / "jobs.json", {"jobs": jobs})
    # Audit every planned random stream, including the independent coarse runs.
    seed_count = 2*sum(s["trajectories"] for s in sources)
    streams = np.empty(seed_count, dtype=np.uint64)
    position = 0
    for source in sources:
        for resolution in range(2):
            n = source["trajectories"]
            streams[position:position+n] = collision_seeds(source["seed"], 0, resolution, n)
            position += n
    streams.sort()
    duplicates = int(np.count_nonzero(streams[1:] == streams[:-1]))
    if duplicates:
        raise ValueError("duplicate planned trajectory seeds")
    write_json(output / "seed_audit.json", {"streams": seed_count, "duplicates": duplicates,
        "note": "Both grid resolutions and all splits/modes; raw CSV seeds are checked again during packing."})
    status(output, "planned", total_jobs=len(jobs), total_finest_samples=seed_count//2,
           completed_jobs=0, completed_finest_samples=0, surface_derivative_sampling=sampling)
    return settings, output, jobs


def sample(settings, output, jobs):
    status(output, "sampling", error=None)
    by_id = {j["id"]: j for j in jobs}
    completed_samples = 0
    start = time.monotonic()
    command = [str(resolve(settings["executable"])), "sample", str(output / "jobs.json"), str(settings["workers"])]
    with (output / "generation.log").open("a", encoding="utf-8", buffering=1) as log:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace") as process:
            try:
                status(output, "sampling", sampler_pid=process.pid)
                for line in process.stdout:
                    log.write(line)
                    match = re.match(r"completed (\d+)/(\d+) (\S+)", line)
                    if match:
                        count = int(match[1])
                        completed_samples += by_id[match[3]]["finest_samples"]
                        status(output, "sampling", completed_jobs=count, completed_finest_samples=completed_samples,
                               sampling_elapsed_seconds=round(time.monotonic()-start, 2))
                        if count == 1 or count % 50 == 0 or count == len(jobs):
                            print(line.rstrip(), flush=True)
                if process.wait():
                    raise RuntimeError("reference sampling failed; see generation.log")
            except BaseException:
                # Never wait for a child while abandoning its stdout pipe:
                # the child could block on a full pipe during context cleanup.
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
                raise


def audit_reference(output):
    manifest = read(output / "manifest.json")
    groups = defaultdict(list)
    per_profile = []
    for source in manifest["sources"]:
        directory = Path(source["directory"])
        with (directory / "first_passage_summary.csv").open(newline="") as stream:
            summaries = {int(r["training_resolution"]): r for r in csv.DictReader(stream)}
        if set(summaries) != {0, 1}:
            raise ValueError("missing reference resolution")
        for row in summaries.values():
            n = int(row["trajectories"])
            if n != source["trajectories"] or int(row["events"])+int(row["censored"]) != n:
                raise ValueError("incomplete reference summary")
        with (directory / "first_passage_curves.csv").open(newline="") as stream:
            curves = {0: [], 1: []}
            for row in csv.DictReader(stream):
                curves[int(row["training_resolution"])].append(float(row["survival"]))
        if len(curves[0]) != len(curves[1]):
            raise ValueError("reference curve grids differ")
        coarse, fine = summaries[0], summaries[1]
        result = {"profile_id": source["profile_id"], "split": source["split"], "mode": source["mode"],
                  "samples": source["trajectories"], "hits": int(fine["events"]), "censored": int(fine["censored"]),
                  "coarse_minus_fine_hit_probability": float(coarse["event_probability"])-float(fine["event_probability"]),
                  "maximum_survival_difference": float(np.max(np.abs(np.array(curves[0])-curves[1])))}
        per_profile.append(result)
        groups[(source["mode"], source["split"])].append(result)
    counts = {}
    for (mode, split), records in groups.items():
        delta = np.array([r["coarse_minus_fine_hit_probability"] for r in records])
        counts.setdefault(mode, {})[split] = {"profiles": len(records),
            "samples": sum(r["samples"] for r in records), "hits": sum(r["hits"] for r in records),
            "censored": sum(r["censored"] for r in records),
            "mean_coarse_minus_fine_hit_probability": float(delta.mean()),
            "profile_standard_error": float(delta.std(ddof=1)/math.sqrt(len(delta))) if len(delta)>1 else None,
            "mean_maximum_survival_difference": float(np.mean([r["maximum_survival_difference"] for r in records])),
            "worst_maximum_survival_difference": max(r["maximum_survival_difference"] for r in records)}
    report = {"counts": counts, "per_profile": per_profile,
              "note": "Independent Monte Carlo realizations per resolution; differences include sampling noise and are not certified discretization bounds."}
    write_json(output / "reference_audit.json", report)
    return counts


def finalize(output):
    from .data import prepare
    import torch
    torch.set_num_threads(2)
    status(output, "auditing")
    counts = audit_reference(output)
    for mode in (*MODES, "combined"):
        status(output, "packing", packing_dataset=mode)
        manifest = output / "manifest.json" if mode == "combined" else output / mode / "manifest.json"
        target = output / "data" / (mode + ".pt")
        report = prepare(manifest, target)
        print(f"packed {mode}: {sum(r['samples'] for r in report['counts'].values())} samples", flush=True)
        gc.collect()
    summary = {"status": "complete", "counts": counts,
               "datasets": {mode: str(output / "data" / (mode + ".pt")) for mode in (*MODES, "combined")},
               "surface_derivative_sampling": read(output / "manifest.json")["surface_derivative_sampling"],
               "seed_audit": read(output / "seed_audit.json"),
               "scope": "Reference dataset generation only; no network training or renderer changes."}
    write_json(output / "dataset_summary.json", summary)
    status(output, "complete", completed_at_utc=datetime.now(timezone.utc).isoformat(), packing_dataset=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    output = resolve(read(args.config)["output_directory"])
    try:
        if args.finalize_only:
            finalize(output)
            return
        settings, output, jobs = make_plan(args.config)
        print(f"planned {len(jobs)} profiles, {sum(j['finest_samples'] for j in jobs)} finest-resolution samples", flush=True)
        if not args.plan_only:
            sample(settings, output, jobs)
            finalize(output)
    except Exception as error:
        status(output, "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
