"""Collect and validate unconditioned first-zero references; never reinterpret A/B data."""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

import torch

from renewal.collect import digest, write_json, resolve
from renewal.data import FEATURES
from renewal.reference_sampler import configure_sampler, sampler_from_source, reference_sampler, require_same_sampler
from renewal.reference_sampler import validate_sampler_row
from . import EVENT, FORMAT, VERSION, SE_KERNEL
from .model import validate_profile


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def collect(config_path):
    settings = read_json(config_path)
    output = resolve(settings["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    previous_manifest = read_json(output/"manifest.json") if (output/"manifest.json").exists() else None
    executable = resolve(settings.get("executable", "build/Release/macrofacet_experiments.exe"))
    inspector = resolve(settings.get("inspector", "build/Release/macrofacet_inspect_first_hit_se.exe"))
    manifest = {"format": FORMAT, "version": VERSION, "event": EVENT, "kernel": SE_KERNEL,
                "split_unit": settings.get("split_unit", "scene"), "sources": [],
                "collection_config_sha256": digest(config_path), "executable_sha256": digest(executable),
                "skipped_domain_misses": []}
    expanded = []
    for source in settings["sources"]:
        pixels = source.get("pixels", [source.get("pixel")])
        expanded.extend((source, pixel) for pixel in pixels)
    workers = settings.get("collection_workers", 1)
    if type(workers) is not int or workers < 1:
        raise ValueError("collection_workers must be a positive integer")

    def collect_source(item):
        index, (source, pixel) = item
        folder = output/f"source_{index:05d}"
        folder.mkdir(exist_ok=True)
        if "render_config" in source:
            if pixel is None or len(pixel) != 2:
                raise ValueError("camera sources require a pixel")
            request = folder/"reference.json"
            inspected = subprocess.run([str(inspector), str(resolve(source["render_config"])), str(folder),
                            str(pixel[0]), str(pixel[1])], capture_output=True, text=True)
            if inspected.returncode:
                if settings.get("skip_domain_misses", False) and inspected.stderr.strip() == "ray misses medium":
                    print(f"skipped ray outside activeDomain: {pixel}", flush=True)
                    return {"skipped": {"scene_id": source["scene_id"], "split": source["split"], "pixel": pixel}}
                raise RuntimeError(f"camera extraction failed for {pixel}: {inspected.stderr}")
            config = read_json(request)
        else:
            config = read_json(resolve(source["reference_config"]))
        fp = config["first_passage"]
        if fp["initial_condition"]["type"] != EVENT:
            raise ValueError("collection accepts only unconditioned_first_zero references")
        if any(k["type"] != SE_KERNEL["type"] or k.get("parameterization") != "unit_length" for k in fp["kernels"]):
            raise ValueError("collection requires unit-length SE")
        horizon = fp["grid"]["max_time"]
        sampler = configure_sampler(fp, settings)
        if previous_manifest is not None:
            require_same_sampler(previous_manifest.get("reference_sampler"), sampler,
                                 "existing collection; choose a new output_directory")
        if sampler is None:
            fp["grid"]["step_sizes"] = [horizon/math.ceil(horizon/step) for step in settings.get("reference_steps", [1/64, 1/128])]
        fp["monte_carlo"] = {"trajectories": source.get("realizations_per_profile", settings.get("realizations_per_profile", 128)),
            "thread_count": settings.get("threads", 4), "write_raw_samples": True}
        fp["rice_series"] = {"enabled": False}
        fp["state_analysis"] = {"enabled": False}
        config["seed"] = settings.get("seed", 20261013)+index*104729
        config["output_directory"] = str(folder/"reference")
        request = folder/"collection_reference.json"
        cache = folder/"completed.json"
        fingerprint = {"request": config, "executable": manifest["executable_sha256"]}
        if not cache.exists() or read_json(cache) != fingerprint:
            write_json(request, config)
            subprocess.run([str(executable), "first-passage", "--config", str(request)], check=True,
                           stdout=subprocess.DEVNULL)
            write_json(cache, fingerprint)
        record = {"scene_id": source["scene_id"], "split": source["split"],
            "directory": str((folder/"reference").resolve()), "camera_metadata": str((folder/"ray.json").resolve())
            if "render_config" in source else None}
        print(f"collected {index+1}/{len(expanded)}: {source['scene_id']} {pixel}", flush=True)
        return {"source": record, "sampler": sampler}

    # Each worker owns one directory. Input order fixes seeds and manifest order,
    # independently of scheduling; stochastic reference configuration is unchanged.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(collect_source, enumerate(expanded)):
            if "skipped" in result:
                manifest["skipped_domain_misses"].append(result["skipped"])
                continue
            if "reference_sampler" in manifest:
                require_same_sampler(manifest["reference_sampler"], result["sampler"], "collection sources")
            manifest["reference_sampler"] = result["sampler"]
            manifest["sources"].append(result["source"])
    write_json(output/"manifest.json", manifest)
    return prepare(output/"manifest.json", output/"dataset.pt")


def prepare(manifest_path, output):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = read_json(manifest_path)
    if (manifest.get("format"), manifest.get("version"), manifest.get("event"), manifest.get("kernel")) != (FORMAT, VERSION, EVENT, SE_KERNEL):
        raise ValueError("incompatible first-hit manifest")
    split_unit = manifest.get("split_unit", "scene")
    if split_unit not in ("scene", "ray"):
        raise ValueError("split_unit must be scene or ray")
    profiles, scenes, signatures, directories = [], {}, {}, set()
    sampler_identity = []
    for source in manifest["sources"]:
        directory = Path(source["directory"])
        if not directory.is_absolute():
            directory = (manifest_path.parent/directory).resolve()
        if str(directory) in directories:
            raise ValueError("duplicate reference source")
        directories.add(str(directory))
        split, scene = source["split"], source["scene_id"]
        if split not in ("train", "validation", "test"):
            raise ValueError("unknown data split")
        if split_unit == "scene" and scenes.setdefault(scene, split) != split:
            raise ValueError("scene leakage across splits")
        config_path = directory/"resolved_first_passage_config.json"
        config = read_json(config_path)["first_passage"]
        sampler = sampler_from_source(config)
        if "reference_sampler" in manifest:
            require_same_sampler(manifest["reference_sampler"], sampler, "manifest and source")
        if sampler_identity:
            require_same_sampler(sampler_identity[0], sampler, "dataset sources")
        else:
            sampler_identity.append(sampler)
        if config["initial_condition"]["type"] != EVENT:
            raise ValueError("conditioned A/B labels cannot be relabelled as first-zero data")
        kernels = {k["id"]: k for k in config["kernels"]}
        if any(k["type"] != SE_KERNEL["type"] or k.get("parameterization") != "unit_length" for k in kernels.values()):
            raise ValueError("reference has incompatible kernel")
        groups, samples = defaultdict(list), defaultdict(list)
        for row in rows(directory/"first_passage_mean_segments.csv"):
            if row["start_mode"] != EVENT:
                raise ValueError("mean profile has a different event convention")
            groups[row["kernel_id"], row["state_id"]].append(row)
        for row in rows(directory/"first_passage_samples.csv"):
            if row["training_resolution"] == "1":
                samples[row["kernel_id"], row["state_id"]].append(row)
        geometry = {(g["kernel_id"], g["state_id"]): g for g in read_json(directory/"first_hit_geometry.json")}
        for key, segments in groups.items():
            segments.sort(key=lambda row: int(row["segment"]))
            features = torch.tensor([[float(row[name]) for name in FEATURES] for row in segments], dtype=torch.float64)
            validate_profile(features)
            previous = 0.
            for i, row in enumerate(segments):
                lo, hi = float(row["x_begin"]), float(row["x_end"])
                if int(row["segment"]) != i or lo != previous or hi <= lo or not math.isclose(hi-lo, math.exp(float(row["log_dx"])), rel_tol=1e-9):
                    raise ValueError("invalid geometry partition")
                previous = hi
            horizon = config["grid"]["max_time"]
            if not math.isclose(previous, horizon, rel_tol=1e-10):
                raise ValueError("censoring horizon and profile disagree")
            labels = samples.pop(key, [])
            if len(labels) != config["monte_carlo"]["trajectories"]:
                raise ValueError("incomplete finest-resolution labels")
            g = geometry[key]
            if source.get("camera_metadata"):
                camera = read_json(source["camera_metadata"])
                g["gradient_covariance"] = camera["gradient_covariance"]
            polynomial = torch.tensor(g["mean_gradient_polynomials"], dtype=torch.float64)
            if polynomial.shape != (len(features), 3, 3) or not torch.isfinite(polynomial).all():
                raise ValueError("invalid full-gradient geometry")
            physical_geometry = {k: v for k, v in g.items() if k not in ("kernel_id", "state_id")}
            signature = hashlib.sha256(json.dumps({"features": features.tolist(), "geometry": physical_geometry}, sort_keys=True).encode()).hexdigest()
            if signature in signatures:
                raise ValueError("duplicate physical profile or leakage across splits")
            signatures[signature] = split
            distance, hit, sign, speed, seen = [], [], [], [], set()
            for row in labels:
                q, k = float(row["event_q"]), int(row["initial_sign"])
                event = row["event"] == "1"
                w = float(row["crossing_slope"]) if event else math.nan
                if row["trajectory"] in seen or row["event"] not in ("0", "1") or int(row["censored"]) != int(not event):
                    raise ValueError("duplicate trajectory or invalid event flag")
                seen.add(row["trajectory"])
                expected_sampler = sampler["type"] if sampler else "unconditioned_grid_circulant"
                validate_sampler_row(sampler, row)
                if (row["sampler_type"] != expected_sampler or k not in (-1, 1) or
                    not (0 < q <= horizon) or not math.isclose(float(row["base_step"]), min(config["grid"]["step_sizes"]), rel_tol=1e-10)):
                    raise ValueError("invalid first-zero label or reference resolution")
                if not math.isclose(float(row["sigma"]), g["sigma"], rel_tol=1e-10) or not math.isclose(float(row["ell"]), g["ell"], rel_tol=1e-10):
                    raise ValueError("inconsistent physical units")
                if not math.isclose(float(row["event_distance"]), q*g["ell"], rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError("invalid physical distance")
                if event and (not math.isfinite(w) or w <= 0 or not math.isclose(float(row["crossing_derivative"]), -k*w*g["sigma"]/g["ell"], rel_tol=1e-9, abs_tol=1e-12)):
                    raise ValueError("signed derivative disagrees with entry sign or crossing speed")
                if not event and (q != horizon or row["crossing_slope"] or row["crossing_derivative"]):
                    raise ValueError("invalid censored label")
                distance.append(q); hit.append(event); sign.append(k); speed.append(w)
            profiles.append({"id": f"{scene}:{key[0]}:{key[1]}:{len(profiles)}", "scene_id": scene, "split": split,
                "features": features, "horizon": horizon, "geometry": g, "reference_sampler": sampler,
                "distance": torch.tensor(distance, dtype=torch.float64), "hit": torch.tensor(hit),
                "initial_sign": torch.tensor(sign), "speed": torch.tensor(speed, dtype=torch.float64),
                "source": str(directory), "resolved_sha256": digest(config_path),
                "samples_sha256": digest(directory/"first_passage_samples.csv")})
        if samples:
            raise ValueError("labels reference an unknown mean profile")
    if not profiles:
        raise ValueError("empty first-hit dataset")
    data = {"format": FORMAT, "version": VERSION, "event": EVENT, "kernel": SE_KERNEL,
            "reference_sampler": sampler_identity[0],
            "profile_mode": "cubic", "split_unit": split_unit, "manifest_sha256": digest(manifest_path), "profiles": profiles}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output)
    report = {"dataset": str(output), "sha256": digest(output), "split_unit": split_unit,
              "reference_sampler": sampler_identity[0], "splits": {}}
    for split in ("train", "validation", "test"):
        selected = [p for p in profiles if p["split"] == split]
        report["splits"][split] = {"profiles": len(selected), "samples": sum(len(p["distance"]) for p in selected),
            "hits": sum(int(p["hit"].sum()) for p in selected),
            "negative_starts": sum(int((p["initial_sign"] < 0).sum()) for p in selected)}
    write_json(output.with_suffix(".json"), report)
    return report


def load_data(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if (data.get("format"), data.get("version"), data.get("event"), data.get("kernel"), data.get("profile_mode")) != (FORMAT, VERSION, EVENT, SE_KERNEL, "cubic"):
        raise ValueError("incompatible first-hit dataset")
    data["reference_sampler"] = reference_sampler(data.get("reference_sampler"))
    for profile in data["profiles"]:
        require_same_sampler(data["reference_sampler"], profile.get("reference_sampler"), "dataset profile")
    return data


def profile_to_device(profile, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in profile.items()}
