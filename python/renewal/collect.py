"""Generate reproducible geometry-disjoint reference experiments with the C++ sampler."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from . import FORMAT_VERSION, KERNEL

ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _invoke(executable: Path, command: str, config: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "generation.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run([str(executable), command, "--config", str(config)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        tail = (output / "generation.log").read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"{command} failed for {config}:\n{tail}")


def _unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def _geometry(rng, family: str):
    radius = rng.uniform(0.45, 0.9)
    center = rng.uniform(-0.1, 0.1, 3)
    spec = {"mean_type": family}
    if family == "plane":
        spec.update(plane_normal=_unit(rng).tolist(), plane_offset=float(rng.uniform(-0.2, 0.2)))
    else:
        spec["sphere_center"] = center.tolist()
        if family == "cutaway_sphere":
            spec.update(outer_radius=float(radius), inner_radius=float(radius * rng.uniform(0.35, 0.7)))
        else:
            spec["sphere_radius"] = float(radius)
            if family == "shader_ball":
                spec.update(groove_axis=_unit(rng).tolist(), groove_radius=float(radius * rng.uniform(0.05, 0.15)))
    return spec


def _ray(rng, spec, sigma: float, ell: float, surface: bool, index: int):
    family = spec["mean_type"]
    if family == "plane":
        normal = np.array(spec["plane_normal"])
        p = rng.uniform(-0.8, 0.8, 3)
        p += (spec["plane_offset"] - np.dot(p, normal)) * normal
    else:
        normal = _unit(rng)
        radius = spec.get("sphere_radius", spec.get("outer_radius"))
        if family == "cutaway_sphere":
            if index % 3 == 0:
                normal[:2] = np.abs(normal[:2])
                radius = spec["inner_radius"]
            elif index % 3 == 1:
                normal[0] = -abs(normal[0])
            else:
                # A cut face between the inner and outer spheres.
                r = rng.uniform(spec["inner_radius"], radius)
                angle = rng.uniform(-math.pi / 2, math.pi / 2)
                p = np.array([0, r * math.cos(angle), r * math.sin(angle)])
                normal = np.array([1.0, 0, 0])
                radius = None
        if radius is not None:
            p = radius * normal
        p += np.array(spec["sphere_center"])
    p += sigma * rng.uniform(-1.5, 2.0) * normal
    direction = _unit(rng)
    if index % 3 == 0:  # Add near-tangent rays without restricting the rest.
        direction -= np.dot(direction, normal) * normal
        direction += rng.uniform(-0.15, 0.15) * normal
        direction /= np.linalg.norm(direction)
    result = {"id": f"ray_{index:05d}", "origin": p.tolist(), "direction": direction.tolist()}
    if surface:
        outward = math.exp(rng.uniform(math.log(0.15), math.log(3.0)))
        transverse = rng.normal(size=3)
        transverse -= np.dot(transverse, direction) * direction
        result["gradient"] = ((sigma / ell) * (outward * direction + transverse)).tolist()
    return result


def collect(config_path: Path) -> Path:
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    if settings.get("schema_version") != FORMAT_VERSION:
        raise ValueError("unsupported collection schema")
    executable = resolve(settings["executable"])
    if not executable.is_file():
        raise FileNotFoundError(executable)
    output = resolve(settings["output_directory"])
    saved = output / "collection_config.json"
    if saved.exists() and json.loads(saved.read_text(encoding="utf-8")) != settings:
        raise ValueError("collection settings changed: use a new output directory")
    write_json(saved, settings)
    count = int(settings["profiles_per_mode_per_scene"])
    horizons = settings["horizons"]
    if count < len(horizons) or count % len(horizons):
        raise ValueError("profiles_per_mode_per_scene must be divisible by horizon count")
    if any(float(x) <= 0 for x in horizons) or settings["realizations_per_profile"] < 1:
        raise ValueError("positive horizons and realization counts are required")
    manifest = {"schema_version": FORMAT_VERSION, "kernel": KERNEL, "sources": [],
                "collection_config_sha256": digest(saved), "generator_sha256": digest(executable)}
    families = settings["families"]
    scene_index = 0
    for split in ("train", "validation", "test"):
        for local in range(int(settings["scenes"][split])):
            scene_id = f"{split}_{local:03d}_{families[local % len(families)]}"
            rng = np.random.default_rng(np.random.SeedSequence([settings["seed"], scene_index]))
            spec = _geometry(rng, families[local % len(families)])
            sigma = float(rng.uniform(*settings["sigma_range"]))
            ell = float(rng.uniform(*settings["ell_range"]))
            # All endpoints and interpolation corners must lie in the full-domain bake.
            radius = spec.get("outer_radius", spec.get("sphere_radius", 1.0))
            bound = float(radius + 0.3 + 2 * sigma + max(horizons) * ell)
            field = {**spec, "sigma": sigma, "correlation_lengths": [ell] * 3,
                     "domain_min": [-bound] * 3, "domain_max": [bound] * 3,
                     "bake_voxel_size": float(rng.uniform(*settings["voxel_size_range"]))}
            directory = output / scene_id
            bake_dir = directory / "bake"
            bake = {"schema_version": 1, "seed": settings["seed"], "field": field,
                    "material": {"ndf_family": "generalized_gaussian", "eta_rgb": [0.2, 0.9, 1.1],
                                 "k_rgb": [3.9, 2.5, 2.2]}, "transport": {"mode": "classic_local"},
                    "numeric": {"relative_tolerance": 1e-6, "absolute_tolerance": 1e-9,
                                "max_quadrature_subdivisions": 2048, "max_root_iterations": 128},
                    "render": {"width": 1, "height": 1, "samples_per_pixel": 1,
                               "camera_position": [0, 0, 4], "camera_target": [0, 0, 0],
                               "vertical_fov_degrees": 45, "environment": "unit_white"},
                    "output_directory": str(bake_dir)}
            bake_config = directory / "bake.json"
            write_json(bake_config, bake)
            _invoke(executable, "bake-field", bake_config, bake_dir)
            resolved = json.loads((bake_dir / "resolved_config.json").read_text(encoding="utf-8"))
            mean = {"mean_type": "nanovdb", "grid_file": resolved["field"]["grid_file"],
                    "sigma": sigma, "use_alpha_grid": False}
            for mode_index, mode in enumerate(("positive_exterior", "collision_state")):
                for h_index, horizon in enumerate(horizons):
                    run = directory / f"{mode}_h{h_index}"
                    rays = [_ray(rng, spec, sigma, ell, mode_index == 1, i)
                            for i in range(count // len(horizons))]
                    source_seed = int(np.random.SeedSequence(
                        [settings["seed"], scene_index, mode_index, h_index, 9981]).generate_state(1)[0])
                    source = {"schema_version": 1, "seed": source_seed, "output_directory": str(run),
                              "first_passage": {
                                  "process": {"mean_field": mean},
                                  "initial_condition": {"type": mode, "rays": rays},
                                  "grid": {"max_time": horizon, "step_sizes": settings["reference_steps"]},
                                  "profile": {"maximum_step": settings["profile_maximum_step"]},
                                  "curve": {"bins": 32},
                                  "monte_carlo": {"trajectories": settings["realizations_per_profile"],
                                                  "thread_count": settings["threads"], "write_raw_samples": True},
                                  "sampler": {"type": "collision_state_auto",
                                              "minimum_step": settings["minimum_step"],
                                              "crossing_tolerance": 1e-9, "bridge_sigma_margin": 6,
                                              "max_refinement_depth": 16},
                                  "kernels": [{"id": "matern32", **KERNEL,
                                               "variance": sigma * sigma, "length_scale": ell}]}}
                    source_path = run / "input.json"
                    write_json(source_path, source)
                    # A matching input and executable are needed to reuse an existing source.
                    stamp = {"input": digest(source_path), "executable": manifest["generator_sha256"]}
                    stamp_path = run / "complete.json"
                    required = ("resolved_first_passage_config.json", "first_passage_mean_segments.csv",
                                "first_passage_samples.csv", "first_passage_summary.csv")
                    if not (stamp_path.exists() and json.loads(stamp_path.read_text()) == stamp and
                            all((run / name).is_file() for name in required)):
                        _invoke(executable, "first-passage", source_path, run)
                        write_json(stamp_path, stamp)
                    manifest["sources"].append({"scene_id": scene_id, "split": split,
                        "directory": str(run), "geometry": spec, "voxel_size": field["bake_voxel_size"],
                        "sigma": sigma, "ell": ell, "mode": mode, "input_sha256": stamp["input"]})
            scene_index += 1
            print(f"collected {scene_id} ({scene_index}/{sum(settings['scenes'].values())})", flush=True)
    path = output / "manifest.json"
    write_json(path, manifest)
    return path
