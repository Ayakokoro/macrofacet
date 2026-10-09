"""Validate C++ reference CSVs and pack shared profiles with censored trajectory labels."""
from __future__ import annotations

import bisect
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from . import FORMAT_VERSION, KERNEL
from .collect import digest, write_json

SPLITS = {"train": 0, "validation": 1, "test": 2}
FEATURES = ("b0", "b1", "dx_db0", "dx_db1", "log_dx")


def _rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        yield from csv.DictReader(stream)


def prepare(manifest_path: Path, output: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FORMAT_VERSION or manifest.get("kernel") != KERNEL:
        raise ValueError("dataset must declare the unit-decay Matern-3/2 convention")
    packed = {"format_version": FORMAT_VERSION, "kernel": KERNEL, "feature_names": list(FEATURES),
              "manifest_sha256": digest(manifest_path)}
    features, offsets, initial, horizons = [], [0], [], []
    scene_ids, split_ids, profile_keys, provenance = [], [], [], []
    sample_profile, hit, speed, end_segment, end_u, distance = [], [], [], [], [], []
    scenes, geometries, source_paths = {}, {}, set()
    resolution_counts = defaultdict(int)
    for source in manifest["sources"]:
        directory = Path(source["directory"])
        if not directory.is_absolute():
            directory = (manifest_path.parent / directory).resolve()
        if directory in source_paths:
            raise ValueError("duplicate reference source")
        source_paths.add(directory)
        scene, split = source["scene_id"], source["split"]
        if split not in SPLITS or scenes.setdefault(scene, split) != split:
            raise ValueError("a scene must belong to exactly one split")
        geometry = json.dumps(source.get("geometry", {"scene_id": scene}), sort_keys=True)
        if geometries.setdefault(geometry, split) != split:
            raise ValueError("identical geometry appears in multiple splits")
        config_path = directory / "resolved_first_passage_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))["first_passage"]
        kernels = {k["id"]: k for k in config["kernels"]}
        if any(k.get("parameterization") != "unit_decay" or k["type"] != "matern_3_2"
               for k in kernels.values()):
            raise ValueError("legacy or non-Matern-3/2 source; regenerate reference data")
        mode = config["initial_condition"]["type"]
        if mode not in ("positive_exterior", "collision_state"):
            raise ValueError("only Renewal A/B start conditions are supported")
        grouped = defaultdict(list)
        for row in _rows(directory / "first_passage_mean_segments.csv"):
            if row["start_mode"] != mode:
                raise ValueError("profile start mode differs from resolved conditioning")
            grouped[(row["kernel_id"], row["state_id"])].append(row)
        mapping, endpoints = {}, {}
        for key, segments in grouped.items():
            if key[0] not in kernels:
                raise ValueError("segment kernel missing in resolved config")
            segments.sort(key=lambda r: int(r["segment"]))
            previous = 0.0
            ends = []
            previous_value = None
            for i, row in enumerate(segments):
                lo, hi = float(row["x_begin"]), float(row["x_end"])
                f = [float(row[k]) for k in FEATURES]
                if (int(row["segment"]) != i or lo != previous or hi <= lo or
                        not all(math.isfinite(v) for v in [lo, hi, *f]) or
                        not math.isclose(math.exp(f[4]), hi-lo, rel_tol=1e-9, abs_tol=1e-13)):
                    raise ValueError("invalid mean segment partition or units")
                if previous_value is not None and not math.isclose(f[0], previous_value, rel_tol=1e-8, abs_tol=1e-9):
                    raise ValueError("discontinuous mean profile")
                features.append(f)
                previous, previous_value = hi, f[1]
                ends.append(hi)
            if not math.isclose(previous, config["grid"]["max_time"], rel_tol=1e-10):
                raise ValueError("profile horizon differs from censoring horizon")
            p = len(initial)
            mapping[key], endpoints[key] = p, ends
            offsets.append(len(features))
            initial.append(None)  # Filled exclusively from known conditions in the sample metadata.
            horizons.append(previous)
            scene_ids.append(scene)
            split_ids.append(SPLITS[split])
            profile_keys.append([str(directory), *key])
        trajectories = set()
        counts_by_profile = defaultdict(int)
        for row in _rows(directory / "first_passage_samples.csv"):
            if row["training_resolution"] != "1":
                continue
            if row["sampler_type"] != "matern32_state_space":
                raise ValueError("reference labels require exact state transitions")
            key = (row["kernel_id"], row["state_id"])
            p = mapping[key]
            if not math.isclose(float(row["base_step"]), min(config["grid"]["step_sizes"]), rel_tol=1e-10):
                raise ValueError("training_resolution must denote the finest reference step")
            kernel = kernels[key[0]]
            if "ell" in row and (not math.isclose(float(row["ell"]), kernel["length_scale"], rel_tol=1e-10) or
                    not math.isclose(float(row["sigma"])**2, kernel["variance"], rel_tol=1e-10)):
                raise ValueError("sample units differ from the declared physical kernel")
            identity = (*key, row["trajectory"])
            if identity in trajectories:
                raise ValueError("duplicate finest-resolution trajectory")
            trajectories.add(identity)
            counts_by_profile[key] += 1
            f0 = features[offsets[p]]
            b0, db0 = f0[0], f0[2] / math.exp(f0[4])
            if not math.isclose(float(row["beta_0"]), b0, rel_tol=1e-8, abs_tol=1e-9):
                raise ValueError("birth mean does not match segment features")
            known = [0.0, b0, 0.0, 0.0]
            if mode == "collision_state":
                g = float(row["beta_g"])
                if not math.isfinite(g) or g <= 0:
                    raise ValueError("surface starts require a known outward derivative")
                known = [1.0, b0, -b0, g-db0]
            elif row["beta_g"]:
                raise ValueError("exterior starts must not contain a sampled derivative")
            if initial[p] is not None and initial[p] != known:
                raise ValueError("random realization leaked into initial conditions")
            initial[p] = known
            event = row["event"] == "1"
            if row["event"] not in ("0", "1") or int(row["censored"]) != int(not event):
                raise ValueError("invalid censoring indicator")
            x, horizon = float(row["event_q"]), horizons[p]
            w = float(row["crossing_slope"]) if event else math.nan
            if not (math.isfinite(x) and 0 < x <= horizon and
                    math.isclose(float(row["max_q"]), horizon, rel_tol=1e-10)):
                raise ValueError("invalid event distance or censoring horizon")
            if event and (not math.isfinite(w) or w <= 0):
                raise ValueError("hit speeds must be positive and finite")
            if "event_distance" in row and not math.isclose(float(row["event_distance"]),
                    x*float(row["ell"]), rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("physical event distance differs from ell*x")
            if event and "crossing_derivative" in row and not math.isclose(float(row["crossing_derivative"]),
                    -w*float(row["sigma"])/float(row["ell"]), rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("physical crossing derivative differs from -sigma/ell*w")
            if not event and (x != horizon or row["crossing_slope"]):
                raise ValueError("misses must remain censored and have no speed label")
            s = bisect.bisect_left(endpoints[key], x)
            lo = 0.0 if s == 0 else endpoints[key][s-1]
            u = (x-lo) / (endpoints[key][s]-lo)
            sample_profile.append(p); hit.append(event); speed.append(w)
            end_segment.append(s); end_u.append(u); distance.append(x)
            resolution_counts[row["base_step"]] += 1
        if any(initial[p] is None for p in mapping.values()):
            raise ValueError("profile has no finest-resolution samples")
        expected = config.get("monte_carlo", {}).get("trajectories")
        if expected is not None and any(n != expected for n in counts_by_profile.values()):
            raise ValueError("incomplete trajectory data; do not drop hits or censored samples")
        provenance.append({"scene_id": scene, "split": split, "directory": str(directory),
                           "resolved_sha256": digest(config_path),
                           "samples_sha256": digest(directory / "first_passage_samples.csv"),
                           "segments_sha256": digest(directory / "first_passage_mean_segments.csv"),
                           "reference_grid": config["grid"], "sampler": config["sampler"]})
    if not sample_profile or set(scenes.values()) != set(SPLITS):
        raise ValueError("nonempty, geometry-disjoint train/validation/test splits are required")
    # Retain double geometry: float32 endpoint cancellation corrupts derivatives
    # in the very short cells near voxel edges. Neural encoders cast afterwards.
    for name, values, dtype in (("features", features, torch.float64), ("offsets", offsets, torch.long),
        ("initial", initial, torch.float64), ("horizon", horizons, torch.float64),
        ("split", split_ids, torch.long), ("sample_profile", sample_profile, torch.long),
        ("hit", hit, torch.bool), ("speed", speed, torch.float32),
        ("end_segment", end_segment, torch.long), ("end_u", end_u, torch.float32),
        ("distance", distance, torch.float64)):
        packed[name] = torch.tensor(values, dtype=dtype)
    packed.update(scene_ids=scene_ids, profile_keys=profile_keys, provenance=provenance)
    counts = {}
    for split, i in SPLITS.items():
        mask = packed["split"][packed["sample_profile"]] == i
        counts[split] = {"scenes": sum(v == split for v in scenes.values()),
                         "profiles": split_ids.count(i), "samples": int(mask.sum()),
                         "hits": int(packed["hit"][mask].sum())}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packed, output)
    report = {"counts": counts, "kernel": KERNEL, "finest_resolution_counts": dict(resolution_counts),
              "segments": len(features), "manifest_sha256": packed["manifest_sha256"],
              "dataset_sha256": digest(output)}
    write_json(output.with_suffix(".json"), report)
    return report


def load_data(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("format_version") != FORMAT_VERSION or data.get("kernel") != KERNEL:
        raise ValueError("unsupported dataset format or kernel convention")
    return data


class SequenceDataset(Dataset):
    def __init__(self, data: dict, split: str):
        self.data = data
        self.indices = torch.where(data["split"][data["sample_profile"]] == SPLITS[split])[0].tolist()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        d, index = self.data, self.indices[index]
        p = int(d["sample_profile"][index])
        begin = int(d["offsets"][p])
        length = int(d["end_segment"][index]) + 1
        return {"features": d["features"][begin:begin+length], "initial": d["initial"][p],
                "hit": d["hit"][index], "speed": d["speed"][index], "u": d["end_u"][index],
                "profile": torch.tensor(p)}


def collate(records):
    batch = {key: torch.stack([r[key] for r in records])
             for key in ("initial", "hit", "speed", "u", "profile")}
    batch["features"] = pad_sequence([r["features"] for r in records], batch_first=True)
    batch["lengths"] = torch.tensor([len(r["features"]) for r in records], dtype=torch.long)
    return batch


def to_device(batch, device):
    # Packed sequence lengths are intentionally retained on CPU.
    return {k: v if k == "lengths" else v.to(device) for k, v in batch.items()}
