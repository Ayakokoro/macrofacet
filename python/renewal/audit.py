"""Compare reference resolutions before interpreting learned-model errors."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .collect import write_json


def audit(manifest_path: Path, output: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = defaultdict(lambda: {"count": 0, "hits": 0, "speed": [], "survival": np.zeros(4),
                                   "per_profile": {}})
    for source in manifest["sources"]:
        directory = Path(source["directory"])
        if not directory.is_absolute():
            directory = manifest_path.parent / directory
        config = json.loads((directory / "resolved_first_passage_config.json").read_text())["first_passage"]
        mode = config["initial_condition"]["type"]
        with (directory / "first_passage_samples.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                group = groups[(source["split"], mode, float(row["base_step"]))]
                event = row["event"] == "1"
                group["count"] += 1
                group["hits"] += event
                if event:
                    group["speed"].append(float(row["crossing_slope"]))
                survived = np.ones(4) if not event else (
                    float(row["event_q"]) > np.array([0.25, 0.5, 0.75, 1.0]) * float(row["max_q"]))
                group["survival"] += survived
                profile = (str(directory), row["kernel_id"], row["state_id"])
                if profile not in group["per_profile"]:
                    group["per_profile"][profile] = [0, 0]
                group["per_profile"][profile][0] += event
                group["per_profile"][profile][1] += 1
    report = []
    for (split, mode, step), group in sorted(groups.items()):
        report.append({"split": split, "mode": mode, "step": step, "samples": group["count"],
                       "hit_probability": group["hits"]/group["count"],
                       "survival_at_horizon_fractions": (group["survival"]/group["count"]).tolist(),
                       "speed_quantiles": np.quantile(group["speed"], [0.1, 0.5, 0.9]).tolist() if group["speed"] else []})
    differences = []
    for split, mode in sorted({(k[0], k[1]) for k in groups}):
        resolutions = sorted(k[2] for k in groups if k[:2] == (split, mode))
        if len(resolutions) < 2:
            continue
        fine, coarse = (groups[(split, mode, step)] for step in (resolutions[0], resolutions[1]))
        if fine["per_profile"].keys() != coarse["per_profile"].keys():
            raise ValueError("reference resolutions must use exactly the same deterministic profiles")
        deltas = np.array([coarse["per_profile"][p][0]/coarse["per_profile"][p][1] -
                           fine["per_profile"][p][0]/fine["per_profile"][p][1] for p in fine["per_profile"]])
        differences.append({"split": split, "mode": mode, "coarse_minus_fine_hit_probability": float(deltas.mean()),
                            "profile_cluster_standard_error": float(deltas.std(ddof=1)/np.sqrt(len(deltas))) if len(deltas)>1 else None,
                            "max_survival_gap": float(np.max(np.abs(coarse["survival"]/coarse["count"]-
                                                                                    fine["survival"]/fine["count"])))})
    result = {"resolutions": report, "differences": differences,
              "note": "Independent realizations per resolution; these diagnostics are not a certified first-passage error bound."}
    write_json(output, result)
    return result
