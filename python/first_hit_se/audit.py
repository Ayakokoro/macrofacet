"""Empirical reference-grid convergence, including first-zero crossing speeds."""
from collections import defaultdict
import math
from pathlib import Path

import numpy as np

from renewal.collect import write_json
from .data import read_json, rows


def cdf_gap(a,b):
    if not len(a) or not len(b):
        return None
    a,b=np.sort(a),np.sort(b)
    query=np.unique(np.concatenate((a[np.isfinite(a)],b[np.isfinite(b)])))
    return float(np.max(np.abs(np.searchsorted(a,query,side="right")/len(a)-np.searchsorted(b,query,side="right")/len(b)))) if len(query) else 0.


def audit(manifest_path, output):
    manifest_path=Path(manifest_path)
    report={"profiles":[],"note":"Empirical grid comparison; DKW bounds measure Monte Carlo uncertainty, not remaining continuous-path discretization bias. Speed CDFs are conditional on a hit and entry sign."}
    for source in read_json(manifest_path)["sources"]:
        directory=Path(source["directory"])
        if not directory.is_absolute(): directory=manifest_path.parent/directory
        grouped=defaultdict(lambda:defaultdict(list))
        for row in rows(directory/"first_passage_samples.csv"):
            grouped[row["kernel_id"],row["state_id"]][float(row["base_step"])].append(row)
        for key,grids in grouped.items():
            steps=sorted(grids)
            fine=grids[steps[0]]
            def distances(samples): return np.array([float(r["event_q"]) if r["event"]=="1" else np.inf for r in samples])
            for step in steps[1:]:
                coarse=grids[step]
                speeds={}
                for sign in (1,-1):
                    a=[float(r["crossing_slope"]) for r in coarse if r["event"]=="1" and int(r["initial_sign"])==sign]
                    b=[float(r["crossing_slope"]) for r in fine if r["event"]=="1" and int(r["initial_sign"])==sign]
                    speeds[str(sign)]={"coarse_hits":len(a),"fine_hits":len(b),"cdf_gap":cdf_gap(a,b),
                        "coarse_quantiles":np.quantile(a,[.1,.5,.9]).tolist() if a else None,
                        "fine_quantiles":np.quantile(b,[.1,.5,.9]).tolist() if b else None}
                report["profiles"].append({"source":str(directory),"kernel":key[0],"state":key[1],
                    "coarse_step":step,"fine_step":steps[0],"coarse_samples":len(coarse),"fine_samples":len(fine),
                    "survival_cdf_gap":cdf_gap(distances(coarse),distances(fine)),
                    "dkw_95_sum":sum(math.sqrt(math.log(80)/(2*n)) for n in (len(coarse),len(fine))),
                    "speed_by_initial_sign":speeds})
    if not report["profiles"]:
        raise ValueError("reference audit requires at least two stored resolutions")
    write_json(Path(output),report)
    return {"output":str(output),"comparisons":len(report["profiles"]),
            "maximum_survival_cdf_gap":max(p["survival_cdf_gap"] for p in report["profiles"])}
