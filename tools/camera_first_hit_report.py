"""Offline interactive views and exact pairwise rankings for first-flight data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).with_name("camera_first_hit_report")
RANKED_METHODS = {"local", "global_conditional", "renewal_1", "renewal_0.1", "renewal_0.01"}
COMPONENTS = {
    "W": "纵向梯度 W = −ω·g × ℓ/σ",
    "norm": "完整梯度模长 |g|",
    "transverse1": "横向梯度 g·e₁",
    "transverse2": "横向梯度 g·e₂",
}


def ecdf_sorted(values, x):
    return np.searchsorted(values, x, side="right") / len(values)


def pairwise_maximum(a, b, survival=False):
    """Exact two-sample CDF distance and a location attaining it, including ties."""
    if not len(a) or not len(b):
        return None
    x = np.union1d(a, b)
    # Escapes have event distance +inf. The maximum before that atom is attained
    # at a finite event; at +inf both CDFs equal one.
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(gap=0., x=0., a=1. if survival else 0., b=1. if survival else 0.)
    ya, yb = ecdf_sorted(a, x), ecdf_sorted(b, x)
    index = int(np.argmax(np.abs(ya-yb)))
    va, vb = float(ya[index]), float(yb[index])
    if survival:
        va, vb = 1-va, 1-vb
    return dict(gap=abs(va-vb), x=float(x[index]), a=va, b=vb)


def make_case(cid, ray, kind, arrays, *, grid=None, window="all", bounds=None, component=None):
    ordered = {m: np.sort(np.asarray(v)) for m, v in arrays.items()}
    eligible = [m for m in ordered if m in RANKED_METHODS]
    pairs = []
    for i, a in enumerate(eligible):
        for b in eligible[i+1:]:
            stat = pairwise_maximum(ordered[a], ordered[b], kind == "transmittance")
            if stat is not None:
                pairs.append(dict(method_a=a, method_b=b, n_a=len(ordered[a]), n_b=len(ordered[b]),
                    sufficient=kind == "transmittance" or min(len(ordered[a]), len(ordered[b])) >= 512, **stat))
    if grid is None:
        finite = [v[np.isfinite(v)] for v in ordered.values() if len(v)]
        pooled = np.concatenate(finite) if finite else np.array([0., 1.])
        if not len(pooled):
            pooled = np.array([0., 1.])
        grid = np.quantile(pooled, np.linspace(0, 1, 1201))
        margin = max(float(np.ptp(grid))*.025, 1e-6)
        grid = np.r_[grid[0]-margin, grid, grid[-1]+margin]
    # Include the exact maxima, rather than hoping a uniform plotting grid sees
    # them. Metrics always use the complete samples, not the display grid.
    x = np.unique(np.r_[grid, [p["x"] for p in pairs]])
    curves = {}
    for m, values in ordered.items():
        if not len(values):
            continue
        y = ecdf_sorted(values, x)
        if kind == "transmittance":
            y = 1-y
        curves[m] = dict(y=y.round(9).tolist(), n=len(values))
    return dict(id=cid, ray=ray["id"], pixel=ray["pixel"], kind=kind, window=window,
                bounds=bounds, component=component, x=x.tolist(), curves=curves, pairs=pairs)


class InteractiveComparison:
    def __init__(self, summary, styles):
        self.payload = dict(version=1, sample_count=summary["trials_per_ray"],
            methods=[dict(id=m["id"], label=styles[m["id"]][0], color=styles[m["id"]][1],
                          dash=styles[m["id"]][2], ranked=m["id"] in RANKED_METHODS)
                     for m in summary["methods"]],
            components=COMPONENTS, rays=summary["rays"], cases=[])

    def add_ray(self, ray, data, profile, windows):
        rid = ray["id"]
        self.payload["cases"].append(make_case(f"t-{rid}",ray,"transmittance",
            {m:d["event"] for m,d in data.items()},grid=profile[:,0]))
        for window in [dict(name="all",low=0.,high=ray["length"]+1e-6),*windows]:
            name = window["name"]
            masks = {m:(d["status"] == 1) & (d["distance"] >= window["low"]) &
                       (d["distance"] < window["high"]) for m,d in data.items()}
            for component in COMPONENTS:
                self.payload["cases"].append(make_case(f"g-{rid}-{name}-{component}",ray,"gradient",
                    {m:d[component][masks[m]] for m,d in data.items()},window=name,
                    bounds=None if name == "all" else [window["low"],window["high"]],component=component))

    def render(self, output):
        rankings = {kind:[] for kind in ("transmittance","gradient")}
        for c in self.payload["cases"]:
            for p in c["pairs"]:
                rankings[c["kind"]].append(dict(case=c["id"],ray=c["ray"],pixel=c["pixel"],
                    window=c["window"],component=c["component"],bounds=c["bounds"],**p))
        for rows in rankings.values():
            rows.sort(key=lambda p:p["gap"],reverse=True)
        (output/"pairwise_differences.json").write_text(json.dumps(rankings,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        encoded = json.dumps(self.payload,ensure_ascii=False,separators=(",",":"),allow_nan=False).replace("</","<\\/")
        # Inline data and assets work when summary.html is opened via file://.
        # No fetch, CDN, server, or package installation is needed.
        template = (ASSETS/"panel.html").read_text(encoding="utf-8")
        return (template.replace("@@CSS@@",(ASSETS/"panel.css").read_text(encoding="utf-8"))
                .replace("@@DATA@@",encoded)
                .replace("@@JS@@",(ASSETS/"panel.js").read_text(encoding="utf-8")))
