"""Plot the native first-flight comparison; no neural inference or resampling.

python tools/analyze_camera_first_hit.py --input outputs/.../main --pilot outputs/.../pilot
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))
sys.path.insert(0, str(ROOT / "outputs/.analysis_deps"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from camera_first_hit_report import ASSETS, InteractiveComparison

STYLE = {
    "local": ("local", "#777777", "-"),
    "global_conditional": ("global conditional", "#111111", "-"),
    "gpis_1": ("GPIS step=1", "#cc79a7", "--"),
    "gpis_0.1": ("GPIS step=0.1", "#e69f00", "--"),
    "gpis_0.01": ("GPIS step=0.01", "#009e73", "-"),
    "renewal_1": ("renewal+ distance=1", "#9467bd", ":"),
    "renewal_0.1": ("renewal+ distance=0.1", "#0072b2", "-."),
    "renewal_0.01": ("renewal+ distance=0.01", "#d55e00", "-"),
}
REFERENCE = "gpis_0.01"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ecdf(values, x):
    return np.searchsorted(np.sort(values), x, side="right") / len(values)


def ks_distance(a, b):
    if not len(a) or not len(b):
        return None
    x = np.union1d(a, b)
    return float(np.max(np.abs(ecdf(a, x) - ecdf(b, x))))


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    w = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return [max(0.0, c-w), min(1.0, c+w)]


def load_samples(folder, job):
    a = np.loadtxt(folder / job["file"], delimiter=",", skiprows=1, ndmin=2)
    if len(a) != job["trials"] or not np.array_equal(a[:, 0], np.arange(len(a))):
        raise ValueError("incomplete or reordered sample file: " + job["file"])
    status = a[:, 1].astype(int)
    if np.any(status == 3):
        raise ValueError("invalid gradients must be resolved before drawing uncensored survival curves")
    if not np.isfinite(a).all() or np.any((status < 0) | (status > 2)):
        raise ValueError("invalid sample data")
    for code, key in enumerate(("escapes", "hits", "rejected_upcrossings")):
        if np.count_nonzero(status == code) != job[key]:
            raise ValueError("sample counts do not match metadata")
    return dict(raw=a, status=status, distance=a[:, 2], gradient=a[:, 3:6],
                event=np.where(status == 0, np.inf, a[:, 2]))


def draw_cdf(ax, values, method, xmin=None, xmax=None):
    if not len(values):
        return
    v = np.sort(values)
    # Downsample the drawing only. All metrics use every raw sample.
    idx = np.unique(np.rint(np.linspace(0, len(v)-1, min(1500, len(v)))).astype(int))
    label, color, style = STYLE[method]
    ax.plot(v[idx], (idx+1)/len(v), color=color, ls=style, lw=1.35, label=label)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.18)
    if xmin is not None or xmax is not None:
        ax.set_xlim(xmin, xmax)


def save(fig, folder, name):
    fig.savefig(folder / (name + ".png"), dpi=160, bbox_inches="tight")
    fig.savefig(folder / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


def html_table(headers, rows):
    return "<table><thead><tr>" + "".join(f"<th>{html.escape(str(v))}</th>" for v in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table>"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--image", type=Path)
    args = p.parse_args()
    folder = args.input.resolve()
    summary = read(folder / "summary.json")
    pilot = read(args.pilot / "summary.json")
    if summary["seed"] == pilot["seed"]:
        raise ValueError("distance windows require an independent pilot seed")
    out = folder.parent
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    methods = [m["id"] for m in summary["methods"]]
    n = summary["trials_per_ray"]
    eps = math.sqrt(math.log(40)/(2*n))
    pilot_rays = {tuple(r["pixel"]): r for r in pilot["rays"]}
    jobs = {(j["ray"], j["method"]): j for j in summary["samples"]}
    pilot_jobs = {(j["ray"], j["method"]): j for j in pilot["samples"]}
    nr = len(summary["rays"])
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    tfig, tax = plt.subplots(nr, 2, figsize=(13, 3.0*nr), squeeze=False)
    gfig, gax = plt.subplots(nr, 2, figsize=(13, 3.0*nr), squeeze=False)
    pfig, pax = plt.subplots(nr, 1, figsize=(12, 2.5*nr), squeeze=False)
    metric_rows, window_rows, validation = [], [], []
    windows, detail_sections = [], []
    interactive = InteractiveComparison(summary, STYLE)
    for r in summary["rays"]:
        rid = r["id"]
        label = f"Ray {rid}: pixel {tuple(r['pixel'])}"
        data = {m: load_samples(folder, jobs[rid, m]) for m in methods}
        ref = data[REFERENCE]
        pr = pilot_rays[tuple(r["pixel"])]
        pa = load_samples(args.pilot, pilot_jobs[pr["id"], REFERENCE])
        ph = pa["distance"][pa["status"] == 1]
        if len(ph) < 30:
            raise ValueError(f"insufficient pilot hits for distance windows: {label}")
        centers = np.quantile(ph, [.1, .5, .9])
        half = min(.25*r["ell"], .45*float(np.min(np.diff(centers))))
        ray_windows = [{"name": name, "center": float(c), "low": max(0., float(c-half)),
                        "high": min(r["length"], float(c+half))}
                       for name, c in zip(("early", "middle", "late"), centers)]
        windows.append({"ray": rid, "pixel": r["pixel"], "windows": ray_windows})
        direction = np.array(r["direction"])
        axis = np.eye(3)[np.argmin(np.abs(direction))]
        e1 = np.cross(direction, axis); e1 /= np.linalg.norm(e1)
        e2 = np.cross(direction, e1)
        for d in data.values():
            d["W"] = -(d["gradient"] @ direction) * r["ell"] / summary["sigma"]
            d["norm"] = np.linalg.norm(d["gradient"], axis=1)
            d["transverse1"] = d["gradient"] @ e1
            d["transverse2"] = d["gradient"] @ e2
        profile = np.loadtxt(folder/f"ray_{rid}_profile.csv", delimiter=",", skiprows=1)
        interactive.add_ray(r, data, profile, ray_windows)
        x = profile[:, 0]
        local_integral = np.exp(-cumulative_trapezoid(profile[:, 4], x, initial=0))
        ref_t = 1-ecdf(ref["event"], x)
        all_events = np.concatenate([d["event"][np.isfinite(d["event"])] for d in data.values()])
        lo, hi = np.quantile(all_events, [.001, .999])
        lo = max(0., lo-.025); hi = min(r["length"], hi+.025)
        for m, d in data.items():
            title, color, ls = STYLE[m]
            t = 1-ecdf(d["event"], x)
            for ax in tax[rid]:
                ax.plot(x, t, color=color, ls=ls, lw=1.5, label=title)
            for ax, key in zip(gax[rid], ("W", "norm")):
                draw_cdf(ax, d[key][d["status"] == 1], m)
            h = d["status"] == 1
            rh = ref["status"] == 1
            job = jobs[rid, m]
            metric_rows.append(dict(ray=rid, pixel=r["pixel"], method=m, n=n,
                escape_probability=job["escapes"]/n, escape_ci95=wilson(job["escapes"], n),
                rejected_probability=job["rejected_upcrossings"]/n, accepted_hits=int(h.sum()),
                max_transmittance_difference=ks_distance(d["event"], ref["event"]),
                mean_absolute_transmittance_difference=float(np.trapezoid(np.abs(t-ref_t), x)/r["length"]),
                gradient_W_ks_all_hits=ks_distance(d["W"][h], ref["W"][rh]),
                gradient_norm_ks_all_hits=ks_distance(d["norm"][h], ref["norm"][rh]),
                W_mean=float(np.mean(d["W"][h])) if h.any() else None,
                gradient_norm_mean=float(np.mean(d["norm"][h])) if h.any() else None))
        tax[rid, 0].set_title(label + " | full medium segment")
        tax[rid, 1].set_title(label + " | collision region")
        tax[rid, 0].set_xlim(0, r["length"])
        tax[rid, 1].set_xlim(lo, hi)
        for ax in tax[rid]:
            ax.set_ylim(-.02, 1.02); ax.set_ylabel("Transmittance T(s)")
            ax.set_xlabel("s from medium entry (world units)"); ax.grid(alpha=.18)
        tax[rid, 0].fill_between(x, np.maximum(0, ref_t-eps), np.minimum(1, ref_t+eps),
                                color=STYLE[REFERENCE][1], alpha=.10)
        for ax, key in zip(gax[rid], ("W", "norm")):
            ax.set_title(label + " | all accepted first hits")
            ax.set_ylabel("CDF")
            ax.set_xlabel("W = -dot(direction, gradient) * ell / sigma" if key == "W" else "Full gradient norm |g|")
        ax = pax[rid, 0]
        ax.plot(x, profile[:, 1]/summary["sigma"], color="#111111", lw=1.2)
        ax.axhline(0, color="#777777", lw=.7)
        for w in ray_windows:
            ax.axvspan(w["low"], w["high"], alpha=.15)
            ax.text(w["center"], .98, w["name"], ha="center", va="top", transform=ax.get_xaxis_transform())
        ax.set_title(label + f" | entry mean / sigma = {r['entry_mean_sigma']:.3f}")
        ax.set_xlabel("s from medium entry (world units)"); ax.set_ylabel("Mean SDF / sigma")
        ax.grid(alpha=.18)
        local_t = 1-ecdf(data["local"]["event"], x)
        entry_drift = max(abs(jobs[rid,m]["entry_distance"]-r["entry_distance"]) for m in methods)
        row = dict(ray=rid, local_mc_vs_trapezoid_max=float(np.max(np.abs(local_t-local_integral))),
                   max_entry_float_drift=entry_drift, distance_valid=True, counts_valid=True)
        if r["length"] <= 1:
            # With identical random streams and no renewal, these must be exactly identical.
            same = np.array_equal(data["global_conditional"]["raw"], data["renewal_1"]["raw"])
            row["gc_equals_renewal1_exactly"] = same
            if not same:
                raise ValueError("GC and renewal+ differ despite no renewal before domain exit")
        validation.append(row)
        fig, axes = plt.subplots(3, 3, figsize=(14, 10))
        ray_window_rows = []
        for wi, w in enumerate(ray_windows):
            rh = (ref["status"] == 1) & (ref["distance"] >= w["low"]) & (ref["distance"] < w["high"])
            for m, d in data.items():
                h = (d["status"] == 1) & (d["distance"] >= w["low"]) & (d["distance"] < w["high"])
                values = {}
                for ai, key in enumerate(("W", "transverse1", "transverse2")):
                    draw_cdf(axes[wi, ai], d[key][h], m)
                    values[key+"_ks"] = ks_distance(d[key][h], ref[key][rh])
                rec = dict(ray=rid, method=m, window=w["name"], low=w["low"], high=w["high"],
                           hits=int(h.sum()), reference_hits=int(rh.sum()), sufficient_hits=bool(min(h.sum(), rh.sum())>=512), **values)
                window_rows.append(rec)
                ray_window_rows.append([w["name"], STYLE[m][0], int(h.sum()),
                    f"{values['W_ks']:.4f}" if values["W_ks"] is not None else "n/a",
                    "OK" if rec["sufficient_hits"] else "LOW N"])
            for ai, name in enumerate(("W (normalized downward speed)", "dot(g, transverse axis 1)", "dot(g, transverse axis 2)")):
                axes[wi, ai].set_title(f"{w['name']}: s in [{w['low']:.5f}, {w['high']:.5f})")
                axes[wi, ai].set_xlabel(name); axes[wi, ai].set_ylabel("Conditional gradient CDF")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(.5, 1.035))
        fig.suptitle(label + " | fixed distance windows, pilot-selected", y=1.065)
        fig.tight_layout()
        save(fig, figures, f"ray_{rid}_gradient_windows")
        detail_sections.append(f"<details><summary>射线 {rid}，像素 {r['pixel']}：早 / 中 / 晚碰撞梯度</summary>"
            f"<p>横向梯度基：e1={e1.round(6).tolist()}，e2={e2.round(6).tolist()}。</p>"
            f"<a href='figures/ray_{rid}_gradient_windows.svg'><img src='figures/ray_{rid}_gradient_windows.png'></a>"
            + html_table(["窗口", "方法", "命中样本数", "W CDF 最大差（对 GPIS .01）", "样本量"], ray_window_rows) + "</details>")
    for fig, axes, name in ((tfig,tax,"transmittance"),(gfig,gax,"gradient_all_hits")):
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(.5,1.025))
        fig.tight_layout()
        save(fig,figures,name)
    pfig.tight_layout(); save(pfig,figures,"mean_profiles")
    if args.image:
        image = plt.imread(args.image)
        fig, ax = plt.subplots(figsize=(7,7)); ax.imshow(image)
        for r in summary["rays"]:
            x,y=r["pixel"]
            ax.plot(x,y,"o",ms=6,mfc="none",mec="#00ffff",mew=1.5)
            ax.annotate(str(r["id"]),(x+5,y-6),color="#00ffff",fontsize=12,fontweight="bold")
        ax.set_title("Selected fixed camera pixels (zero based)"); ax.set_axis_off()
        save(fig,figures,"selected_rays")
    aggregate = []
    common_windows = {(r["id"], name) for r in summary["rays"] for name in ("early", "middle", "late")}
    common_windows -= {(v["ray"], v["window"]) for v in window_rows if not v["sufficient_hits"]}
    for m in methods:
        rows = [v for v in metric_rows if v["method"] == m]
        win = [v for v in window_rows if v["method"] == m and v["sufficient_hits"]]
        common = [v for v in win if (v["ray"], v["window"]) in common_windows]
        aggregate.append(dict(method=m,
            mean_max_transmittance_difference=float(np.mean([v["max_transmittance_difference"] for v in rows])),
            worst_max_transmittance_difference=max(v["max_transmittance_difference"] for v in rows),
            mean_all_hit_W_ks=float(np.mean([v["gradient_W_ks_all_hits"] for v in rows])),
            mean_window_W_ks=float(np.mean([v["W_ks"] for v in win])) if win else None,
            mean_common_window_W_ks=float(np.mean([v["W_ks"] for v in common])) if common else None,
            worst_window_W_ks=max((v["W_ks"] for v in win), default=None),
            mean_window_transverse1_ks=float(np.mean([v["transverse1_ks"] for v in common])) if common else None,
            mean_window_transverse2_ks=float(np.mean([v["transverse2_ks"] for v in common])) if common else None,
            common_sufficient_windows=len(common),
            sufficient_windows=len(win), rejected=sum(jobs[r["id"],m]["rejected_upcrossings"] for r in summary["rays"])))
    result = dict(reference=REFERENCE, reference_is_exact=False, dkw95_single_curve=eps,
                  dkw95_pair_difference_bound=2*math.sqrt(math.log(80)/(2*n)),
                  pilot_seed=pilot["seed"], seed=summary["seed"], trials_per_ray=n,
                  aggregate=aggregate, rays=metric_rows, windows=window_rows, validation=validation)
    # Descriptive follow-up on the largest fine-renewal discrepancy: shrinking
    # the window checks whether within-bin mixing alone explains the difference.
    focus = max((v for v in window_rows if v["method"] == "renewal_0.01" and v["sufficient_hits"]),
                key=lambda v: v["W_ks"])
    focus_ray = summary["rays"][focus["ray"]]
    center = (focus["low"]+focus["high"])/2
    half = (focus["high"]-focus["low"])/8
    focused = {}
    for m in methods:
        d = load_samples(folder, jobs[focus["ray"],m])
        mask = (d["status"] == 1) & (np.abs(d["distance"]-center) < half)
        focused[m] = -(d["gradient"][mask] @ np.array(focus_ray["direction"])) * focus_ray["ell"]/summary["sigma"]
    result["narrow_window_check"] = dict(ray=focus["ray"], pixel=focus_ray["pixel"], low=center-half, high=center+half,
        selection="largest renewal_0.01 W discrepancy; descriptive reuse of main samples, not independent validation",
        methods=[dict(method=m, hits=len(focused[m]), W_ks=ks_distance(focused[m],focused[REFERENCE])) for m in methods])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ypos = np.arange(len(methods))
    for ax, key, title in zip(axes,
            ("mean_max_transmittance_difference", "mean_common_window_W_ks"),
            ("Mean maximum transmittance difference", "Mean gradient W CDF difference (common windows)")):
        values = [100*a[key] if a[key] is not None else 0 for a in aggregate]
        ax.barh(ypos, values, color=[STYLE[m][1] for m in methods], alpha=.85)
        ax.set_yticks(ypos, [STYLE[m][0] for m in methods]); ax.invert_yaxis()
        ax.set_xlabel("CDF / probability difference (percentage points)")
        ax.set_title(title); ax.grid(axis="x", alpha=.2); ax.set_axisbelow(True)
        ax.set_xlim(0, max(values)*1.22 if max(values)>0 else 1)
        for y, value, m in zip(ypos, values, methods):
            ax.text(value+.15, y, "reference" if m == REFERENCE else f"{value:.2f}", va="center", fontsize=9)
    fig.suptitle(f"{nr} selected camera rays, {n:,} samples per method per ray | discrete reference: GPIS step=0.01")
    fig.tight_layout(); save(fig, figures, "comparison_summary")
    write(out/"metrics.json",result); write(out/"distance_windows.json",windows)
    for name, rows in (("transmittance_gradient_metrics",metric_rows),("gradient_window_metrics",window_rows),("aggregate_metrics",aggregate)):
        with (out/(name+".csv")).open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    source=read(ROOT/summary["source_config"])
    interactive_section = interactive.render(out)
    paths=[Path(__file__).resolve(), ROOT/"tools/compare_camera_first_hit.cpp", ROOT/"build/Release/macrofacet_compare_camera_first_hit.exe",
           ROOT/summary["source_config"], ROOT/source["field"]["grid_file"],folder/"summary.json",folder/"request.json",args.pilot/"summary.json",
           ROOT/"tools/camera_first_hit_report.py", *sorted(ASSETS.glob("panel.*"))]
    write(out/"provenance.json",{str(path):digest(path) for path in paths})
    atable=html_table(["方法", "平均 max|ΔT|", "最差 max|ΔT|", "所有命中 W KS", "共同窗口 W KS", "最差窗口 W KS", "足够样本窗口", "拒绝上穿"],
        [[STYLE[a['method']][0],f"{a['mean_max_transmittance_difference']:.5f}",f"{a['worst_max_transmittance_difference']:.5f}",
          f"{a['mean_all_hit_W_ks']:.5f}",f"{a['mean_common_window_W_ks']:.5f}" if a['mean_common_window_W_ks'] is not None else "n/a",
          f"{a['worst_window_W_ks']:.5f}" if a['worst_window_W_ks'] is not None else "n/a",
          f"{a['sufficient_windows']}/{nr*3}",a['rejected']] for a in aggregate])
    etable=html_table(["射线", "像素", "方法", "末端透射率", "95% Wilson 区间", "max|ΔT|", "W CDF 最大差", "接受命中数"],
        [[a['ray'],a['pixel'],STYLE[a['method']][0],f"{a['escape_probability']:.5f}",
          f"[{a['escape_ci95'][0]:.5f}, {a['escape_ci95'][1]:.5f}]",f"{a['max_transmittance_difference']:.5f}",
          f"{a['gradient_W_ks_all_hits']:.5f}",a['accepted_hits']] for a in metric_rows])
    ray_image = '<img src="figures/selected_rays.png">' if args.image else '<p>像素坐标见逐射线数据。</p>'
    best = min((a for a in aggregate if not a['method'].startswith('gpis_')),key=lambda a:a['mean_max_transmittance_difference'])
    focus_rows = [v for v in window_rows if v['ray']==focus['ray'] and v['window']==focus['window']]
    narrow = result['narrow_window_check']
    ftable = html_table(["方法", "原窗口命中数", "原窗口 W KS", "窄窗口命中数", "窄窗口 W KS"],
        [[STYLE[v['method']][0],v['hits'],f"{v['W_ks']:.4f}",q['hits'],f"{q['W_ks']:.4f}" if q['W_ks'] is not None else 'n/a']
         for v,q in zip(focus_rows,narrow['methods'])])
    doc=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>相机首段透射率与梯度分布对比</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1400px;margin:30px auto;padding:0 22px;color:#182536}}h1,h2{{line-height:1.3}}img{{max-width:100%}}table{{border-collapse:collapse;font-size:13px;width:100%;margin:20px 0}}th,td{{padding:7px;border:1px solid #d7dfe8;text-align:right}}th{{background:#eef3f8}}summary{{cursor:pointer;font-weight:bold;padding:14px;background:#eef3f8;margin-top:15px}}.note{{background:#f4f7fa;padding:16px}}a{{color:#0969a9}}</style>
<h1>相机首段：local / global_conditional / GPIS / renewal+</h1>
<p>Shader ball，σ={summary['sigma']:.8f}，ell≈{summary['rays'][0]['ell']:.8f}。{nr} 条固定相机射线，每种方法每条 {n:,} 次，共 {nr*len(methods)*n:,} 次首段采样。距离 s 从 activeDomain 入口起算，无反射或后续段。相机与 VDB 使用当前 local → GPIS 渲染配置。</p>
{interactive_section}
<details id="static-results"><summary>静态导出与完整统计（不受曲线开关影响）</summary>
<div class="note">GPIS 使用生产代码 gpis_function_space_tungsten，8 点 / batch，renewal+，gp_discrete，conditioned_gaussian，skip_space=0。比较 step_size=1、0.1、0.01；renewal_plus 比较 renewal_distance=1、0.1、0.01，均为世界距离。前者控制批次跨度 / 采样间距，后者控制重新抽取条件状态的距离；含义不同。GPIS 批次跨度为 8×min(step_size, 剩余长度/8)，故本场景 step_size=1 实际用一批覆盖整段，而非相邻点间隔 1。local 与 global_conditional 没有该步长参数。</div>
<p>GPIS .01 只是当前渲染配置下的离散参考，未声称其已收敛或等于精确 GP 真值。GPIS 初始批次未施加 F(entry)&gt;0 条件，而 GC / renewal+ 显式施加；本次入口均值、负初始符号数见原始元数据。GPIS 用割线纵向梯度及浮点采样，local / GC 使用各自的解析碰撞梯度律，这也是本实验比较的实际实现差异。</p>
<p>相同射线 / 不同方法使用相同初始随机种子，但消费方式不同，不构成逐样本配对。单条透射率 ECDF 的 95% DKW 界 ±{eps:.5f}；一对曲线差的保守 95% 界为 ±{result['dkw95_pair_difference_bound']:.5f}，不含离散偏差，也不是所有射线同时的置信保证。以下静态汇总的差值对 GPIS .01 计算，参考自身的零差值是定义值；上方交互区使用所选 local / GC / plus 方法之间的直接两两差异。</p>
<h2>等权汇总（只代表所选射线）</h2><a href="figures/comparison_summary.svg"><img src="figures/comparison_summary.png"></a>{atable}
<p><strong>本次透射率平均差最小的非 GPIS 方法是 {STYLE[best['method']][0]}</strong>，平均曲线最大差为 {100*best['mean_max_transmittance_difference']:.2f} 个百分点。但透射率更接近不保证碰撞梯度也逐处接近。GC renewal+ 的步长还改变了端点正值条件与丢弃旧观测的近似，不能把缩短更新距离当作普通积分的精度收敛。</p>
<p>max|ΔT| 是整条首段曲线的最大差；W = −direction·g × ell/σ。所有命中的梯度分布混合了碰撞位置的差异。固定距离窗口由独立预采样的 GPIS .01 命中距离 10%、50%、90% 分位点确定，半宽不超过 ell/4；各方法使用完全相同的区间。窗口仍有有限宽度，不能视为严格同一空间点的条件分布。每个方法及参考均至少 512 个命中才进入各自窗口汇总，不足者在明细标注 LOW N。上表共同窗口 W KS 仅在所有方法均达标的 {len(common_windows)} 个窗口上平均。</p>
<p>检测到的上穿也作为首交叉事件使透射率终止，单独报告为拒绝；不伪装成逃逸。梯度 CDF 仅使用被渲染器接受的下穿首次碰撞。无效梯度必须为零，才生成本报告。</p>
<h2>射线位置</h2>{ray_image}
<h2>透射率</h2><p>左：完整介质段；右：碰撞区域放大。绿色淡带：参考曲线的 DKW 界。</p><a href="figures/transmittance.svg"><img src="figures/transmittance.png"></a>
<h2>所有首次碰撞的梯度分布</h2><a href="figures/gradient_all_hits.svg"><img src="figures/gradient_all_hits.png"></a>
<h2>按相同碰撞距离窗口比较梯度</h2>{''.join(detail_sections)}
<h2>平均值掩盖的局部梯度差异</h2><p>renewal+ .01 差异最大的有效窗口是射线 {focus['ray']}（像素 {focus_ray['pixel']}）的 {focus['window']} 窗口 s∈[{focus['low']:.6f}, {focus['high']:.6f})。将半宽缩至四分之一，在 s∈[{narrow['low']:.6f}, {narrow['high']:.6f}) 内复核，避免只根据较宽距离区间的混合分布下结论。此为同一批主样本的描述性复核，并非独立验证，窄窗也不是精确固定点条件分布。</p>{ftable}
<h2>均值剖面与窗口位置</h2><a href="figures/mean_profiles.svg"><img src="figures/mean_profiles.png"></a>
<h2>逐射线数据</h2>{etable}
<h2>校验及复现</h2><p>各 CSV 的样本编号、状态计数、有限性、距离范围已校验。若整段长度 ≤1，GC 与 renewal+ 距离 1 的原始样本必须逐项相同。local 的蒙特卡洛透射率另与稠密剖面积分交叉核对（积分仅作数值校验）。细节见 <a href="metrics.json">metrics.json</a>。</p>
<p>采样耗时 {summary['sampling_wall_seconds']:.1f}s（{summary['thread_count']} 个并行任务，含 CSV 写入；不是公平的串行性能基准）。原始样本：<a href="{folder.name}/summary.json">{folder.name}/summary.json</a>、各 ray_* CSV；<a href="transmittance_gradient_metrics.csv">曲线与梯度指标</a>；<a href="gradient_window_metrics.csv">窗口指标</a>；<a href="distance_windows.json">窗口定义</a>；<a href="provenance.json">输入及程序 SHA-256</a>。</p>
<pre>cmake --build build --config Release --target macrofacet_compare_camera_first_hit --parallel 4
.\\build\\Release\\macrofacet_compare_camera_first_hit.exe configs/compare_camera_first_hit_shader_ball_sigma002.json
python tools/analyze_camera_first_hit.py --input outputs/camera_first_hit_comparison_20261017/main --pilot outputs/camera_first_hit_comparison_20261017/pilot --image outputs/render_first_hit_local_gpis_shader_ball_sigma002/render_hybrid_first_hit_directional.bmp</pre>
<p>复跑时将比较配置中的 output_directory 改为新目录，避免覆盖已有样本。报告绘图只对显示点抽稀，所有指标计算使用全部样本。</p></details></html>"""
    (out/"summary.html").write_text(doc,encoding="utf-8")
    print(json.dumps({"aggregate":aggregate,"validation":validation,"report":str(out/"summary.html")},indent=2))


if __name__ == "__main__":
    main()
