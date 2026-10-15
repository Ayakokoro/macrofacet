"""Reproduce scalar RNO/GPIS first-hit renders with an identical renewal tail."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))
try:
    import matplotlib
except ModuleNotFoundError:
    dependency = ROOT / "outputs/.analysis_deps"
    if not (dependency / "matplotlib").exists():
        raise RuntimeError("Install matplotlib or provide outputs/.analysis_deps") from None
    sys.path.insert(0, str(dependency))
    import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def csv(path):
    with Path(path).open(encoding="utf-8") as f:
        names = f.readline().strip().split(",")
    integers = {"pixel_x", "pixel_y", "sample", "entered", "hit", "initial_sign"}
    # Seeds exceed 2**53; numpy's inferred float dtype would discard low bits.
    dtype = [(n, "u8" if n.endswith("_seed") else "i8" if n in integers else "f8") for n in names]
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=dtype, encoding="utf-8"))


def xyz(data, names):
    return np.column_stack([data[n] for n in names])


def ecdf(values, x):
    return np.searchsorted(np.sort(values), x, side="right") / len(values)


def gap(a, b):
    nodes = np.unique(np.concatenate([a, b]))
    return float(np.max(np.abs(ecdf(a, nodes) - ecdf(b, nodes))))


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= .0031308, 12.92*x, 1.055*x**(1/2.4)-.055)


def read_pfm(path):
    with Path(path).open("rb") as f:
        if f.readline().strip() != b"PF":
            raise ValueError("expected RGB PFM")
        width, height = map(int, f.readline().split())
        scale = float(f.readline())
        image = np.fromfile(f, dtype="<f4" if scale < 0 else ">f4").reshape(height,width,3)
    return image[::-1] * abs(scale)


def run(args, output):
    source = read_json(ROOT / args.config)
    export = read_json(ROOT / source["transport"]["first_hit"]["model"])
    sampler = source["transport"]["first_hit"]["sampler"]
    for key in ("history_points", "batch_size", "step_size", "rank_tolerance", "crossing_rule"):
        if sampler[key] != export["reference_sampler"][key]:
            raise ValueError(f"GPIS reference differs from model training: {key}")
    exe = ROOT / "build/Release/macrofacet_experiments.exe"
    probe = ROOT / "build/Release/macrofacet_compare_first_hit_render.exe"
    fixed_config = output / "fixed_ray_config.json"
    write_json(fixed_config, source)
    commands = []
    for label, method in (("rno", "neural_first_hit_se"), ("gpis", "gpis_function_space")):
        cfg = json.loads(json.dumps(source))
        cfg["transport"]["first_hit"]["method"] = method
        if label == "gpis":
            cfg["transport"]["first_hit"].pop("model", None)
        cfg["render"].update(width=args.width, height=args.height, samples_per_pixel=args.spp,
                              thread_count=args.threads, write_first_hit_samples=True)
        cfg["output_directory"] = str(output / label)
        path = output / f"{label}_config.json"
        write_json(path, cfg)
        commands.append((label, [str(exe), "render", "--config", str(path)]))
    commands.append(("fixed_rays", [str(probe), str(fixed_config), str(output / "fixed_rays"), str(args.ray_trials)]))
    provenance = {"source_config": str(args.config), "commands": commands, "sha256": {},
                  "sampling_driver_sha256": digest(Path(__file__))}
    files = [ROOT / args.config, exe, probe,
             ROOT / source["field"]["grid_file"],
             ROOT / source["transport"]["first_hit"]["model"],
             ROOT / source["transport"]["continuation"]["model"]]
    for path in files:
        provenance["sha256"][str(path)] = digest(path)
    write_json(output / "provenance.json", provenance)

    def execute(job):
        label, command = job
        print(f"Starting {label}", flush=True)
        with (output / f"{label}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Completed {label}", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(execute, commands))


def report(output):
    data = {label: csv(output / label / "first_hit_samples.csv") for label in ("rno", "gpis")}
    config = read_json(output / "rno_config.json")
    width, height, spp = (config["render"][k] for k in ("width", "height", "samples_per_pixel"))
    metrics = {"resolution": [width, height], "spp": spp, "methods": {}, "fixed_rays": []}
    for field in ("pixel_x", "pixel_y", "sample", "ray_dx", "ray_dy", "ray_dz", "entry_x", "entry_y", "entry_z",
                  "entry_distance", "maximum_distance", "entered", "camera_seed", "event_seed", "gradient_seed", "continuation_seed"):
        if not np.array_equal(data["rno"][field], data["gpis"][field]):
            raise AssertionError(f"paired render mismatch: {field}")
    metrics["identical_camera_rays_and_stage_seeds"] = True
    metric = np.array(read_json(output / "rno/resolved_config.json")["derived"]["kernel_metric"])
    sigma = read_json(output / "rno/resolved_config.json")["derived"]["sigma"]
    images, maps = {}, {}
    for label, d in data.items():
        selected = d["hit"].astype(bool)
        h = d[selected]
        w = xyz(h, ("ray_dx", "ray_dy", "ray_dz"))
        g = xyz(h, ("gx", "gy", "gz"))
        n = xyz(h, ("nx", "ny", "nz"))
        ell = 1/np.sqrt(np.einsum("ni,ij,nj->n", w, metric, w))
        derivative = -h["initial_sign"]*sigma/ell*h["speed"]
        checks = {
            "max_distance_unit_error": float(np.max(np.abs(h["distance"]-h["q"]*ell))),
            "max_position_error": float(np.max(np.linalg.norm(xyz(h,("x","y","z"))-xyz(h,("entry_x","entry_y","entry_z"))-h["distance"][:,None]*w,axis=1))),
            "max_gradient_projection_error": float(np.max(np.abs(np.sum(w*g,axis=1)-derivative))),
            "max_normal_unit_error": float(np.max(np.abs(np.linalg.norm(n,axis=1)-1))),
        }
        if max(checks.values()) > 1e-10 or np.any(h["speed"] <= 0):
            raise AssertionError(checks)
        summary = read_json(output / label / "hybrid_summary.json")
        if summary["numerical_failures"] or summary["safety_cap_terminations"]:
            raise AssertionError(f"incomplete paths: {summary}")
        summary.update(checks)
        summary["hit_probability_given_domain_entry"] = float(np.mean(d["hit"][d["entered"].astype(bool)]))
        metrics["methods"][label] = summary
        image = xyz(d,("radiance_r","radiance_g","radiance_b")).reshape(height,width,spp,3).mean(axis=2)
        environment_suffix = "white" if config["render"]["environment"] == "unit_white" else "directional"
        pfm = read_pfm(output / label / f"render_hybrid_first_hit_{environment_suffix}.pfm")
        summary["max_pfm_vs_sample_mean_error"] = float(np.max(np.abs(image-pfm)))
        if summary["max_pfm_vs_sample_mean_error"] > 1e-6:
            raise AssertionError("PFM differs from sampled radiance")
        images[label] = image
        hit = d["hit"].reshape(height,width,spp)
        counts = hit.sum(axis=2)
        distance = (d["distance"].reshape(height,width,spp)*hit).sum(axis=2)/np.maximum(counts,1)
        oriented_n = xyz(d,("nx","ny","nz"))*d["initial_sign"][:,None]
        normal = oriented_n.reshape(height,width,spp,3).sum(axis=2)/np.maximum(counts[...,None],1)
        normal = normal/np.maximum(np.linalg.norm(normal,axis=2,keepdims=True),1e-15)
        maps[label] = (counts/spp, np.where(counts>0,distance,np.nan), np.where(counts[...,None]>0,.5*(normal+1),1))
    differences = (xyz(data["rno"],("radiance_r","radiance_g","radiance_b"))-
                   xyz(data["gpis"],("radiance_r","radiance_g","radiance_b"))).reshape(height,width,spp,3)
    variance_of_pixel_difference = differences.var(axis=2,ddof=1)/spp
    pixel_difference = images["rno"]-images["gpis"]
    metrics["image"] = {
        "linear_rgb_rmse_finite_spp": float(np.sqrt(np.mean(pixel_difference**2))),
        "linear_rgb_mae_finite_spp": float(np.mean(np.abs(pixel_difference))),
        "estimated_rms_sampling_noise_in_difference": float(np.sqrt(np.mean(variance_of_pixel_difference))),
        "mean_rgb_difference": pixel_difference.mean(axis=(0,1)).tolist(),
        "mean_rgb_difference_standard_error": (np.sqrt(variance_of_pixel_difference.sum(axis=(0,1)))/(width*height)).tolist(),
    }
    fig, axes = plt.subplots(2,4,figsize=(14,7),layout="constrained")
    vmin = min(np.nanmin(maps[k][1]) for k in maps)
    vmax = max(np.nanmax(maps[k][1]) for k in maps)
    for row, label in enumerate(("rno","gpis")):
        axes[row,0].imshow(srgb(images[label])); axes[row,0].set_title(f"{label.upper()} first hit + renewal")
        im = axes[row,1].imshow(maps[label][0],vmin=0,vmax=1,cmap="viridis"); axes[row,1].set_title("First-hit probability")
        fig.colorbar(im,ax=axes[row,1],fraction=.045)
        im = axes[row,2].imshow(maps[label][1],vmin=vmin,vmax=vmax,cmap="magma"); axes[row,2].set_title("Mean hit distance from entry")
        fig.colorbar(im,ax=axes[row,2],fraction=.045)
        axes[row,3].imshow(maps[label][2]); axes[row,3].set_title("Mean facing normal")
        for ax in axes[row]: ax.axis("off")
    save_figure(fig,output,"render_comparison")
    fig, axes = plt.subplots(1,3,figsize=(12,4),layout="constrained")
    for ax,label in zip(axes,("rno","gpis")):
        ax.imshow(srgb(images[label])); ax.set_title(label.upper()); ax.axis("off")
    im=axes[2].imshow(np.mean(np.abs(pixel_difference),axis=2),cmap="inferno",vmin=0)
    axes[2].set_title("Absolute linear RGB difference\n(includes sampling noise)"); axes[2].axis("off")
    fig.colorbar(im,ax=axes[2],fraction=.045)
    save_figure(fig,output,"image_difference")

    fixed = read_json(output / "fixed_rays/summary.json")
    fig,axes=plt.subplots(len(fixed["rays"]),3,figsize=(14,10),layout="constrained")
    for row, ray in enumerate(fixed["rays"]):
        samples={label:csv(output/f"fixed_rays/ray_{ray['id']}_{label}.csv") for label in ("rno","gpis")}
        curves=csv(output/f"fixed_rays/ray_{ray['id']}_curve.csv")
        result={"pixel":ray["pixel"],"methods":{}}
        w=np.array(ray["direction"])
        speeds,normals={},{}
        for label,color in (("rno","#c54838"),("gpis","#257bb8")):
            d=samples[label]; h=d[d["hit"].astype(bool)]
            T=1-np.searchsorted(np.sort(h["distance"]),curves["distance"],side="right")/len(d)
            axes[row,0].plot(curves["distance"],T,color=color,label=f"{label.upper()} sampled")
            speeds[label]=h["speed"]
            normals[label]=np.abs(xyz(h,("nx","ny","nz"))@w)
            for ax,values in ((axes[row,1],speeds[label]),(axes[row,2],normals[label])):
                ordered=np.sort(values); ax.plot(ordered,np.arange(1,len(values)+1)/len(values),color=color,label=label.upper())
            result["methods"][label]={"trials":len(d),"hits":len(h),"hit_probability":len(h)/len(d),
                "sample_T_max_difference_from_network":float(np.max(np.abs(T-curves["transmittance"]))) }
        axes[row,0].plot(curves["distance"],curves["transmittance"],"k--",lw=1,label="RNO predicted T")
        result["joint_hit_speed_two_sample_cdf_gap"]=gap(speeds["rno"],speeds["gpis"])
        result["normal_cosine_two_sample_cdf_gap"]=gap(normals["rno"],normals["gpis"])
        metrics["fixed_rays"].append(result)
        for col,title in enumerate(("Transmittance","Hit speed W CDF (joint marginal)","Normal |omega dot n| CDF")):
            axes[row,col].set_title(f"Pixel {tuple(ray['pixel'])}: {title}")
            axes[row,col].grid(alpha=.2); axes[row,col].legend(fontsize=8); axes[row,col].set_ylim(-.02,1.02)
        axes[row,0].set_xlabel("Physical distance from medium entry")
        axes[row,1].set_xlabel("Positive normalized crossing speed W")
        axes[row,2].set_xlabel("Absolute incident cosine")
    save_figure(fig,output,"fixed_ray_comparison")
    write_json(output/"metrics.json",metrics)
    provenance = read_json(output/"provenance.json")
    if str(Path(__file__)) in provenance["sha256"]:
        provenance["sampling_driver_sha256"] = provenance["sha256"].pop(str(Path(__file__)))
    provenance["report_script_sha256"] = digest(Path(__file__))
    write_json(output/"provenance.json",provenance)
    rows="".join(f"<tr><td>{label.upper()}</td><td>{m['first_hits']:,}</td><td>{m['hit_probability_given_domain_entry']:.6f}</td><td>{m['upcrossings']}</td><td>{m['seconds']:.2f}</td></tr>" for label,m in metrics["methods"].items())
    document=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>First-hit hybrid render comparison</title>
<style>body{{max-width:1280px;margin:40px auto;padding:0 24px;font:16px/1.65 system-ui;color:#23303f;background:#f7f9fc}}img{{width:100%;background:white;border-radius:8px}}table{{border-collapse:collapse;background:white}}td,th{{padding:9px 20px;border:1px solid #dce2eb}}code{{overflow-wrap:anywhere}}a{{color:#145bab}}</style>
<h1>RNO / GPIS 首次碰撞 + 相同 neural renewal 后续传播</h1>
<p>{width} × {height}，每像素 {spp} 个样本。sigma={sigma:.6g}，场文件 <code>{config['field']['grid_file']}</code>。两种方法使用完全相同的相机射线与分阶段随机种子；后续均为标量 SE renewal、{config['transport']['continuation']['profile_mode']} 剖面。</p>
<p>首次距离从 activeDomain 入口开始。RNO 与 GPIS 首次段均使用 cubic 剖面。参考参数：<code>{json.dumps(config['transport']['first_hit']['sampler'])}</code>。保留上穿零点，均值场和完整梯度一起按路径所在侧变换。</p>
<table><tr><th>首次段</th><th>命中数</th><th>进入域后的命中率</th><th>上穿命中</th><th>渲染秒数</th></tr>{rows}</table>
<p>接入检查通过：相机与四组随机种子逐样本一致；距离单位、位置重建、梯度投影及单位法线误差均小于 1e-10；无数值失败或深度上限截断。</p>
<img src="render_comparison.png" alt="render, hit probability, distance and facing normal maps">
<p>线性 RGB 图像 RMSE：{metrics['image']['linear_rgb_rmse_finite_spp']:.6f}；差值中的估计 RMS 采样噪声：{metrics['image']['estimated_rms_sampling_noise_in_difference']:.6f}。有限 spp 图像差异包含模型差异与 Monte Carlo 噪声，不能全部解释为模型偏差。</p>
<img src="image_difference.png" alt="finite sample image difference">
<p>下面三条固定射线使用渲染器同一个首次段适配器重复独立采样。速度与法线 CDF 是命中样本的联合分布边缘量，同时受到距离分布和梯度分布的影响。GPIS 的完整梯度同样由纵向割线速度和解析 SE 横向条件高斯重建，并非独立采得的完整三维实现场。</p>
<img src="fixed_ray_comparison.png" alt="fixed ray transmittance and gradient distributions">
<p><a href="metrics.json">完整指标</a> · <a href="provenance.json">命令与文件哈希</a> · <a href="rno_config.json">RNO 运行配置</a> · <a href="gpis_config.json">GPIS 运行配置</a> · <a href="rno/first_hit_samples.csv">RNO 逐样本数据</a> · <a href="gpis/first_hit_samples.csv">GPIS 逐样本数据</a></p>
<p>两个阶段组成混合模型，后续 renewal 仅以当前碰撞位置和完整梯度初始化，未保留相机路径的完整 GP 历史。这里的 GPIS 对照只替换首次段。</p></html>"""
    (output/"summary.html").write_text(document,encoding="utf-8")
    print(output/"summary.html",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/render_first_hit_rno_renewal_shader_ball_sigma005.json")
    parser.add_argument("--output",default="outputs/first_hit_hybrid_render_20261014")
    parser.add_argument("--width",type=int,default=64); parser.add_argument("--height",type=int,default=64)
    parser.add_argument("--spp",type=int,default=64); parser.add_argument("--threads",type=int,default=8)
    parser.add_argument("--ray-trials",type=int,default=4096)
    parser.add_argument("--report-only",action="store_true")
    args=parser.parse_args()
    if args.spp < 2 or min(args.width,args.height,args.threads,args.ray_trials) < 1:
        parser.error("positive dimensions/threads/trials and spp >= 2 required")
    output=(ROOT/args.output).resolve(); output.mkdir(parents=True,exist_ok=True)
    if not args.report_only: run(args,output)
    report(output)


if __name__ == "__main__":
    main()
