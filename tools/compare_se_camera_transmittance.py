"""Run existing camera inspection/curves tools on selected SE camera pixels."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIXELS = [(x, y) for y in (144, 256, 368) for x in (144, 256, 368)] + [
    (72, 256), (440, 256), (256, 72), (256, 440), (32, 32), (480, 32), (480, 480)]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(command, log):
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run([str(v) for v in command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    return result.returncode, time.perf_counter() - started


def add_gp_references(out, trials, workers):
    selection = read(out / "selection.json")
    records = read(out / "results.json")
    exe = ROOT / "build/Release/macrofacet_experiments.exe"
    source = selection["render_config"]
    if digest(source["field"]["grid_file"]) != selection["field_sha256"]:
        raise RuntimeError("field changed since the transport comparison")
    settings = dict(trajectories_per_resolution=trials, maximum_coarse_step=1/64,
                    maximum_fine_step=1/128, threads_per_process=4,
                    executable_sha256=digest(exe), field_sha256=selection["field_sha256"],
                    selection_sha256=digest(out / "selection.json"),
                    rays={r["directory"]: digest(out / r["directory"] / "neural/ray.json")
                          for r in records if r["status"] == "evaluated"},
                    note=("Independent coarse/fine SE first-passage samples using the full VDB mean and exported start condition; numerical reference, not exact truth."
                          if selection.get("flight") == "after_first_hit" else
                          "Independent coarse/fine SE positive-exterior first-passage samples using the full VDB mean; numerical reference, not exact truth."))
    if (out / "gp_selection.json").exists() and read(out / "gp_selection.json") != settings:
        raise RuntimeError("GP reference inputs changed; use a new output directory")
    write(out / "gp_selection.json", settings)

    def evaluate(record):
        folder = out / record["directory"]
        if record["status"] != "evaluated":
            return dict(id=record["id"], pixel=record["pixel"], status=record["status"])
        gp = folder / "gp"
        gp.mkdir(exist_ok=True)
        meta = read(folder / "neural/ray.json")
        x, y = record["pixel"]
        result = dict(id=record["id"], pixel=record["pixel"], status="evaluated", resolutions={})
        maximum = meta["reference_maximum_x"]
        step = maximum / math.ceil(maximum * 64)
        for resolution, name in enumerate(("coarse", "fine")):
            config = read(folder / "neural" / f"reference_{name}.json")
            config["seed"] = source["seed"] + 20261010 + (y * meta["width"] + x) * 1000003 + resolution * 104729
            fp = config["first_passage"]
            fp["grid"]["step_sizes"] = [step / 2**resolution]
            fp["monte_carlo"].update(trajectories=trials, thread_count=4, write_raw_samples=False)
            config["output_directory"] = str(gp / name)
            request = gp / f"reference_{name}.json"
            if request.exists() and read(request) != config:
                raise RuntimeError(f"reference configuration changed: {request}")
            write(request, config)
            complete = gp / f"complete_{name}.json"
            if complete.exists():
                result["resolutions"][name] = read(complete)
                continue
            print(f"[GP {record['id']}/{len(records)}] ({x},{y}) {name}: {trials} trajectories", flush=True)
            code, elapsed = run([exe, "first-passage", "--config", request], gp / f"{name}.log")
            if code:
                raise RuntimeError((gp / f"{name}.log").read_text(encoding="utf-8", errors="replace"))
            result["resolutions"][name] = dict(seconds=elapsed, step=step / 2**resolution,
                                                trajectories=trials, config_sha256=digest(request))
            write(complete, result["resolutions"][name])
            print(f"[GP {record['id']}/{len(records)}] ({x},{y}) {name} completed in {elapsed:.1f}s", flush=True)
        return result

    results = []
    with ThreadPoolExecutor(max_workers=min(workers, 2)) as pool:
        for future in as_completed([pool.submit(evaluate, record) for record in records]):
            results.append(future.result())
            write(out / "gp_results.json", sorted(results, key=lambda r: r["id"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/render_neural_renewal_shader_ball_ply_point_linear_se_h16.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pixels", nargs="+", help="zero-based x,y pairs; defaults to the existing 16-pixel selection")
    parser.add_argument("--trials", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preview-spp", type=int, default=4)
    parser.add_argument("--gp-trials", type=int, default=0, help="add coarse/fine SE GP first-passage reference; 65536 recommended")
    parser.add_argument("--after-first-hit", action="store_true", help="compare the reflected flight conditioned on the first neural hit")
    parser.add_argument("--camera-suite", type=Path, help="reuse the matching camera suite preview and link the two flight views")
    args = parser.parse_args()
    os.chdir(ROOT)
    source = read(args.config)
    pixels = [tuple(map(int, s.split(","))) for s in args.pixels] if args.pixels else DEFAULT_PIXELS
    if source["field"]["kernel_type"] != "squared_exponential" or source["transport"]["mode"] != "neural_renewal":
        parser.error("requires a squared_exponential neural render configuration")
    if min(args.trials, args.workers, args.preview_spp) < 1 or args.gp_trials < 0 or len(set(pixels)) != len(pixels):
        parser.error("budgets must be positive and pixels unique")
    if any(len(p) != 2 or not (0 <= p[0] < source["render"]["width"] and 0 <= p[1] < source["render"]["height"]) for p in pixels):
        parser.error("pixels must lie inside the configured image")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    exe = ROOT / "build/Release/macrofacet_experiments.exe"
    inspect = ROOT / "build/Release/macrofacet_inspect_renewal_ray.exe"
    selection = dict(render_config=source, pixels=pixels, trials_per_mode=args.trials,
                     confidence_level=.95, bins=4096, sample_index=0, preview_spp=args.preview_spp,
                     model_sha256=digest(source["transport"]["renewal"]["model"]),
                     field_sha256=digest(source["field"]["grid_file"]),
                     inspect_sha256=digest(inspect), curves_sha256=digest(exe),
                     note="Fixed selected pixels, sample 0. Pairwise method differences, not GP ground-truth errors.")
    selection = json.loads(json.dumps(selection))
    if args.after_first_hit:
        selection.update(flight="after_first_hit", note="Sample 0 first neural hit, then fixed F=0 and full gradient; no prior-path history. classic_local is an unconditioned baseline.")
    if args.camera_suite:
        args.camera_suite = args.camera_suite.resolve()
        camera_selection = read(args.camera_suite / "selection.json")
        if (not args.after_first_hit or args.camera_suite == out or
                camera_selection["render_config"] != source or camera_selection["pixels"] != selection["pixels"] or
                camera_selection["field_sha256"] != selection["field_sha256"] or
                camera_selection["model_sha256"] != selection["model_sha256"]):
            parser.error("camera suite must have the same field, model, render config and selected pixels")
    if (out / "selection.json").exists() and read(out / "selection.json") != selection:
        raise RuntimeError("inputs changed; choose a new output directory")
    write(out / "selection.json", selection)
    write(out / "render_config.json", source)

    def evaluate(number, pixel):
        x, y = pixel
        folder = out / f"pixel_{x}_{y}"
        folder.mkdir(exist_ok=True)
        if (folder / "complete.json").exists():
            return read(folder / "complete.json")
        print(f"[{number}/{len(pixels)}] ({x},{y}) inspecting camera ray", flush=True)
        record = dict(id=number, pixel=pixel, directory=folder.name, seconds={})
        command = [inspect, out / "render_config.json", folder / "neural", x, y]
        if args.after_first_hit:
            command.append("--after-first-hit")
        code, elapsed = run(command, folder / "inspect.log")
        record["seconds"]["inspect"] = elapsed
        if code:
            message = (folder / "inspect.log").read_text(encoding="utf-8", errors="replace")
            if "selected camera ray misses the active domain" in message:
                record["status"] = "domain_miss"
            elif args.after_first_hit and "selected camera sample escapes without a first hit" in message:
                record["status"] = "camera_escape"
            else:
                raise RuntimeError(message)
        else:
            meta = read(folder / "neural/ray.json")
            assert meta["kernel"] == "squared_exponential_unit_length" and meta["pixel"] == list(pixel)
            config = json.loads(json.dumps(source))
            config["transport"]["mode"] = "classic_local"
            config["material"]["gp_model"] = "local_tangent"
            config["output_directory"] = str(folder / "transport")
            config["transmittance"] = {
                "modes": ["classic_local", "global_conditional"],
                "curve": {"bins": selection["bins"], "spacing": "linear", "distance_origin": "ray_origin"},
                "monte_carlo": {"trials_per_ray": args.trials, "confidence_level": .95, "write_raw_samples": False},
                "conditional_birth": {"policy": "sample_positive_exterior"},
                "comparison": {"constant_exponential": {"enabled": False}},
                "rays": [{"id": folder.name, "origin": meta["entry_origin"], "direction": meta["direction"],
                          "max_distance": meta["reference_maximum_x"] * meta["ell"]}]}
            if args.after_first_hit:
                assert meta["start_condition"] == "collision_state"
                config["transmittance"]["conditional_birth"] = dict(
                    policy="fixed_observation", value=0., gradient=meta["source_hit"]["gradient"])
            write(folder / "transport_config.json", config)
            print(f"[{number}/{len(pixels)}] ({x},{y}) sampling {args.trials} trials per transport mode", flush=True)
            code, elapsed = run([exe, "curves", "--config", folder / "transport_config.json"], folder / "curves.log")
            if code:
                raise RuntimeError((folder / "curves.log").read_text(encoding="utf-8", errors="replace"))
            record["seconds"]["curves"] = elapsed
            record["status"] = "evaluated"
        write(folder / "complete.json", record)
        print(f"[{number}/{len(pixels)}] ({x},{y}) {record['status']}", flush=True)
        return record

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate, i, pixel) for i, pixel in enumerate(pixels, 1)]
        for future in as_completed(futures):
            records.append(future.result())
            write(out / "results.json", sorted(records, key=lambda r: r["id"]))

    if args.gp_trials:
        add_gp_references(out, args.gp_trials, args.workers)

    # Same camera and kernel: this is a location preview, not a transmittance image.
    preview = json.loads(json.dumps(source))
    preview["render"]["samples_per_pixel"] = args.preview_spp
    preview["output_directory"] = str(out / "preview")
    write(out / "preview_config.json", preview)
    if args.camera_suite and not (out / "preview/run_summary.json").exists():
        # The plotting helper validates the reused image's camera/field/model.
        (out / "preview").mkdir(exist_ok=True)
        for path in (args.camera_suite / "preview").glob("*.pfm"):
            shutil.copyfile(path, out / "preview" / path.name)
        shutil.copyfile(args.camera_suite / "preview_config.json", out / "preview_config.json")
        shutil.copyfile(args.camera_suite / "preview/run_summary.json", out / "preview/run_summary.json")
    if not (out / "preview/run_summary.json").exists():
        print("Rendering SE scene preview", flush=True)
        code, elapsed = run([exe, "render", "--config", out / "preview_config.json"], out / "preview.log")
        if code:
            raise RuntimeError((out / "preview.log").read_text(encoding="utf-8", errors="replace"))
        write(out / "preview_timing.json", {"seconds": elapsed})
    sys.path.insert(0, str(ROOT / "data_analysis"))
    from plot_transport_camera_rays import plot_suite
    if args.camera_suite:
        for root in (out, args.camera_suite):
            links = read(root / "view_links.json") if (root / "view_links.json").exists() else {}
            links.update(dict(
                camera=Path(os.path.relpath(args.camera_suite / "comparison.html", root)).as_posix(),
                surface=Path(os.path.relpath(out / "comparison.html", root)).as_posix()))
            write(root / "view_links.json", links)
    plot_suite(out)
    if args.camera_suite:
        plot_suite(args.camera_suite)
    print(f"Completed: {out / 'comparison.html'}", flush=True)


if __name__ == "__main__":
    main()
