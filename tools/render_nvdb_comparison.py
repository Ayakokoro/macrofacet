"""Render one NanoVDB scene with classic local/global, conditional and joint neural transport.

Uses only the Python standard library. All relative paths (including paths in
the source config) are relative to the repository root, as with the C++ CLI.
"""
from __future__ import annotations

import argparse
import codecs
import copy
import csv
import html
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
MODES = {
    "classic_local": "classic_local",
    "classic_global": "classic_global",
    "global_conditional": "global_conditional",
    "neural": "neural_causal_renewal",
}
ALIASES = {
    "class_local": "classic_local",
    "class_global": "classic_global",
    "neural_causal_renewal": "neural",
}


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def mode_name(value: str) -> str:
    value = ALIASES.get(value, value)
    if value not in MODES:
        raise argparse.ArgumentTypeError("choose from " + ", ".join(MODES))
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True, type=Path, help="base NanoVDB render JSON")
    result.add_argument("--nvdb", type=Path, help="override field.grid_file for every mode")
    result.add_argument("--model", type=Path, help="joint export-joint JSON; otherwise use the base config")
    result.add_argument("--sigma", type=positive_float,
                        help="field sigma, required for NVDBs without stored sigma; must match stored sigma")
    result.add_argument("--modes", nargs="+", type=mode_name, default=list(MODES),
                        help="default: classic_local classic_global global_conditional neural; class_* aliases accepted")
    result.add_argument("--width", type=positive_int, help="otherwise inherit base config")
    result.add_argument("--height", type=positive_int, help="otherwise inherit base config")
    result.add_argument("--spp", type=positive_int, help="otherwise inherit base config")
    result.add_argument("--threads", type=nonnegative_int, help="CPU threads; 0=automatic; does not control CUDA")
    result.add_argument("--seed", type=nonnegative_int, help="shared seed; CPU/GPU RNG streams still differ")
    result.add_argument("--classic-phase-proposal", choices=("uniform", "paper_mixture", "target_vndf"),
                        help="classic normal sampling: inherit explicit config; otherwise target_vndf for Gaussian NDFs")
    result.add_argument("--neural-backend", choices=("scalar", "cuda_wavefront"),
                        help="otherwise inherit base config; scalar defaults to cumulative_inversion")
    result.add_argument("--neural-sampler", choices=("cumulative_inversion", "segmented_delta_tracking"),
                        help="otherwise inherit base config; delta tracking requires cuda_wavefront")
    result.add_argument("--exe", type=Path, help="renderer executable; auto-detect under build/ by default")
    result.add_argument("--output", type=Path, help="default: outputs/compare_<NVDB filename stem>")
    result.add_argument("--dry-run", action="store_true", help="write configs and commands without rendering")
    return result


def absolute(path: str | Path) -> Path:
    return (ROOT / Path(path).expanduser()).resolve()


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise ValueError(f"{description} does not exist: {path}")
    return path


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return data


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def executable(args: argparse.Namespace) -> Path:
    name = "macrofacet_experiments.exe" if os.name == "nt" else "macrofacet_experiments"
    candidates = [absolute(args.exe)] if args.exe else [
        ROOT / "build" / "Release" / name, ROOT / "build" / name,
        ROOT / "build" / "RelWithDebInfo" / name,
    ]
    selected = next((path for path in candidates if path.is_file()), candidates[0])
    if selected.suffix.lower() in (".cmd", ".bat"):
        raise ValueError("--exe must point to the renderer binary, not its .cmd/.bat launcher")
    if not args.dry_run:
        require_file(selected, "Renderer (build macrofacet_experiments or pass --exe)")
    return selected


def prepare(args: argparse.Namespace) -> tuple[dict, Path, list[dict]]:
    source_path = require_file(absolute(args.config), "Base config")
    base = read_json(source_path)
    for key in ("field", "material", "transport", "numeric", "render"):
        if not isinstance(base.get(key), dict):
            raise ValueError(f"base config requires an object: {key}")
    field = base["field"]
    if field.get("mean_type") != "nanovdb":
        raise ValueError("base config must use field.mean_type=nanovdb")
    grid = args.nvdb or field.get("grid_file")
    if not grid:
        raise ValueError("provide --nvdb or field.grid_file in the base config")
    nvdb = require_file(absolute(grid), "NanoVDB input")
    field["grid_file"] = nvdb.as_posix()
    if args.sigma is not None:
        field["sigma"] = args.sigma
    modes = list(dict.fromkeys(args.modes))
    if any(mode != "classic_local" for mode in modes):
        if field.get("use_alpha_grid", True):
            raise ValueError("global/neural comparison requires field.use_alpha_grid=false")
        if base["material"].get("ndf_family", field.get("ndf_family")) != "generalized_gaussian":
            raise ValueError("global/neural comparison requires ndf_family=generalized_gaussian")
    if any(mode in ("global_conditional", "neural") for mode in modes):
        if field.get("kernel_type", "squared_exponential") != "squared_exponential":
            raise ValueError("global_conditional and the joint neural model require squared_exponential")
    for option, key in (("width", "width"), ("height", "height"),
                        ("spp", "samples_per_pixel"), ("threads", "thread_count")):
        if getattr(args, option) is not None:
            base["render"][key] = getattr(args, option)
    if args.seed is not None:
        if args.seed >= 1 << 64:
            raise ValueError("seed must fit in uint64")
        base["seed"] = args.seed

    renewal = copy.deepcopy(base["transport"].get("renewal", {}))
    if "neural" in modes:
        model = args.model or renewal.get("model")
        if not model:
            raise ValueError("neural requires --model or transport.renewal.model (export-joint JSON)")
        model_path = require_file(absolute(model), "Joint neural model")
        if read_json(model_path).get("format") != "macrofacet.causal_renewal":
            raise ValueError("neural expects an export-joint bundle with format=macrofacet.causal_renewal")
        renewal["model"] = model_path.as_posix()
        if args.neural_backend:
            renewal["backend"] = args.neural_backend
            if args.neural_backend == "scalar" and args.neural_sampler is None:
                renewal["sampler"] = "cumulative_inversion"
        if args.neural_sampler:
            renewal["sampler"] = args.neural_sampler
        backend = renewal.setdefault("backend", "scalar")
        sampler = renewal.setdefault("sampler", "cumulative_inversion")
        if backend not in ("scalar", "cuda_wavefront"):
            raise ValueError("joint neural backend must be scalar or cuda_wavefront")
        if sampler not in ("cumulative_inversion", "segmented_delta_tracking"):
            raise ValueError(f"unknown neural sampler: {sampler}")
        if sampler == "segmented_delta_tracking" and backend != "cuda_wavefront":
            raise ValueError("segmented_delta_tracking requires --neural-backend cuda_wavefront")

    exe = executable(args)
    output = absolute(args.output or Path("outputs") / f"compare_{nvdb.stem}")
    jobs = []
    configs = []
    for label in modes:
        config = copy.deepcopy(base)
        # Mode-specific blocks are validated even when unused by the renderer.
        for key in ("renewal", "first_hit", "continuation", "global_conditional_renewal_plus"):
            config["transport"].pop(key, None)
        config["transport"]["mode"] = MODES[label]
        config["material"]["gp_model"] = "local_tangent" if label == "classic_local" else "global_pointwise"
        if label in ("classic_local", "classic_global"):
            # Neural configs often omit this classic-only option. The C++
            # default (uniform normals) has large importance weights, whereas
            # target_vndf samples the same Gaussian scattering law directly.
            family = config["material"].get("ndf_family", field.get("ndf_family"))
            default_proposal = "uniform" if family == "ggx" else "target_vndf"
            config["transport"]["classic_phase_proposal"] = (
                args.classic_phase_proposal or
                config["transport"].get("classic_phase_proposal", default_proposal))
        if label == "neural":
            config["transport"]["renewal"] = copy.deepcopy(renewal)
        directory = output / label
        config["output_directory"] = directory.as_posix()
        config_path = output / "configs" / f"{label}.json"
        if config_path == source_path:
            raise ValueError("output config would overwrite the source; choose a different --output")
        command = [str(exe), "render", "--config", str(config_path)]
        jobs.append({"label": label, "mode": MODES[label], "status": "planned",
                     "config": str(config_path), "output": str(directory), "command": command})
        configs.append(config)
    manifest = {"source_config": str(source_path), "working_directory": str(ROOT),
                "nvdb": str(nvdb), "seed": base.get("seed", 17429),
                "render": base["render"], "dry_run": args.dry_run, "jobs": jobs}
    return manifest, output, configs


def run_logged(command: list[str], log_path: Path) -> None:
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert process.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while chunk := process.stdout.read1(4096):
                log.write(chunk)
                log.flush()
                sys.stdout.write(decoder.decode(chunk))
                sys.stdout.flush()
            sys.stdout.write(decoder.decode(b"", final=True))
            if process.wait() != 0:
                raise RuntimeError(f"renderer exited with code {process.returncode}; see {log_path}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()


def collect_result(job: dict, config: dict, wall_seconds: float) -> dict:
    directory = Path(job["output"])
    summary = require_file(directory / "render_summary.csv", "Renderer summary")
    with summary.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1 or rows[0].get("mode") != job["mode"]:
        raise ValueError(f"unexpected render mode/row count in {summary}")
    row = rows[0]
    for column, key in (("width", "width"), ("height", "height"), ("spp", "samples_per_pixel")):
        if int(row[column]) != config["render"][key]:
            raise ValueError(f"renderer {column} differs from the requested config: {summary}")
    if int(row["numerical_failures"]) != 0:
        raise ValueError(f"renderer reported numerical failures: {summary}")
    if not read_json(directory / "run_summary.json").get("success"):
        raise ValueError(f"renderer did not complete successfully: {directory}")
    suffix = "white" if row["environment"] == "unit_white" else "directional"
    stem = f"render_{job['mode']}_{suffix}"
    for extension in ("pfm", "bmp"):
        path = require_file(directory / f"{stem}.{extension}", "Rendered image")
        row[extension] = f"{job['label']}/{path.name}"
    row.update(label=job["label"], wall_seconds=wall_seconds,
               config=f"configs/{job['label']}.json",
               classic_phase_proposal=config["transport"].get("classic_phase_proposal", "")
               if job["label"] in ("classic_local", "classic_global") else "",
               sampler=config["transport"].get("renewal", {}).get("sampler", ""))
    return row


def write_results(output: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(["label", "mode", "seconds", "wall_seconds", "pfm", "bmp"] +
                                [key for row in rows for key in row]))
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    cards = []
    for row in rows:
        e = lambda key: html.escape(str(row[key]), quote=True)
        cards.append(f'<figure><figcaption><b>{e("label")}</b> &middot; '
                     f'{float(row["seconds"]):.3f} s &middot; {e("width")} &times; '
                     f'{e("height")} &middot; {e("spp")} spp</figcaption>'
                     f'<img src="{e("bmp")}" alt="{e("label")}">'
                     f'<p><a href="{e("pfm")}">Linear PFM</a> &middot; '
                     f'<a href="{e("config")}">Config</a></p></figure>')
    page = ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>NanoVDB render comparison</title><style>'
            'body{font:16px system-ui;margin:24px;background:#191b20;color:#eee}'
            'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px}'
            'figure{margin:0}img{width:100%;margin-top:12px}a{color:#9acbff}'
            '</style><h1>NanoVDB render comparison</h1>'
            '<p>Shared scene settings. CPU and GPU use different random streams. '
            'Timings are renderer seconds; comparison.csv also includes process wall time.</p>'
            '<p><a href="comparison.csv">Statistics CSV</a> &middot; '
            '<a href="comparison.json">Run status and commands</a></p><main>' +
            "\n".join(cards) + '</main></html>\n')
    (output / "comparison.html").write_text(page, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        manifest, output, configs = prepare(args)
        (output / "configs").mkdir(parents=True, exist_ok=True)
        manifest_path = output / "comparison.json"
        write_json(manifest_path, manifest)
        for job, config in zip(manifest["jobs"], configs):
            write_json(Path(job["config"]), config)
            display = subprocess.list2cmdline(job["command"]) if os.name == "nt" else shlex.join(job["command"])
            print(f"{job['label']}: {display}", flush=True)
        print(f"Shared NVDB: {manifest['nvdb']}", flush=True)
        if args.dry_run:
            print(f"Configs written; no renders started: {output}")
            return 0
        rows = []
        write_results(output, rows)
        for index, (job, config) in enumerate(zip(manifest["jobs"], configs), 1):
            directory = Path(job["output"])
            directory.mkdir(parents=True, exist_ok=True)
            # A failed rerun must not consume an older successful summary.
            for name in ("render_summary.csv", "run_summary.json"):
                (directory / name).unlink(missing_ok=True)
            job["status"] = "running"
            write_json(manifest_path, manifest)
            print(f"\n[{index}/{len(configs)}] Rendering {job['label']} ...", flush=True)
            start = time.perf_counter()
            try:
                run_logged(job["command"], directory / "render.log")
                row = collect_result(job, config, time.perf_counter() - start)
            except (OSError, ValueError, RuntimeError, KeyError, KeyboardInterrupt) as error:
                job.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                           error=str(error), wall_seconds=time.perf_counter() - start)
                write_json(manifest_path, manifest)
                raise
            job.update(status="completed", statistics=row)
            rows.append(row)
            write_results(output, rows)
            write_json(manifest_path, manifest)
            print(f"Completed {job['label']}: {float(row['seconds']):.3f} render seconds", flush=True)
        print(f"\nComparison: {output / 'comparison.html'}")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted; completed renders have been retained.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
