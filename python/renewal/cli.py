from __future__ import annotations

import argparse
import json
from pathlib import Path

from .collect import collect
from .data import prepare
from .evaluate import evaluate
from .train import train
from .audit import audit
from .export import export_hazard, export_model
from .verify_cpp import verify_cpp


def main():
    parser = argparse.ArgumentParser(description="Renewal+ sequence data and neural likelihood training")
    commands = parser.add_subparsers(dest="command", required=True)
    collector = commands.add_parser("collect")
    collector.add_argument("--config", type=Path, required=True)
    packer = commands.add_parser("prepare")
    packer.add_argument("--manifest", type=Path, required=True)
    packer.add_argument("--output", type=Path, required=True)
    trainer = commands.add_parser("train")
    trainer.add_argument("--config", type=Path, required=True)
    trainer.add_argument("--resume", action="store_true")
    auditor = commands.add_parser("audit-reference")
    auditor.add_argument("--manifest", type=Path, required=True)
    auditor.add_argument("--output", type=Path, required=True)
    exporter = commands.add_parser("export-hazard")
    exporter.add_argument("--checkpoint", type=Path, required=True)
    exporter.add_argument("--output", type=Path, required=True)
    full_exporter = commands.add_parser("export-model")
    full_exporter.add_argument("--checkpoint", type=Path, required=True)
    full_exporter.add_argument("--output", type=Path, required=True)
    verifier = commands.add_parser("verify-cpp")
    verifier.add_argument("--checkpoint", type=Path, required=True)
    verifier.add_argument("--executable", type=Path, required=True)
    verifier.add_argument("--config", type=Path, required=True)
    verifier.add_argument("--output", type=Path, required=True)
    verifier.add_argument("--trials", type=int, default=16384)
    verifier.add_argument("--limit-rays", type=int, default=4)
    evaluator = commands.add_parser("evaluate")
    evaluator.add_argument("--checkpoint", type=Path, required=True)
    evaluator.add_argument("--dataset", type=Path, required=True)
    evaluator.add_argument("--output", type=Path, required=True)
    evaluator.add_argument("--split", choices=("train", "validation", "test"), default="test")
    evaluator.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "collect":
        result = collect(args.config)
    elif args.command == "prepare":
        result = prepare(args.manifest, args.output)
    elif args.command == "train":
        result = train(args.config, args.resume)
    elif args.command == "audit-reference":
        result = audit(args.manifest, args.output)
    elif args.command == "export-hazard":
        result = export_hazard(args.checkpoint, args.output)
    elif args.command == "export-model":
        result = export_model(args.checkpoint, args.output)
    elif args.command == "verify-cpp":
        result = verify_cpp(args.checkpoint, args.executable, args.config, args.output, args.trials, args.limit_rays)
    else:
        result = evaluate(args.checkpoint, args.dataset, args.output, args.split, args.device)
    print(json.dumps(result, indent=2, default=str, allow_nan=False))
