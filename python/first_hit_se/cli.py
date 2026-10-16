import argparse
import json
from pathlib import Path

from .data import collect, prepare
from .train import train, export_model
from .evaluate import compare
from .verify_cpp import verify_cpp
from .audit import audit


def main():
    parser = argparse.ArgumentParser(description="Stationary-SE first-hit data, neural models, and controlled comparisons")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "train", "spline-train"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
        if name == "train":
            sub.add_argument("--resume", action="store_true")
    sub = commands.add_parser("prepare")
    sub.add_argument("--manifest", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub = commands.add_parser("export-model")
    sub.add_argument("--checkpoint", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub = commands.add_parser("compare")
    sub.add_argument("--checkpoint", type=Path, action="append", required=True)
    sub.add_argument("--dataset", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--split", choices=("train", "validation", "test"), default="test")
    sub.add_argument("--device", default="cpu")
    sub.add_argument("--points", type=int, default=129)
    sub.add_argument("--subdivisions", type=int, nargs="+", default=[1,2,4,8])
    sub.add_argument("--limit", type=int)
    sub = commands.add_parser("verify-cpp")
    sub.add_argument("--checkpoint", type=Path, required=True)
    sub.add_argument("--dataset", type=Path, required=True)
    sub.add_argument("--executable", type=Path, default=Path("build/Release/macrofacet_query_first_hit_se.exe"))
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--limit", type=int, default=3)
    sub.add_argument("--trials", type=int, default=512)
    sub = commands.add_parser("audit-reference")
    sub.add_argument("--manifest", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub = commands.add_parser('ablation')
    sub.add_argument('--config', type=Path, required=True)
    sub.add_argument('--prepare-only', action='store_true')
    sub = commands.add_parser('ablation-evaluate')
    sub.add_argument('--config', type=Path, required=True)
    sub.add_argument('--watch', action='store_true')
    sub = commands.add_parser('ablation-report')
    sub.add_argument('--config', type=Path, required=True)
    sub = commands.add_parser('ablation-audit')
    sub.add_argument('--config', type=Path, required=True)
    sub.add_argument('--watch', action='store_true')
    for name in ('gradient-head', 'gradient-head-report'):
        sub = commands.add_parser(name)
        sub.add_argument('--config', type=Path, required=True)
    sub = commands.add_parser('data-ablation')
    sub.add_argument('--config', type=Path, required=True)
    sub.add_argument('--phase', choices=('all', 'collect', 'train', 'report'), default='all')
    sub = commands.add_parser('joint-state')
    sub.add_argument('--config', type=Path, required=True)
    sub.add_argument('--phase', choices=('all','prepare','train','evaluate','report','audit'), default='all')
    sub = commands.add_parser('joint-state-train')
    sub.add_argument('--config', type=Path, required=True)
    sub.add_argument('--name', required=True)
    sub = commands.add_parser('single-ray')
    sub.add_argument('--config', type=Path, required=True)
    sub.add_argument('--phase', choices=('all', 'collect', 'train', 'evaluate', 'report'), default='all')
    sub = commands.add_parser('camera-study')
    sub.add_argument('--config', type=Path, required=True)
    sub.add_argument('--phase', choices=('all', 'collect', 'train', 'report'), default='all')
    sub = commands.add_parser('spline-study')
    sub.add_argument('--phase', choices=('all', 'evaluate', 'benchmark', 'report'), default='all')
    sub = commands.add_parser('hazard-study')
    sub.add_argument('--config', type=Path, required=True)
    sub.add_argument('--phase', choices=('all', 'train', 'evaluate', 'benchmark', 'report'), default='all')
    sub = commands.add_parser('distance-study')
    sub.add_argument('--config', type=Path, required=True)
    sub.add_argument('--phase', choices=('all', 'collect', 'audit', 'train', 'evaluate', 'probe',
                                        'benchmark', 'report'), default='all')
    sub = commands.add_parser('camera-holdout')
    sub.add_argument('--config', type=Path, required=True)
    sub.add_argument('--phase', choices=('all', 'select', 'collect', 'evaluate', 'report'), default='all')
    sub = commands.add_parser('multiview-data')
    sub.add_argument('--config', type=Path, required=True)
    sub.add_argument('--phase', choices=('all', 'select', 'collect'), default='all')
    sub = commands.add_parser('multiview-study')
    sub.add_argument('--config', type=Path, required=True)
    sub.add_argument('--phase', choices=('all', 'distance', 'evaluate', 'gradient', 'sampling', 'report'), default='all')
    sub = commands.add_parser('residual-supervision')
    sub.add_argument('--config', type=Path, required=True)
    sub.add_argument('--phase', choices=('all', 'train', 'evaluate', 'report'), default='all')
    sub = commands.add_parser('causal-residual')
    sub.add_argument('--config', type=Path, required=True)
    sub.add_argument('--phase', choices=('all', 'train', 'evaluate', 'diagnose', 'report'), default='all')
    sub = commands.add_parser('causal-gradient')
    sub.add_argument('--config', type=Path, required=True)
    sub.add_argument('--phase', choices=('all', 'train', 'evaluate', 'diagnose', 'report'), default='all')
    sub = commands.add_parser('causal-gradient-step')
    sub.add_argument('--config', type=Path, required=True)
    sub.add_argument('--phase', choices=('all', 'evaluate', 'report'), default='all')
    args = parser.parse_args()
    if args.command == "collect": result = collect(args.config)
    elif args.command == "prepare": result = prepare(args.manifest, args.output)
    elif args.command == "train": result = train(args.config, args.resume)
    elif args.command == "spline-train":
        from .spline_train import train as spline_train
        result = spline_train(args.config)
    elif args.command == "export-model": result = export_model(args.checkpoint, args.output)
    elif args.command == "verify-cpp": result = verify_cpp(args.checkpoint,args.dataset,args.executable,args.output,args.limit,args.trials)
    elif args.command == "audit-reference": result = audit(args.manifest,args.output)
    elif args.command == 'ablation':
        from .ablation import run
        result = run(args.config, args.prepare_only)
    elif args.command == 'ablation-evaluate':
        from .ablation_report import evaluate
        result = evaluate(args.config, args.watch)
    elif args.command == 'ablation-report':
        from .ablation_report import report
        result = report(args.config)
    elif args.command == 'ablation-audit':
        from .ablation_report import audit
        result = audit(args.config)
        if args.watch:
            import time
            from .data import read_json
            expected = len(read_json(args.config)['variants'])
            while len(result['models']) < expected:
                time.sleep(30)
                result = audit(args.config)
    elif args.command == 'gradient-head':
        from .gradient_head import run
        result = run(args.config)
    elif args.command == 'gradient-head-report':
        from .gradient_head_report import report
        result = report(args.config)
    elif args.command == 'data-ablation':
        from .data_ablation import run
        result = run(args.config, args.phase)
    elif args.command == 'joint-state':
        from .joint_state import run
        result = run(args.config, args.phase)
    elif args.command == 'joint-state-train':
        from .joint_state import train_run
        result = train_run(args.config, args.name)
    elif args.command == 'single-ray':
        from .single_ray import run
        result = run(args.config, args.phase)
    elif args.command == 'camera-study':
        from .camera_study import run
        result = run(args.config, args.phase)
    elif args.command == 'spline-study':
        from .spline_study import run
        result = run(args.phase)
    elif args.command == 'hazard-study':
        from .hazard_study import run
        result = run(args.config, args.phase)
    elif args.command == 'distance-study':
        from .distance_study import run
        result = run(args.config, args.phase)
    elif args.command == 'camera-holdout':
        from .camera_holdout import run
        result = run(args.config, args.phase)
    elif args.command == 'multiview-data':
        from .multiview_data import run
        result = run(args.config, args.phase)
    elif args.command == 'multiview-study':
        from .multiview_study import run
        result = run(args.config, args.phase)
    elif args.command == 'residual-supervision':
        from .residual_supervision import run
        result = run(args.config, args.phase)
    elif args.command == 'causal-residual':
        if args.phase == 'all':
            from .causal_workflow import run_all
            result = run_all(args.config)
        else:
            from .causal_study import run
            result = run(args.config, args.phase)
    elif args.command == 'causal-gradient':
        from .causal_gradient_study import run
        result = run(args.config, args.phase)
    elif args.command == 'causal-gradient-step':
        from .causal_gradient_step import run
        result = run(args.config, args.phase)
    else: result = compare(args.checkpoint, args.dataset, args.output, args.split, args.device, args.points, args.subdivisions, args.limit)
    print(json.dumps(result, indent=2, allow_nan=False))
