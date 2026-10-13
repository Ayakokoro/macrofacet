"""Compare fixed-distance first-hit gradients at the existing neural camera hits."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import shutil
import sys
import random

from compare_se_camera_transmittance import ROOT, digest, read, run, write


def conditional_target(meta, segments, quantile):
    """Invert the exported Bernstein hazard conditional on a hit before escape."""
    if not 0 < quantile < 1:
        raise ValueError('conditional target quantile must lie in (0,1)')
    totals=[(s['end_x']-s['begin_x'])*math.fsum(s['rates'])/4 for s in segments]
    total=math.fsum(totals)
    probability=-math.expm1(-total)
    if not probability > 0:
        return None
    optical_depth=-math.log1p(-quantile*probability)
    remaining=optical_depth
    for segment,mass in zip(segments,totals):
        if remaining <= mass:
            coefficients=[0.]
            for rate in segment['rates']:
                coefficients.append(coefficients[-1]+rate)
            dx=segment['end_x']-segment['begin_x']
            def integral(u):
                return dx/4*sum(c*math.comb(4,k)*u**k*(1-u)**(4-k) for k,c in enumerate(coefficients))
            lo,hi=0.,1.
            for _ in range(64):
                mid=(lo+hi)/2
                if integral(mid)<remaining:lo=mid
                else:hi=mid
            distance=(segment['begin_x']+dx*(lo+hi)/2)*meta['ell']
            return dict(distance=distance,conditional_quantile=quantile,
                        hit_probability=probability,optical_depth=optical_depth,
                        horizon_cumulative_hazard=total,
                        source_replayed_sample_hit=meta['replayed_sample_hit'])
        remaining-=mass
    raise RuntimeError('conditional collision quantile was not bracketed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera-suite', type=Path, required=True)
    parser.add_argument('--surface-suite', type=Path, help='existing reflected B suite, to compare the second collision')
    parser.add_argument('--target-policy', choices=('replayed','conditional_median','conditional_random'), default='replayed')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=65536)
    parser.add_argument('--gp-trials', type=int, default=65536)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if min(args.trials, args.gp_trials, args.workers) < 1:
        parser.error('sample budgets and worker count must be positive')
    os.chdir(ROOT)
    camera = args.camera_suite.resolve()
    flight_suite=args.surface_suite.resolve() if args.surface_suite else camera
    surface=bool(args.surface_suite)
    out = args.output.resolve()
    if out in (camera,flight_suite):
        parser.error('use a separate gradient output directory')
    selection = read(camera / 'selection.json')
    records = read(flight_suite / 'results.json')
    source = selection['render_config']
    flight_selection=read(flight_suite/'selection.json')
    if surface and (flight_selection.get('flight')!='after_first_hit' or
                    flight_selection['render_config']!=source or flight_selection['pixels']!=selection['pixels']):
        parser.error('surface suite must use the same render configuration and selected pixels')
    if not surface and args.target_policy!='replayed':
        parser.error('conditional target selection is only enabled for the second collision')
    if source['field']['kernel_type'] != 'squared_exponential':
        parser.error('requires the SE camera comparison')
    for path, key in ((source['field']['grid_file'], 'field_sha256'),
                      (source['transport']['renewal']['model'], 'model_sha256')):
        if digest(ROOT / path) != selection[key]:
            raise RuntimeError(f'{key} changed since the camera comparison')
    inspect = ROOT / 'build/Release/macrofacet_inspect_collision_gradients.exe'
    exe = ROOT / 'build/Release/macrofacet_experiments.exe'
    settings = dict(camera_suite=str(camera), render_config=source,
        camera_selection_sha256=digest(camera/'selection.json'), field_sha256=selection['field_sha256'],
        model_sha256=selection['model_sha256'], inspect_sha256=digest(inspect), gp_executable_sha256=digest(exe),
        trials_per_transport_mode=args.trials, gp_proposals_per_resolution=args.gp_trials,
        pixels=selection['pixels'], maximum_gp_coarse_step=1/64, maximum_gp_fine_step=1/128,
        rays={r['directory']: digest(flight_suite/r['directory']/'neural/ray.json')
              for r in records if r['status']=='evaluated'},
        conditioning='Same fixed distance: existing sample 0 neural first hit. No distance bins. Gradients are resampled, not fixed to the original hit gradient.')
    if surface:
        settings.update(collision_index=2,source_suite=str(flight_suite),
            source_selection_sha256=digest(flight_suite/'selection.json'),target_policy=args.target_policy,
            neural_segments_sha256={r['directory']:digest(flight_suite/r['directory']/'neural/neural_segments.json')
                                    for r in records if r['status']=='evaluated'},
            conditioning='Fixed first-hit position, full gradient and reflection direction; same selected second-collision distance for all modes; no earlier camera-path survival history. classic_local ignores the birth observation.')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'selection.json').exists() and read(out/'selection.json') != settings:
        raise RuntimeError('inputs changed; choose a new output directory')
    write(out/'selection.json',settings)
    write(out/'render_config.json',source)
    shutil.copyfile(camera/'scene.png',out/'scene.png')

    def evaluate(record):
        folder=out/record['directory']
        folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():
            return read(folder/'complete.json')
        result=dict(id=record['id'],pixel=record['pixel'],directory=record['directory'],status='no_target',seconds={})
        if record['status']=='evaluated':
            meta=read(flight_suite/record['directory']/'neural/ray.json')
            target=dict(distance=meta['replayed_sample_distance_from_entry']) if meta['replayed_sample_hit'] else None
            if args.target_policy!='replayed':
                quantile=.5 if args.target_policy=='conditional_median' else random.Random(source['seed']+record['id']*1000003+20261012).random()
                target=conditional_target(meta,read(flight_suite/record['directory']/'neural/neural_segments.json'),quantile)
            if target is not None:
                result['status']='evaluated'
                target['policy']=args.target_policy
                write(folder/'target_selection.json',target)
                print(f"[{record['id']}/16] {record['pixel']} transport gradients",flush=True)
                command=[inspect,out/'render_config.json',flight_suite/record['directory']/'neural/ray.json',folder,args.trials]
                if args.target_policy!='replayed':command+=['--distance',format(target['distance'],'.17g')]
                code,seconds=run(command,folder/'inspect.log')
                if code:
                    raise RuntimeError((folder/'inspect.log').read_text(encoding='utf-8',errors='replace'))
                result['seconds']['transport']=seconds
                q=target['distance']/meta['ell']
                intervals=max(2,math.ceil(q*64))
                for resolution,name in enumerate(('coarse','fine')):
                    config=read(flight_suite/record['directory']/'neural/reference_fine.json')
                    config['seed']=source['seed']+20261011+record['id']*1000003+resolution*104729+(1000000007 if surface else 0)
                    fp=config['first_passage']
                    fp['grid']=dict(max_time=q,step_sizes=[q/(intervals*2**resolution)])
                    fp['fixed_endpoint']=dict(enabled=True,only=True,distances=[target['distance']])
                    fp['monte_carlo'].update(trajectories=args.gp_trials,thread_count=4,write_raw_samples=True)
                    config['output_directory']=str(folder/f'gp_{name}')
                    request=folder/f'gp_{name}.json'
                    write(request,config)
                    print(f"[{record['id']}/16] {record['pixel']} GP {name}: {args.gp_trials} proposals",flush=True)
                    code,seconds=run([exe,'first-passage','--config',request],folder/f'gp_{name}.log')
                    if code:
                        raise RuntimeError((folder/f'gp_{name}.log').read_text(encoding='utf-8',errors='replace'))
                    result['seconds'][f'gp_{name}']=seconds
                print(f"[{record['id']}/16] {record['pixel']} done",flush=True)
        write(folder/'complete.json',result)
        return result

    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(evaluate,r) for r in records]):
            results.append(future.result())
            write(out/'results.json',sorted(results,key=lambda r:r['id']))
    sys.path.insert(0,str(ROOT/'data_analysis'))
    from plot_collision_gradients import plot_suite
    plot_suite(out)
    from plot_transport_camera_rays import plot_suite as plot_transport_suite
    source_suites=[camera]
    if (camera/'view_links.json').exists():
        links=read(camera/'view_links.json')
        if 'surface' in links:
            source_suites.append((camera/links['surface']).resolve().parent)
    for suite in source_suites:
        links=read(suite/'view_links.json') if (suite/'view_links.json').exists() else {}
        links['gradient_second' if surface else 'gradient']=Path(os.path.relpath(out/'comparison.html',suite)).as_posix()
        write(suite/'view_links.json',links)
        plot_transport_suite(suite)
    if surface:
        previous_links=read(camera/'view_links.json')
        if 'gradient' in previous_links:
            first=(camera/previous_links['gradient']).resolve().parent
            for directory in (first,out):
                write(directory/'view_links.json',dict(
                    gradient=Path(os.path.relpath(first/'comparison.html',directory)).as_posix(),
                    gradient_second=Path(os.path.relpath(out/'comparison.html',directory)).as_posix()))
                plot_suite(directory,reuse_reports=True)
    print(f'Completed: {out / "comparison.html"}',flush=True)


if __name__=='__main__':
    main()
