"""Camera-only Tungsten-linear first-hit collection, training and physical-unit figures."""
from __future__ import annotations

import csv
import html
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

from renewal.collect import digest, resolve, write_json
from renewal.model import mixture_cdf
from renewal.reference_sampler import require_same_sampler
from .batch import make_plan, query_many
from .data import collect, load_data, read_json
from .geometry import mean_gradient, sample_gradient
from .train import export_model, load_model, train


def write_csv(path, records):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def pixel_of(profile):
    return tuple(read_json(Path(profile['source']).parent/'ray.json')['pixel'])


def audit_data(spec):
    """Check the actual training tensors against the requested camera and raw CSVs."""
    data_path = resolve(spec['dataset'])
    data = load_data(data_path)
    settings = read_json(resolve(spec['collection_config']))
    scene_path = resolve(spec['render_config'])
    scene = read_json(scene_path)
    manifest = read_json(data_path.parent/'manifest.json')
    if data['split_unit'] != 'ray' or data['event'] != 'unconditioned_first_zero':
        raise ValueError('camera study requires independent rays and first-zero labels')
    require_same_sampler(settings['reference_sampler'], data['reference_sampler'], 'camera study')
    if data['reference_sampler']['crossing_rule'] != 'tungsten_linear':
        raise ValueError('this study requires Tungsten total-field secant labels')
    expected = {}
    for source in settings['sources']:
        if resolve(source.get('render_config', '')) != scene_path or 'reference_config' in source:
            raise ValueError('all sources must use the requested camera scene')
        for pixel in source['pixels']:
            key = tuple(pixel)
            if key in expected:
                raise ValueError('pixel leakage across camera splits')
            expected[key] = (source['split'], source['realizations_per_profile'])
    records, seen_pixels, seeds = [], set(), set()
    max_origin_error = 0.
    for p in data['profiles']:
        ray = read_json(Path(p['source']).parent/'ray.json')
        pixel = tuple(ray['pixel'])
        if pixel in seen_pixels or expected[pixel] != (p['split'], len(p['hit'])):
            raise ValueError('wrong camera split, duplicate ray or missing realizations')
        seen_pixels.add(pixel)
        if resolve(ray['source_config']) != scene_path:
            raise ValueError('camera metadata references another scene')
        direction = np.asarray(ray['direction'])
        origin = np.asarray(ray['entry_origin'])
        predicted_origin = np.asarray(scene['render']['camera_position']) + ray['entry_distance']*direction
        error = float(np.max(np.abs(origin-predicted_origin)))
        max_origin_error = max(max_origin_error, error)
        if error > 1e-9 or not np.allclose(origin, p['geometry']['origin'], atol=1e-10, rtol=0):
            raise ValueError('training origin is not the medium entry on the camera ray')
        if ray['distance_convention'] != 'distance from activeDomain entry':
            raise ValueError('camera distance was not excluded')
        if not np.allclose(direction, p['geometry']['direction'], atol=1e-10, rtol=0):
            raise ValueError('training ray direction differs from camera ray')
        for name in ('sigma', 'ell'):
            if not math.isclose(ray[name], p['geometry'][name], rel_tol=1e-12):
                raise ValueError('physical units differ from VDB metadata')
        raw = Path(p['source'])/'first_passage_samples.csv'
        if digest(raw) != p['samples_sha256']:
            raise ValueError('raw reference samples changed after dataset preparation')
        with raw.open(encoding='utf-8', newline='') as stream:
            for i, row in enumerate(csv.DictReader(stream)):
                if row['seed'] in seeds:
                    raise ValueError('GP realization seeds overlap')
                seeds.add(row['seed'])
                if row['crossing_slope_method'] != 'total_field_linear_secant':
                    raise ValueError('nonlinear gradient labels in Tungsten dataset')
                if not math.isclose(float(row['event_distance']), float(p['distance'][i])*ray['ell'], abs_tol=1e-11):
                    raise ValueError('label distance includes an offset or uses wrong units')
        records.append({'profile': p['id'], 'split': p['split'], 'pixel_x': pixel[0], 'pixel_y': pixel[1],
                        'camera_to_entry_excluded': ray['entry_distance'], 'sigma': ray['sigma'], 'ell': ray['ell'],
                        'horizon_from_entry': p['horizon']*ray['ell'], 'samples': len(p['hit']),
                        'hits': int(p['hit'].sum()), 'negative_starts': int((p['initial_sign'] < 0).sum()),
                        **{f'entry_{a}': float(origin[j]) for j, a in enumerate('xyz')},
                        **{f'direction_{a}': float(direction[j]) for j, a in enumerate('xyz')}})
    skipped = {tuple(r['pixel']) for r in manifest['skipped_domain_misses']}
    if seen_pixels & skipped or seen_pixels | skipped != set(expected):
        raise ValueError('incomplete camera coverage')
    output = resolve(spec['output_directory'])
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'camera_rays.csv', records)
    summary = {'render_config': str(scene_path), 'render_config_sha256': digest(scene_path),
               'vdb': scene['field']['grid_file'], 'vdb_sha256': digest(resolve(scene['field']['grid_file'])),
               'dataset_sha256': digest(data_path), 'reference_sampler': data['reference_sampler'],
               'camera': scene['render'], 'all_sources_are_camera_rays': True,
               'ray_disjoint_splits': True, 'unique_gp_seeds': len(seeds),
               'distance_convention': 's = ell*q from activeDomain entry; camera-to-entry distance excluded',
               'maximum_entry_origin_error': max_origin_error, 'skipped_domain_misses': manifest['skipped_domain_misses'],
               'splits': {s: {'rays': sum(r['split'] == s for r in records),
                              **{k: sum(r[k] for r in records if r['split'] == s)
                                 for k in ('samples', 'hits', 'negative_starts')}}
                          for s in ('train', 'validation', 'test')}}
    write_json(output/'data_audit.json', summary)
    return summary


def plotting():
    os.environ.setdefault('MPLCONFIGDIR', str(resolve('outputs/.matplotlib')))
    try:
        import matplotlib
    except ImportError:
        # Reuse the repository's existing plot-only dependencies when available.
        dependency = resolve('outputs/.analysis_deps')
        if not (dependency/'matplotlib').exists():
            raise RuntimeError('Install matplotlib or provide outputs/.analysis_deps')
        sys.path.insert(0, str(dependency))
        import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .18, 'svg.fonttype': 'none'})
    return plt


def save_figure(plt, figure, output, name):
    figure.savefig(output/f'{name}.png', dpi=170, bbox_inches='tight')
    figure.savefig(output/f'{name}.svg', bbox_inches='tight')
    plt.close(figure)


def wilson(observed, n):
    z = 1.959963984540054
    center = (observed+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(observed*(1-observed)/n+z*z/(4*n*n))/(1+z*z/n)
    return center-half, center+half


def sample_mixture(mixture, uniforms):
    """Inverse-CDF sampling avoids unstable tail subtraction for truncated normals."""
    lo = torch.zeros(len(uniforms), dtype=torch.float64)
    hi = torch.ones_like(lo)
    u = torch.as_tensor(uniforms, dtype=torch.float64)
    for _ in range(64):
        mask = mixture_cdf(*mixture, hi) < u
        if not mask.any():
            break
        hi[mask] *= 2
    else:
        raise FloatingPointError('failed to bracket gradient mixture quantile')
    for _ in range(48):
        mid = (lo+hi)/2
        below = mixture_cdf(*mixture, mid) < u
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    return ((lo+hi)/2).numpy()


def gradient_windows(profile, centers, half_width):
    """Use pilot realizations to place windows, then evaluate disjoint realizations."""
    mid = len(profile['hit'])//2
    pilot = profile['distance'][:mid][profile['hit'][:mid]]
    if len(pilot) < 256:
        raise ValueError('gradient ray has too few pilot hits')
    result = []
    for center in centers:
        lo, at, hi = torch.quantile(pilot, torch.tensor([center-half_width, center, center+half_width], dtype=torch.float64))
        at = pilot[torch.argmin(torch.abs(pilot-at))]
        mask = profile['hit'].clone()
        mask[:mid] = False
        mask &= (profile['distance'] >= lo) & (profile['distance'] < hi)
        indices = mask.nonzero().flatten()
        if len(indices) < 128:
            raise ValueError('gradient window has too few independent held-out hits')
        result.append((float(lo), float(at), float(hi), indices))
    return result


@torch.no_grad()
def evaluate_curves(model, profiles):
    records = []
    for begin in range(0, len(profiles), 8):
        batch = profiles[begin:begin+8]
        queries = [torch.linspace(0, p['horizon'], 513, dtype=torch.float64) for p in batch]
        predictions = query_many(model, [p['features'] for p in batch], queries, with_mixture=False)
        for p, q, prediction in zip(batch, queries, predictions):
            hits = p['distance'][p['hit']].sort().values
            observed = (1-torch.searchsorted(hits, q, right=True).double()/len(p['hit'])).numpy()
            predicted = prediction['T'].numpy()
            if abs(predicted[0]-1) > 1e-12 or np.any(np.diff(predicted) > 1e-12):
                raise AssertionError('transmittance must start at one and decrease')
            low, high = wilson(observed, len(p['hit']))
            records.append({'pixel': list(pixel_of(p)), 'profile': p['id'], 'samples': len(p['hit']),
                            'q': q.tolist(), 'distance': (q*p['geometry']['ell']).tolist(),
                            'reference': observed.tolist(), 'rno': predicted.tolist(),
                            'reference_low': low.tolist(), 'reference_high': high.tolist(),
                            'max_abs_T_error': float(np.max(np.abs(predicted-observed))),
                            'rmse_T': float(np.sqrt(np.mean((predicted-observed)**2))),
                            'terminal_reference_T': float(observed[-1]), 'terminal_rno_T': float(predicted[-1])})
    return records


@torch.no_grad()
def evaluate_gradients(model, profiles, spec, output):
    rng = np.random.default_rng(spec['seed'])
    records = []
    for pixel in spec['gradient_pixels']:
        p = next(p for p in profiles if pixel_of(p) == tuple(pixel))
        windows = gradient_windows(p, spec['gradient_quantiles'], spec['gradient_quantile_half_width'])
        all_indices = torch.cat([item[3] for item in windows])
        locations = p['distance'][all_indices]
        prediction = query_many(model, [p['features']], [locations])[0]
        offset = 0
        for number, (lo, at, hi, indices) in enumerate(windows):
            n = len(indices)
            row = torch.arange(offset, offset+n)
            branch = (p['initial_sign'][indices] < 0).long()
            mixture = tuple(prediction[key][branch, row] for key in ('log_weights', 'means', 'scales'))
            observed_w = p['speed'][indices].numpy()
            predicted_w = sample_mixture(mixture, rng.uniform(1e-12, 1-1e-12, n))
            scale = p['geometry']['sigma']/p['geometry']['ell']
            signs = p['initial_sign'][indices].numpy()
            ref_d, model_d = -signs*scale*observed_w, -signs*scale*predicted_w
            ref_g, model_g = [], []
            for q, d1, d2 in zip(p['distance'][indices].tolist(), ref_d, model_d):
                mu = mean_gradient(p, q)
                # Paired transverse draws isolate the error of the learned head.
                state = rng.bit_generator.state
                ref_g.append(sample_gradient(p['geometry'], mu, d1, rng))
                rng.bit_generator.state = state
                model_g.append(sample_gradient(p['geometry'], mu, d2, rng))
            ref_g, model_g = np.asarray(ref_g), np.asarray(model_g)
            direction = np.asarray(p['geometry']['direction'])
            projection_error = max(float(np.max(np.abs(ref_g@direction-ref_d))),
                                   float(np.max(np.abs(model_g@direction-model_d))))
            if projection_error > 1e-10:
                raise AssertionError('reconstructed gradient changed longitudinal label')
            grid = torch.linspace(0, max(np.quantile(observed_w, .995)*1.3, 2.), 161, dtype=torch.float64)
            cdf = torch.stack([mixture_cdf(*mixture, value).mean() for value in grid]).numpy()
            empirical = np.searchsorted(np.sort(observed_w), grid.numpy(), side='right')/n
            pit = np.sort(mixture_cdf(*mixture, torch.as_tensor(observed_w)).numpy())
            ks = float(max(np.max(np.arange(1, n+1)/n-pit), np.max(pit-np.arange(n)/n)))
            ell = p['geometry']['ell']
            point = np.asarray(p['geometry']['origin'])+at*ell*direction
            name = f'gradient_{pixel[0]}_{pixel[1]}_{number+1}'
            np.savez_compressed(output/f'{name}.npz', q=p['distance'][indices].numpy(),
                                reference_gradient=ref_g, rno_gradient=model_g,
                                reference_W=observed_w, rno_W=predicted_w,
                                reference_derivative=ref_d, rno_derivative=model_d)
            records.append({'name': name, 'pixel': pixel, 'window': number+1, 'reference_hits': n,
                            'q_interval': [lo, hi], 'q_center': at, 'distance_from_entry': at*ell,
                            'distance_interval': [lo*ell, hi*ell], 'world_point': point.tolist(),
                            'direction': direction.tolist(), 'gradient_projection_error': projection_error,
                            'speed_pit_ks': ks, 'cdf_max_error': float(np.max(np.abs(cdf-empirical))),
                            'reference_cdf_dkw95': math.sqrt(math.log(40)/(2*n)),
                            'W_grid': grid.tolist(), 'reference_cdf': empirical.tolist(), 'rno_cdf': cdf.tolist(),
                            'reference_mean_gradient': ref_g.mean(0).tolist(), 'rno_mean_gradient': model_g.mean(0).tolist()})
            offset += n
        print(f'gradient windows evaluated: {pixel}', flush=True)
    return records


def make_figures(spec, report, data, output):
    plt = plotting()
    wanted = {tuple(p) for p in spec['visualization_pixels']}
    selected = [r for r in report['curves'] if tuple(r['pixel']) in wanted]
    fig, axes = plt.subplots(math.ceil(len(selected)/2), 2, figsize=(13, 3.1*math.ceil(len(selected)/2)), squeeze=False)
    for ax, r in zip(axes.flat, selected):
        x = r['distance']
        ax.fill_between(x, r['reference_low'], r['reference_high'], color='#64748b', alpha=.25, label='MC pointwise 95% CI')
        ax.plot(x, r['reference'], color='#334155', lw=1.7, label='Tungsten-linear MC')
        ax.plot(x, r['rno'], color='#d94824', lw=1.7, label='RNO h16')
        ax.set(title=f"Pixel {tuple(r['pixel'])} | max error {r['max_abs_T_error']:.4f}",
               xlabel='Distance from medium entry (world units)', ylabel='Transmittance', ylim=(-.025, 1.025))
    for ax in list(axes.flat)[len(selected):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Held-out camera rays | first collision only | camera-to-entry distance excluded', fontsize=14)
    fig.tight_layout()
    save_figure(plt, fig, output, 'transmittance')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    markers = {'train': 'o', 'validation': 's', 'test': '^'}
    for split, marker in markers.items():
        ps = [p for p in data['profiles'] if p['split'] == split]
        xy = np.asarray([pixel_of(p) for p in ps])
        axes[0].scatter(xy[:, 0], xy[:, 1], s=22, marker=marker, alpha=.65, label=f'{split}: {len(ps)}')
    for pixel in spec['visualization_pixels']:
        axes[0].annotate(str(tuple(pixel)), pixel, xytext=(3, 5), textcoords='offset points', fontsize=7)
        axes[0].scatter(*pixel, s=75, facecolors='none', edgecolors='black')
    camera = report['data_audit']['camera']
    axes[0].set(xlim=(0, camera['width']), ylim=(camera['height'], 0), aspect='equal',
                xlabel='Pixel x', ylabel='Pixel y', title='Independent camera-ray splits')
    axes[0].legend(fontsize=8)
    history = read_json(resolve(spec['checkpoint']).parent/'history.json')
    axes[1].plot([r['epoch'] for r in history], [r['training_nll'] for r in history], label='Training joint NLL')
    axes[1].plot([r['epoch'] for r in history], [r['validation']['joint_nll'] for r in history], label='Validation joint NLL')
    axes[1].axvline(report['training']['selected_epoch'], ls='--', color='#d94824', label='Selected checkpoint')
    axes[1].set(xlabel='Epoch', ylabel='Joint NLL', title='RNO trained from scratch on this sampler')
    axes[1].legend()
    fig.tight_layout()
    save_figure(plt, fig, output, 'coverage_training')

    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    lanes = fig.add_subplot(1, 2, 2)
    camera_origin = np.asarray(camera['camera_position'])
    ax.scatter(*camera_origin, c='black', marker='s', s=45, label='Camera')
    extent = [camera_origin]
    for i, pixel in enumerate(spec['gradient_pixels']):
        p = next(p for p in data['profiles'] if pixel_of(p) == tuple(pixel))
        origin, direction = np.asarray(p['geometry']['origin']), np.asarray(p['geometry']['direction'])
        end = origin+p['horizon']*p['geometry']['ell']*direction
        outside = np.stack((camera_origin, origin))
        inside = np.stack((origin, end))
        ax.plot(*outside.T, color='gray', ls='--', alpha=.5, label='Excluded camera segment' if i == 0 else None)
        ax.plot(*inside.T, lw=1.3, label=str(tuple(pixel)))
        ax.scatter(*origin, marker='x', color='black', s=25)
        extent.extend((origin, end))
        lane = len(spec['gradient_pixels'])-1-i
        lanes.plot([0, p['horizon']*p['geometry']['ell']], [lane, lane], color='#94a3b8', lw=1.5)
        lanes.scatter(0, lane, marker='x', color='black')
        for j, r in enumerate(report['gradients']):
            if r['pixel'] != pixel:
                continue
            s = r['distance_from_entry']
            lanes.scatter(s, lane, color='#d94824', s=24)
            lanes.annotate(f'P{j+1}', (s, lane), xytext=(6, 14+20*(r['window']-1)),
                           textcoords='offset points', fontsize=9,
                           arrowprops={'arrowstyle': '-', 'lw': .6, 'color': '#475569'})
    for j, r in enumerate(report['gradients']):
        point = np.asarray(r['world_point'])
        ax.scatter(*point, color='#d94824', s=16)
        g = np.asarray(r['rno_mean_gradient'])
        ax.quiver(*point, *g, length=.15, normalize=True, color='#d94824')
    ax.set(xlabel='World x', ylabel='World y', zlabel='World z', title='Camera, medium entry and collision points')
    ax.set_box_aspect(np.ptp(np.asarray(extent), axis=0))
    ax.view_init(elev=24, azim=42)
    ax.legend(fontsize=8)
    lanes.set(xlabel='Distance from medium entry (world units)', ylabel='Camera pixel',
              yticks=list(range(len(spec['gradient_pixels']))),
              yticklabels=[str(tuple(p)) for p in reversed(spec['gradient_pixels'])],
              ylim=(-.3, len(spec['gradient_pixels'])-.25),
              title='Nine representative first-hit locations\nEach lane starts at its own medium entry')
    lanes.axvline(0, color='black', ls='--', lw=.8)
    fig.suptitle('Dashed 3D segments: excluded camera travel | Red arrows: normalized mean RNO gradient', fontsize=12)
    fig.tight_layout()
    save_figure(plt, fig, output, 'collision_locations')

    for r in report['gradients']:
        a = np.load(output/f"{r['name']}.npz")
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
        for j, ax in enumerate(axes[0]):
            ref, pred = a['reference_gradient'][:, j], a['rno_gradient'][:, j]
            edges = np.histogram_bin_edges(np.concatenate((ref, pred)), bins=34)
            ax.hist(ref, bins=edges, density=True, alpha=.30, color='#334155', label='Reference + SE transverse')
            ax.hist(pred, bins=edges, density=True, histtype='step', lw=1.7, color='#d94824', label='RNO + SE transverse')
            ax.set(xlabel=f"Gradient g{'xyz'[j]} (physical units)", ylabel='Density')
        axes[0, 0].legend(fontsize=8)
        ax = axes[1, 0]
        ref, pred = a['reference_derivative'], a['rno_derivative']
        edges = np.histogram_bin_edges(np.concatenate((ref, pred)), bins=34)
        ax.hist(ref, bins=edges, density=True, alpha=.3, color='#334155', label='Reference secant')
        ax.hist(pred, bins=edges, density=True, histtype='step', color='#d94824', lw=1.7, label='RNO secant')
        ax.set(xlabel='dot(ray direction, gradient)', ylabel='Density')
        ax.legend(fontsize=8)
        ax = axes[1, 1]
        observed = np.asarray(r['reference_cdf'])
        band = r['reference_cdf_dkw95']
        ax.fill_between(r['W_grid'], np.maximum(0, observed-band), np.minimum(1, observed+band), color='#64748b', alpha=.2)
        ax.plot(r['W_grid'], observed, color='#334155', label='Reference + 95% DKW band')
        ax.plot(r['W_grid'], r['rno_cdf'], color='#d94824', label='RNO averaged at held-out hits')
        ax.set(xlabel='W = -sign * ell/sigma * dot(direction, gradient)', ylabel='CDF', ylim=(0, 1),
               title=f"CDF error {r['cdf_max_error']:.4f} | PIT KS {r['speed_pit_ks']:.4f}")
        ax.legend(fontsize=7)
        fig.delaxes(axes[1, 2])
        ax = fig.add_subplot(2, 3, 6, projection='3d')
        for key, color, label in [('reference_gradient', '#334155', 'Reference'), ('rno_gradient', '#d94824', 'RNO')]:
            g = a[key][:min(72, len(a[key]))]
            g = g/np.linalg.norm(g, axis=1)[:, None]
            zero = np.zeros(len(g))
            ax.quiver(zero, zero, zero, *g.T, color=color, alpha=.22, linewidth=.5, length=1.)
        ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), xlabel='nx', ylabel='ny', zlabel='nz', title='Normalized gradient directions')
        ax.set_box_aspect((1, 1, 1))
        fig.suptitle(f"Pixel {tuple(r['pixel'])}, point {r['window']} | s = {r['distance_from_entry']:.5f}\n"
                     f"Independent first hits within s=[{r['distance_interval'][0]:.5f}, {r['distance_interval'][1]:.5f}] | n={r['reference_hits']}")
        fig.tight_layout()
        save_figure(plt, fig, output, r['name'])


def write_html(spec, report, output):
    training, audit = report['training'], report['data_audit']
    test = training['metrics']['test']
    camera = audit['camera']
    worst_gradient = max(report['gradients'], key=lambda r: r['cdf_max_error'])
    counts = '；'.join(f"{s}：{v['rays']} 条射线 / {v['samples']:,} 个实现" for s, v in audit['splits'].items())
    content = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
               '<meta name="viewport" content="width=device-width, initial-scale=1">',
               '<title>Tungsten 相机首次碰撞 · RNO</title>',
               '<style>body{max-width:1200px;margin:36px auto;padding:0 22px;color:#172235;background:#f8fafc;font:16px/1.65 system-ui,sans-serif}h1,h2{line-height:1.3}img{max-width:100%;background:white;border:1px solid #dce3ec;border-radius:8px}table{width:100%;border-collapse:collapse;background:white;font-size:14px}td,th{padding:9px;border-bottom:1px solid #dce3ec;text-align:left}code{overflow-wrap:anywhere}a{color:#175cad}.note{border-left:4px solid #d94824;padding:12px 18px;background:white}details{margin:20px 0}</style>',
               '<h1>Tungsten 相机首次碰撞 · RNO</h1>',
               '<p>场景：shader ball，σ≈0.05。全部数据来自指定配置中的 VDB 与相机；每个随机实现只记录进入介质后的首次过零，未碰撞保留为删失样本。</p>',
               f'<p>相机位置 {camera["camera_position"]}，目标 {camera["camera_target"]}，垂直视场角 {camera["vertical_fov_degrees"]}°，分辨率 {camera["width"]}×{camera["height"]}。VDB：<code>{html.escape(audit["vdb"])}</code>。</p>',
               '<p class="note">距离定义：<b>s=ℓq，从 activeDomain 入口开始，入口处 s=0、T(0)=1。</b>相机到入口的距离仅保存在几何审计中，不进入训练距离标签。入口不固定场值，保留正负两种初始符号。</p>',
               f'<p>{counts}。另有 {len(audit["skipped_domain_misses"])} 条候选射线未进入介质。</p>',
               '<p>采样固定为 GPIS function space，H=1、B=8、Δq=1/64、rank tolerance=1e-10；Tungsten 总场值线性求交、割线斜率标签。均值使用 VDB 的连续分片三次剖面。</p>',
               f'<p>RNO 隐状态 16 维，{training["parameters"]:,} 个参数；从随机初始化训练 {training["completed_epochs"]} 轮，按验证集联合 NLL 选择第 {training["selected_epoch"]} 轮。训练用时 {training["training_seconds"]/60:.1f} 分钟，停止原因：{html.escape(training["stop_reason"])}。</p>',
               f'<p>独立测试集：联合 NLL={test["joint_nll"]:.5f}，距离 NLL={test["distance_nll"]:.5f}，速度 PIT KS={test["speed_pit_ks"]:.5f}；逐射线最大透射率误差的平均值={report["mean_ray_max_T_error"]:.5f}。</p>',
               f'<p class="note">误差并非均匀分布：测试射线中最大的透射率绝对误差为 {report["maximum_ray_max_T_error"]:.5f}；九个梯度窗口中，像素 {worst_gradient["pixel"]} 的第 {worst_gradient["window"]} 个窗口速度 CDF 最大差为 {worst_gradient["cdf_max_error"]:.5f}，参考 DKW 95% 半宽为 {worst_gradient["reference_cdf_dkw95"]:.5f}。全局 PIT 较小不能替代局部梯度检查。</p>',
               '<p><a href="data_audit.json">数据与距离审计</a> · <a href="camera_rays.csv">相机射线 CSV</a> · <a href="analysis.json">完整指标</a> · <a href="transmittance.csv">透射率 CSV</a> · <a href="collision_points.csv">碰撞点 CSV</a> · <a href="first_hit_model.json">导出模型</a> · <a href="cpp/verification.json">Python/C++ 校验</a></p>',
               '<h2>相机射线覆盖与训练</h2><img src="coverage_training.png" alt="射线划分与训练曲线">',
               '<h2>测试射线透射率</h2><p>灰色为同配置 Monte Carlo 参考及逐点 95% Wilson 区间，红色为 RNO。所有横轴均为从介质入口起算的物理距离。区间只衡量 Monte Carlo 误差，不包含采样离散或有限历史偏差。</p>',
               '<a href="transmittance.svg">下载 SVG</a><img src="transmittance.png" alt="八条相机射线的透射率">',
               '<details><summary>全部测试射线误差</summary><table><tr><th>像素</th><th>max |ΔT|</th><th>RMSE</th><th>参考末端 T</th><th>RNO 末端 T</th></tr>']
    for r in report['curves']:
        content.append(f'<tr><td>{r["pixel"]}</td><td>{r["max_abs_T_error"]:.5f}</td><td>{r["rmse_T"]:.5f}</td><td>{r["terminal_reference_T"]:.5f}</td><td>{r["terminal_rno_T"]:.5f}</td></tr>')
    content += ['</table></details>', '<h2>代表碰撞位置与梯度</h2>',
                '<p>先用每条射线前半部分独立实现的首次命中距离，选取最接近 20%、50%、80% 分位的实际碰撞点；窗口宽度为两侧各 7.5% 分位。后半部分实现用于参考比较。图中位置是窗口的代表点，分布对应该距离窗口，不能解读为精确固定点的条件真值。</p>',
                '<p>纵向标签为 Tungsten 割线 D=−k·σ/ℓ·W。三维梯度使用该纵向分量与平稳 SE 的条件横向高斯补全；参考和 RNO 采用配对的横向随机数。灰色三维分布是“参考割线＋横向高斯”重建，不是额外独立测量的三维梯度。CDF 在相同的留出碰撞位置与符号上平均，用来单独评估梯度头。</p>',
                '<img src="collision_locations.png" alt="相机、介质入口与九个代表碰撞点">',
                '<table><tr><th>点</th><th>像素</th><th>入口距离 s</th><th>世界坐标</th><th>参考命中数</th><th>速度 PIT KS</th></tr>']
    for i, r in enumerate(report['gradients']):
        point = ', '.join(f'{v:.4f}' for v in r['world_point'])
        content.append(f'<tr><td><a href="#{r["name"]}">P{i+1}</a></td><td>{r["pixel"]}</td><td>{r["distance_from_entry"]:.5f}</td><td>({point})</td><td>{r["reference_hits"]}</td><td>{r["speed_pit_ks"]:.5f}</td></tr>')
    content.append('</table>')
    for i, r in enumerate(report['gradients']):
        name = r['name']
        content.append(f'<h3 id="{name}">P{i+1} · 像素 {r["pixel"]} · s={r["distance_from_entry"]:.5f}</h3><p><a href="{name}.svg">SVG</a> · <a href="{name}.npz">梯度样本 NPZ</a></p><img src="{name}.png" alt="P{i+1} 的梯度分量、纵向 CDF 和方向">')
    content += ['<h2>复现与适用范围</h2>',
                f'<pre><code>python python/first_hit_se_cli.py camera-study --config {html.escape(spec["study_config"])} --phase all</code></pre>',
                '<p>本结果只验证此场景中留出的相机射线，未验证跨场景或多次反弹。采用本仓库的 Tungsten 式线性求交后端；保留入口固定网格，不复制 Tungsten 原版的首点偏移、位置扰动或外层上穿拒绝规则。改变 H、B、步长或求交规则需重新采集和训练。</p></html>']
    (output/'summary.html').write_text('\n'.join(content), encoding='utf-8')


@torch.no_grad()
def report_study(spec):
    from .verify_cpp import verify_cpp
    torch.set_num_threads(1)
    output = resolve(spec['output_directory'])
    audit = audit_data(spec)
    data = load_data(resolve(spec['dataset']))
    model, _ = load_model(resolve(spec['checkpoint']))
    require_same_sampler(model.reference_sampler, data['reference_sampler'], 'camera report')
    profiles = [p for p in data['profiles'] if p['split'] == 'test']
    training = read_json(resolve(spec['checkpoint']).parent/'run_report.json')
    curves = evaluate_curves(model, profiles)
    print('held-out transmittance curves evaluated', flush=True)
    gradients = evaluate_gradients(model, profiles, spec, output)
    report = {'data_audit': audit, 'training': training, 'curves': curves, 'gradients': gradients,
              'checkpoint_sha256': digest(resolve(spec['checkpoint'])),
              'report_implementation_sha256': digest(Path(__file__)),
              'study_config_sha256': digest(resolve(spec['study_config'])),
              'mean_ray_max_T_error': float(np.mean([r['max_abs_T_error'] for r in curves])),
              'maximum_ray_max_T_error': max(r['max_abs_T_error'] for r in curves)}
    write_json(output/'analysis.json', report)
    write_csv(output/'transmittance.csv', [
        {'pixel_x': r['pixel'][0], 'pixel_y': r['pixel'][1], 'q': q, 'distance_from_entry': r['distance'][i],
         'reference_T': r['reference'][i], 'rno_T': r['rno'][i],
         'reference_lower95': r['reference_low'][i], 'reference_upper95': r['reference_high'][i]}
        for r in curves for i, q in enumerate(r['q'])])
    write_csv(output/'collision_points.csv', [
        {'name': r['name'], 'pixel_x': r['pixel'][0], 'pixel_y': r['pixel'][1],
         'distance_from_entry': r['distance_from_entry'], 'window_low': r['distance_interval'][0],
         'window_high': r['distance_interval'][1], **dict(zip('xyz', r['world_point'])),
         'reference_hits': r['reference_hits'], 'speed_pit_ks': r['speed_pit_ks'],
         'speed_cdf_max_error': r['cdf_max_error']} for r in gradients])
    export_model(resolve(spec['checkpoint']), output/'first_hit_model.json')
    verify_cpp(resolve(spec['checkpoint']), resolve(spec['dataset']),
               resolve('build/Release/macrofacet_query_first_hit_se.exe'), output/'cpp', limit=3, trials=512)
    torch.set_num_threads(1)
    make_figures(spec, report, data, output)
    write_html(spec, report, output)
    return {'summary': str(output/'summary.html'), 'mean_ray_max_T_error': report['mean_ray_max_T_error'],
            'test_metrics': training['metrics']['test'], 'gradient_points': len(gradients)}


def run(config_path, phase='all'):
    spec = {**read_json(config_path), 'study_config': str(config_path)}
    if phase in ('all', 'collect'):
        collect(resolve(spec['collection_config']))
        audit_data(spec)
    if phase in ('all', 'train'):
        audit_data(spec)
        config = read_json(resolve(spec['training_config']))
        folder = resolve(config['output_directory'])
        if (folder/'run_report.json').exists():
            if read_json(folder/'training_config.json') != config:
                raise ValueError('completed training config changed')
            if read_json(folder/'run_report.json')['dataset_sha256'] != digest(resolve(spec['dataset'])):
                raise ValueError('completed training dataset changed')
        else:
            train(resolve(spec['training_config']), resume=(folder/'last.pt').exists())
    if phase in ('all', 'report'):
        return report_study(spec)
    return {'phase': phase, 'output': spec['output_directory']}
