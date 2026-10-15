"""Paired frozen-state and joint h16 training on the same trajectory batches."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import subprocess
import sys
import time

import torch

from renewal.collect import digest, resolve, write_json
from renewal.model import mixture_log_prob
from . import FORMAT, VERSION, EVENT, SE_KERNEL
from .batch import CudaIntegrator, make_plan, losses_many, query_many
from .data import load_data, read_json
from .data_ablation import immutable_json
from .gradient_head import backbone_digest, parameters_at, evaluate_head, to_device
from .train import load_model, metrics


LABEL_FIELDS = ('distance','hit','initial_sign','speed')


def epoch_batches(profiles, settings, seed, epoch):
    key = f'first_hit_joint_h16:{seed}:{epoch}'.encode('ascii')
    value = int.from_bytes(hashlib.blake2b(key,digest_size=8).digest(),'little')
    generator = torch.Generator().manual_seed(value)
    order = torch.randperm(len(profiles),generator=generator).tolist()
    for begin in range(0,len(order),settings['profiles_per_batch']):
        yield [(i,torch.randperm(len(profiles[i]['hit']),generator=generator)[:settings['samples_per_profile']])
               for i in order[begin:begin+settings['profiles_per_batch']]]


def draw_batch(profiles, selection):
    return [{**profiles[i], **{key:profiles[i][key][indices] for key in LABEL_FIELDS}}
            for i,indices in selection]


def next_trace(previous, selection):
    state=hashlib.sha256(bytes.fromhex(previous))
    for i,indices in selection:
        state.update(i.to_bytes(8,'little'))
        state.update(indices.contiguous().numpy().tobytes())
    return state.hexdigest()


@torch.no_grad()
def cache_distance_terms(source, profiles, contexts, device):
    """Cache source-only distance terms and exact trajectory-to-hit mapping."""
    records=[]
    assert len(profiles)==len(contexts['profiles'])
    for begin in range(0,len(profiles),8):
        batch=profiles[begin:begin+8]
        predictions=query_many(source,[p['features'] for p in batch],[p['distance'] for p in batch],with_mixture=False)
        for p,r,meta in zip(batch,predictions,contexts['profiles'][begin:begin+8]):
            if p['id']!=meta['id'] or int(p['hit'].sum())!=meta['hits']:
                raise ValueError('cached profile or hit order changed')
            torch.testing.assert_close(p['speed'][p['hit']],contexts['speed'][meta['begin']:meta['end']],atol=0,rtol=0)
            branch=(p['initial_sign']<0).long().to(device)
            rows=torch.arange(len(branch),device=device)
            constant=-r['log_prior'][branch]+r['H'][branch,rows]-p['hit'].to(device)*r['log_hazard'][branch,rows]
            mapping=torch.full((len(branch),),-1,dtype=torch.long)
            mapping[p['hit']]=torch.arange(meta['begin'],meta['end'])
            records.append({'constant':constant.cpu(),'hit_index':mapping})
    return records


def cached_loss(model, contexts, distances, selection):
    count=sum(len(indices) for _,indices in selection)
    constant=sum(distances[i]['constant'][indices].sum() for i,indices in selection)
    index=torch.cat([distances[i]['hit_index'][indices] for i,indices in selection])
    index=index[index>=0].to(contexts['speed'].device)
    if not len(index):
        raise ValueError('head-only batch contains no gradient labels')
    mixture=parameters_at(model,contexts['inputs'][index],contexts['derivative'][index])
    speed=-mixture_log_prob(*mixture,contexts['speed'][index]).sum()
    return (constant.to(speed.device)+speed)/count, speed.detach(), len(index), count


def parameter_changes(source, model):
    return {name:max(float((value-source.state_dict()[key]).abs().max())
                     for key,value in model.state_dict().items() if key.startswith(name+'.'))
            for name in ('initial_encoder','dynamics','hazard_head','mixture_head')}


def prepare(config_path):
    spec=read_json(config_path)
    root=resolve(spec['output_directory']);root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(spec['cpu_threads'])
    source,_=load_model(resolve(spec['initialize_from']),spec['device'])
    if source.config.architecture!='rno' or source.config.hidden!=16 or source.config.components!=8:
        raise ValueError('this experiment holds h16/K8 fixed')
    profiles=load_data(resolve(spec['dataset']))['profiles']
    training=[p for p in profiles if p['split']=='train']
    spec_identity={key:digest(resolve(spec[key])) for key in ('dataset','evaluation_dataset','scene_dataset',
                     'initialize_from','training_contexts','evaluation_contexts')}
    identity={'specification_sha256':digest(config_path),'sha256':spec_identity,
              'source_model_config':source.configuration(),
              'implementation_sha256':{str(Path(__file__).relative_to(resolve('.'))):digest(Path(__file__)),
                  'python/first_hit_se/batch.py':digest(resolve('python/first_hit_se/batch.py')),
                  'python/first_hit_se/model.py':digest(resolve('python/first_hit_se/model.py'))},
              'updates_per_run':spec['training']['epochs']*math.ceil(len(training)/spec['training']['profiles_per_batch']),
              'objective':'Complete first-zero joint NLL per trajectory, including censored paths; same batches and loss in both modes.',
              'sampler':'Uniform profile permutation and fixed-size trajectory subset per profile each epoch; BLAKE2b(seed,epoch).',
              'selection':'Primary fixed final epoch; secondary minimum original validation joint NLL.',
              'runs':[{'name':f'{mode}_s{seed}','mode':mode,'seed':seed}
                      for seed in spec['seeds'] for mode in spec['modes']]}
    immutable_json(root/'experiment.json',identity)
    cached=torch.load(resolve(spec['training_contexts']),map_location='cpu',weights_only=True)
    if cached['identity']!={'dataset_sha256':spec_identity['dataset'],'checkpoint_sha256':spec_identity['initialize_from']}:
        raise ValueError('source training contexts do not match model/data')
    frozen=torch.load(resolve(spec['evaluation_contexts']),map_location='cpu',weights_only=True)
    if (frozen['identity']['dataset_sha256']!=spec_identity['evaluation_dataset'] or
            frozen['identity']['scene_dataset_sha256']!=spec_identity['scene_dataset'] or
            frozen['identity']['checkpoint_sha256']!=spec_identity['initialize_from']):
        raise ValueError('source evaluation contexts do not match model/data')
    if not (root/'distance_cache.pt').exists():
        print('Caching fixed-state distance likelihoods',flush=True)
        distance=cache_distance_terms(source,training,cached['train'],spec['device'])
        validation=[p for p in profiles if p['split']=='validation']
        initial_validation=metrics(source,validation,device=spec['device'],batch_size=8)
        torch.save({'identity':spec_identity,'profiles':distance,'validation':initial_validation},root/'distance_cache.pt')
    return identity


def train_run(config_path,name):
    spec=read_json(config_path);root=resolve(spec['output_directory'])
    experiment=read_json(root/'experiment.json')
    if digest(config_path)!=experiment['specification_sha256']:
        raise ValueError('joint experiment specification changed')
    for path,value in experiment['implementation_sha256'].items():
        if digest(resolve(path))!=value:
            raise ValueError('training implementation changed during experiment')
    for key,value in experiment['sha256'].items():
        if digest(resolve(spec[key]))!=value:
            raise ValueError(f'experiment input changed: {key}')
    record=next(r for r in experiment['runs'] if r['name']==name)
    mode,seed=record['mode'],record['seed'];settings=spec['training']
    directory=resolve(spec['model_directory'])/name;directory.mkdir(parents=True,exist_ok=True)
    config={**settings,'mode':mode,'seed':seed,'experiment_sha256':digest(root/'experiment.json')}
    immutable_json(directory/'training_config.json',config)
    torch.set_num_threads(spec['cpu_threads']);torch.manual_seed(seed)
    source,_=load_model(resolve(spec['initialize_from']),spec['device'])
    source.requires_grad_(False)
    model=deepcopy(source)
    model.requires_grad_(mode=='joint');model.mixture_head.requires_grad_(True)
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=settings['epochs'],eta_min=settings['minimum_learning_rate'])
    data=load_data(resolve(spec['dataset']))['profiles']
    training=[p for p in data if p['split']=='train'];validation=[p for p in data if p['split']=='validation']
    history=[];best=math.inf;start=1;updates=0;trace='00'*32
    total_paths=total_hits=0
    if (directory/'last.pt').exists():
        saved=torch.load(directory/'last.pt',map_location=spec['device'],weights_only=True)
        if saved['training_config']!=config:
            raise ValueError('resume configuration changed')
        model.load_state_dict(saved['model_state']);optimizer.load_state_dict(saved['optimizer_state'])
        scheduler.load_state_dict(saved['scheduler_state'])
        history,best,start,updates=saved['history'],saved['best'],saved['epoch']+1,saved['optimizer_updates']
        trace,total_paths,total_hits=saved['sample_trace'],saved['paths_seen'],saved['hits_seen']
    if start>settings['epochs'] and (directory/'run_report.json').exists():
        return read_json(directory/'run_report.json')
    integrator=None
    if mode=='joint':
        train_plans=[make_plan(p['features'],model.config.maximum_step,spec['device']) for p in training]
        val_plans=[make_plan(p['features'],model.config.maximum_step,spec['device']) for p in validation]
        if settings.get('cuda_graph_solver'):
            maximum=max(len(p['dt']) for p in train_plans+val_plans)
            print(f'CAPTURE {name}: {maximum} steps, {settings["profiles_per_batch"]} rays',flush=True)
            integrator=CudaIntegrator(model,maximum,settings['profiles_per_batch'])
    else:
        contexts=to_device(torch.load(resolve(spec['training_contexts']),map_location='cpu',weights_only=True)['train'],spec['device'])
        cached=torch.load(root/'distance_cache.pt',map_location='cpu',weights_only=True)
        if cached['identity']!=experiment['sha256']:
            raise ValueError('distance likelihood cache changed')
        distances=cached['profiles'];v0=cached['validation']
        val_context=to_device(torch.load(resolve(spec['evaluation_contexts']),map_location='cpu',weights_only=True)['splits']['validation'],spec['device'])
    checkpoint_seconds=0.
    for epoch in range(start,settings['epochs']+1):
        began=time.perf_counter();model.train();total=speed_total=0.;paths=hits=0
        for selection in epoch_batches(training,settings,seed,epoch):
            trace=next_trace(trace,selection)
            optimizer.zero_grad(set_to_none=True)
            if mode=='joint':
                batch=draw_batch(training,selection)
                batch=[{k:v.to(spec['device']) if isinstance(v,torch.Tensor) else v for k,v in p.items()} for p in batch]
                results=losses_many(model,batch,[train_plans[i] for i,_ in selection],integrator)
                n=sum(len(p['hit']) for p in batch);hit_count=sum(int(p['hit'].sum()) for p in batch)
                loss=sum(r['complete_joint'].sum() for r in results)/n
                speed=sum(r['speed'].sum().detach() for r in results)
            else:
                loss,speed,hit_count,n=cached_loss(model,contexts,distances,selection)
            if not torch.isfinite(loss):
                raise FloatingPointError('joint-state loss is nonfinite')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable,settings['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();updates+=1
            total+=float(loss.detach())*n;speed_total+=float(speed);paths+=n;hits+=hit_count
            del loss,speed
            if mode=='joint':
                del results,batch
        rate=optimizer.param_groups[0]['lr'];scheduler.step()
        total_paths+=paths;total_hits+=hits
        checkpoint_seconds+=time.perf_counter()-began
        if epoch==1 or epoch%settings['validation_every']==0 or epoch==settings['epochs']:
            model.eval();began=time.perf_counter()
            if mode=='joint':
                val=metrics(model,validation,device=spec['device'],batch_size=settings['profiles_per_batch'],plans=val_plans,integrator=integrator)
            else:
                val={**v0,**evaluate_head(model,val_context,detailed=False)}
                delta=(val['speed_nll_on_hits']-v0['speed_nll_on_hits'])*v0['hits']/v0['samples']
                for key in ('joint_nll','complete_joint_nll'):
                    val[key]=v0[key]+delta
            checkpoint_seconds+=time.perf_counter()-began
            score=val['joint_nll']
            if not math.isfinite(score):
                raise FloatingPointError('nonfinite validation score')
            improved=score<best;best=min(score,best)
            if mode=='head_only' and backbone_digest(model)!=backbone_digest(source):
                raise AssertionError('frozen control backbone changed')
            history.append({'epoch':epoch,'updates':updates,'training_joint_nll':total/paths,
                            'training_speed_nll_on_hits':speed_total/hits,'validation':val,
                            'learning_rate':rate,'seconds':checkpoint_seconds})
            checkpoint_seconds=0.
            bundle={'format':FORMAT,'version':VERSION,'event':EVENT,'kernel':SE_KERNEL,
                    'model_config':model.configuration(),'model_state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    'training_config':config,'epoch':epoch,'optimizer_updates':updates,'validation':val,
                    'sample_trace':trace,'paths_seen':total_paths,'hits_seen':total_hits}
            if improved: torch.save(bundle,directory/'best.pt')
            torch.save({**bundle,'optimizer_state':optimizer.state_dict(),'scheduler_state':scheduler.state_dict(),
                        'history':history,'best':best},directory/'last.pt')
            write_json(directory/'history.json',history)
            print(f'{name} epoch={epoch} updates={updates}: val joint={score:.6f} speed={val["speed_nll_on_hits"]:.6f}',flush=True)
    changes=parameter_changes(source,model)
    if mode=='joint' and not all(v>0 for v in changes.values()):
        raise AssertionError('joint training did not update every network module')
    report={'name':name,'mode':mode,'seed':seed,'completed_epochs':settings['epochs'],'optimizer_updates':updates,
            'paths_seen':total_paths,'hits_seen':total_hits,'sample_trace':trace,'parameter_max_changes':changes,
            'trainable_parameters':sum(p.numel() for p in trainable),'training_seconds':sum(h['seconds'] for h in history),
            'checkpoint_sha256':digest(directory/'last.pt')}
    write_json(directory/'run_report.json',report)
    return report


def run(config_path,phase='all'):
    spec=read_json(config_path);root=resolve(spec['output_directory'])
    if phase in ('prepare','train','all'):
        experiment=prepare(config_path)
    if phase in ('train','all'):
        (root/'logs').mkdir(exist_ok=True)
        cli=Path(__file__).resolve().parents[1]/'first_hit_se_cli.py'
        for i,r in enumerate(experiment['runs'],1):
            path=resolve(spec['model_directory'])/r['name']/'run_report.json'
            if path.exists() and read_json(path)['optimizer_updates']==experiment['updates_per_run']:
                print('SKIP',r['name'],flush=True);continue
            print(f'START {i}/{len(experiment["runs"])} {r["name"]}',flush=True)
            with (root/'logs'/f'{r["name"]}.log').open('a',encoding='utf-8') as stream:
                subprocess.run([sys.executable,'-u',str(cli),'joint-state-train','--config',str(config_path),'--name',r['name']],
                               stdout=stream,stderr=subprocess.STDOUT,check=True)
            print(f'DONE {i}/{len(experiment["runs"])} {r["name"]}',flush=True)
    if phase in ('evaluate','report','audit','all'):
        from .joint_state_report import evaluate, report, audit
        if phase in ('evaluate','all'): evaluate(config_path)
        if phase in ('audit','all'): audit(config_path)
        if phase in ('report','all'): return report(config_path)
    return {'output':str(root),'phase':phase}
