"""Train either architecture on identical unconditioned first-zero supervision."""
import math
from pathlib import Path
import time

import torch

from renewal.collect import digest, resolve, write_json
from renewal.profiles import subdivide_features
from renewal.reference_sampler import reference_sampler, require_same_sampler
from . import FORMAT, VERSION, EVENT, SE_KERNEL
from .data import load_data, profile_to_device, read_json
from .model import FirstHitNetwork, ModelConfig
from .batch import make_plan, losses_many, CudaIntegrator


def load_model(path, device="cpu"):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if (bundle.get("format"), bundle.get("version"), bundle.get("event"), bundle.get("kernel")) != (FORMAT, VERSION, EVENT, SE_KERNEL):
        raise ValueError("incompatible first-hit checkpoint")
    if bundle["model_config"]["architecture"] == "spline":
        from .spline import SplineConfig, SplineFirstHitNetwork
        model = SplineFirstHitNetwork(SplineConfig(**bundle["model_config"])).to(device)
    else:
        model = FirstHitNetwork(ModelConfig(**bundle["model_config"])).to(device)
    model.reference_sampler = reference_sampler(bundle.get("reference_sampler"))
    model.load_state_dict(bundle["model_state"])
    model.eval()
    return model, bundle


@torch.no_grad()
def metrics(model, profiles, device="cpu", *, batch_size=None, plans=None, integrator=None):
    for profile in profiles:
        require_same_sampler(getattr(model, "reference_sampler", None), profile.get("reference_sampler"), "evaluation")
    totals = {key: 0. for key in ("joint", "distance", "complete_joint", "speed")}
    count = events = negative = 0
    pits = []
    if plans is not None and len(plans) != len(profiles):
        raise ValueError("one geometry plan is required per evaluation profile")
    for begin in range(0, len(profiles), batch_size or 1):
        batch = [profile_to_device(p, device) for p in profiles[begin:begin+(batch_size or 1)]]
        if batch_size:
            results = losses_many(model, batch, plans=None if plans is None else plans[begin:begin+batch_size],
                                  integrator=integrator)
        else:
            p = batch[0]
            results = [model.losses(p["features"], p["distance"], p["hit"], p["initial_sign"], p["speed"])]
        for p, result in zip(batch, results):
            for key in totals:
                totals[key] += float(result[key].sum())
            count += len(p["distance"])
            events += int(p["hit"].sum())
            negative += int((p["initial_sign"] < 0).sum())
            pits.append(result["pit"][p["hit"]].cpu())
    if not count:
        raise ValueError("empty evaluation split")
    pit = torch.cat(pits).sort().values
    n = len(pit)
    ks = None if not n else max(float((torch.arange(1,n+1)/n-pit).max()), float((pit-torch.arange(n)/n).max()))
    return {"samples": count, "hits": events, "negative_starts": negative,
            "joint_nll": totals["joint"]/count, "distance_nll": totals["distance"]/count,
            "complete_joint_nll": totals["complete_joint"]/count,
            "speed_nll_on_hits": totals["speed"]/events if events else None, "speed_pit_ks": ks}


def train(config_path, resume=False):
    config = read_json(config_path)
    data_path, output = resolve(config["dataset"]), resolve(config["output_directory"])
    data = load_data(data_path)
    if "reference_sampler" in config:
        require_same_sampler(config["reference_sampler"], data["reference_sampler"], "training config and dataset")
    splits = {s: [p for p in data["profiles"] if p["split"] == s] for s in ("train", "validation", "test")}
    if not splits["train"] or not splits["validation"]:
        raise ValueError("training requires independent train and validation profiles")
    output.mkdir(parents=True, exist_ok=True)
    if (output/"last.pt").exists() and not resume:
        raise ValueError("output already contains a run; use --resume or a new directory")
    seed = config.get("seed", 20261013)
    torch.manual_seed(seed)
    torch.set_num_threads(config.get("cpu_threads", 4))
    device = config.get("device", "cpu")
    model = FirstHitNetwork(ModelConfig(**config.get("model", {}))).to(device)
    model.reference_sampler = data["reference_sampler"]
    initialization = None
    if config.get("initialize_from"):
        path = resolve(config["initialize_from"])
        initialized, _ = load_model(path)
        require_same_sampler(model.reference_sampler, initialized.reference_sampler, "warm start")
        if initialized.configuration() != model.configuration():
            if not config.get('expand_hidden_state', False):
                raise ValueError("warm start requires the same model configuration")
            model.expand_hidden_from(initialized)
        else:
            model.load_state_dict(initialized.state_dict())
        initialization = {"checkpoint": str(path), "checkpoint_sha256": digest(path)}
        if config.get('expand_hidden_state', False):
            initialization['method'] = 'function_preserving_hidden_expansion'
            initialization['source_hidden'] = initialized.config.hidden
        del initialized
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 3e-4), weight_decay=config.get("weight_decay", 1e-5))
    scheduler = None
    scheduler_type = 'plateau'
    if config.get("lr_scheduler"):
        settings = dict(config['lr_scheduler'])
        scheduler_type = settings.pop('type', 'plateau')
        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **settings)
        elif scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **settings)
        else:
            raise ValueError('unknown learning-rate scheduler')
    history, best, start, stale, updates, stopping_best = [], math.inf, 1, 0, 0, math.inf
    data_hash = digest(data_path)
    if resume:
        saved = torch.load(output/"last.pt", map_location=device, weights_only=True)
        if saved["training_config"] != config or saved["dataset_sha256"] != data_hash:
            raise ValueError("resume requires exactly the same configuration and dataset")
        if saved.get("initialization") != initialization:
            raise ValueError("the warm-start checkpoint changed since this run began")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        history, best, stale, start = saved["history"], saved["best"], saved["stale"], saved["epoch"]+1
        updates = saved.get("optimizer_updates", saved["epoch"]*math.ceil(len(splits["train"])/config.get("profiles_per_batch", 4)))
        stopping_best = saved.get("stopping_best", best)
        if scheduler:
            scheduler.load_state_dict(saved["scheduler_state"])
    write_json(output/"training_config.json", config)
    factors = config.get("subdivision_factors", [1])
    if not factors or any(type(n) is not int or n < 1 for n in factors):
        raise ValueError("subdivision factors must be positive integers")
    batch_size = config.get("profiles_per_batch", 4)
    if type(batch_size) is not int or batch_size < 1 or config.get("epochs",32) < 1:
        raise ValueError("positive batch size and epoch count are required")
    batched = config.get("batched_solver", False)
    captured = config.get("cuda_graph_solver", False)
    if captured and (not batched or factors != [1]):
        raise ValueError("CUDA graph solver requires batching and fixed partitions")
    curve_weight = config.get('curve_loss_weight', 0.)
    curve_points = config.get('curve_points', 33)
    if not math.isfinite(curve_weight) or curve_weight < 0 or (curve_weight and curve_points < 2):
        raise ValueError('invalid curve supervision configuration')
    if curve_weight:
        # Freeze empirical survival from every stored training realization,
        # rather than the randomly subsampled event labels in each epoch.
        for p in splits['train']:
            q = torch.linspace(0, p['horizon'], curve_points, dtype=torch.float64)
            p['curve_queries'] = q
            p['curve_reference'] = ((~p['hit'][:, None]) | (p['distance'][:, None] > q)).double().mean(0)
    plans, integrator = {}, None
    if batched:
        for split in ("train", "validation"):
            plans[split] = [make_plan(p["features"], model.config.maximum_step, device) for p in splits[split]]
    if captured and model.config.architecture == "rno":
        maximum = max(len(p["dt"]) for split in plans.values() for p in split)
        print(f"Capturing RNO solver: {maximum} steps, {batch_size} profiles", flush=True)
        integrator = CudaIntegrator(model, maximum, batch_size)
    write_json(output/"split_summary.json", {s: {"profiles": len(ps), "samples": sum(len(p["hit"]) for p in ps),
        "hits": sum(int(p["hit"].sum()) for p in ps)} for s, ps in splits.items()})
    stopped = "epoch_limit"
    for epoch in range(start, config.get("epochs", 32)+1):
        began = time.perf_counter()
        generator = torch.Generator().manual_seed(seed+epoch)
        order = torch.randperm(len(splits["train"]), generator=generator).tolist()
        model.train()
        train_sum = joint_sum = curve_sum = count = 0
        for begin in range(0, len(order), batch_size):
            optimizer.zero_grad(set_to_none=True)
            losses, weights, batch, batch_plans = [], [], [], []
            joint_terms, curve_terms = [], []
            for i in order[begin:begin+batch_size]:
                p = profile_to_device(splits["train"][i], device)
                sample_limit = config.get("samples_per_profile_per_epoch", len(p["distance"]))
                if sample_limit < 1:
                    raise ValueError("samples_per_profile_per_epoch must be positive")
                if sample_limit < len(p["distance"]):
                    indices = torch.randperm(len(p["distance"]), generator=generator)[:sample_limit].to(device)
                    p = {k: v[indices] if k in ("distance","hit","initial_sign","speed") else v for k,v in p.items()}
                features = p["features"]
                if factors != [1]:
                    choices = torch.randint(len(factors), (len(features),), generator=generator)
                    counts = torch.tensor(factors)[choices].to(device)
                    features = subdivide_features(features, counts)
                p = {**p, "features": features}
                batch.append(p)
                if batched:
                    batch_plans.append(plans["train"][i] if factors == [1] else make_plan(features, model.config.maximum_step, device))
            if batched:
                results = losses_many(model, batch, plans=batch_plans, integrator=integrator)
            else:
                results = [model.losses(p["features"], p["distance"], p["hit"], p["initial_sign"], p["speed"]) for p in batch]
            for p, result in zip(batch, results):
                values = result["complete_joint"]
                if epoch <= config.get("distance_warmup_epochs", 0):
                    values = values-result["speed"]
                loss = values.mean()
                joint_terms.append(loss.detach()*len(values))
                if curve_weight:
                    prediction = result['curve_prediction'] if batched else model.query(
                        p['features'], p['curve_queries'], with_mixture=False)['T']
                    curve_mse = ((prediction-p['curve_reference'])**2).mean()
                    curve_terms.append(curve_mse.detach()*len(values))
                    loss = loss+curve_weight*curve_mse
                losses.append(loss*len(values)); weights.append(len(values))
            loss = torch.stack(losses).sum()/sum(weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite first-hit training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("gradient_clip", 1.), error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            train_sum += float(loss.detach())*sum(weights); count += sum(weights)
            joint_sum += float(torch.stack(joint_terms).sum())
            if curve_terms:
                curve_sum += float(torch.stack(curve_terms).sum())
            # Release autograd graphs before the captured solver is replayed.
            del results, result, loss, losses, values
        model.eval()
        validation = metrics(model, splits["validation"], device, batch_size=batch_size if batched else None,
                             plans=plans.get("validation"), integrator=integrator)
        score = validation["joint_nll"]
        if not math.isfinite(score):
            raise FloatingPointError("nonfinite validation likelihood")
        improved = score < best
        if improved:
            best = score
        if score < stopping_best-config.get("early_stopping_min_delta", 0.):
            stopping_best, stale = score, 0
        else:
            stale += 1
        used_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step() if scheduler_type == 'cosine' else scheduler.step(score)
        entry = {"epoch": epoch, "training_nll": joint_sum/count, 'training_objective': train_sum/count,
                 'training_curve_mse': curve_sum/count if curve_weight else None, "validation": validation,
                 "seconds": time.perf_counter()-began, "best": improved, "learning_rate": used_lr,
                 "optimizer_updates": updates, "samples_this_epoch": count, "stale_epochs": stale}
        history.append(entry)
        bundle = {"format": FORMAT, "version": VERSION, "event": EVENT, "kernel": SE_KERNEL,
                  "reference_sampler": model.reference_sampler,
                  "model_config": model.configuration(), "model_state": model.state_dict(), "epoch": epoch,
                  "training_config": config, "dataset_sha256": data_hash, "validation": validation,
                  "initialization": initialization, "optimizer_updates": updates}
        if improved:
            torch.save(bundle, output/"best.pt")
        torch.save({**bundle, "optimizer_state": optimizer.state_dict(), "history": history,
                    "best": best, "stale": stale, "stopping_best": stopping_best,
                    "scheduler_state": scheduler.state_dict() if scheduler else None}, output/"last.pt")
        write_json(output/"history.json", history)
        print(f"{model.config.architecture} epoch {epoch}: train={train_sum/count:.6f} validation={score:.6f} lr={used_lr:.2g} updates={updates} stale={stale} seconds={entry['seconds']:.2f}", flush=True)
        if epoch >= config.get("minimum_epochs", 1) and stale >= config.get("patience", 8):
            stopped = "validation_plateau"
            break
    # Keep the parameter objects referenced by the captured solver.
    selected = torch.load(output/"best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(selected["model_state"])
    if batched:
        plans["test"] = [make_plan(p["features"], model.config.maximum_step, device) for p in splits["test"]]
        maximum = max(len(p["dt"]) for split in plans.values() for p in split)
        if integrator is not None and maximum > integrator.maximum_steps:
            integrator = CudaIntegrator(model, maximum, batch_size)
    report = {"architecture": model.config.architecture, "dataset_sha256": data_hash,
              "reference_sampler": model.reference_sampler,
              "selected_epoch": selected["epoch"], "parameters": sum(p.numel() for p in model.parameters()),
              "completed_epochs": history[-1]["epoch"], "optimizer_updates": updates,
              "stop_reason": stopped, "training_seconds": sum(h["seconds"] for h in history),
              "initialization": initialization,
              "metrics": {s: metrics(model, p, device, batch_size=batch_size if batched else None,
                                     plans=plans.get(s), integrator=integrator) for s, p in splits.items() if p}}
    write_json(output/"run_report.json", report)
    return {"checkpoint": str(output/"best.pt"), **report}


def export_model(checkpoint, output):
    model, bundle = load_model(checkpoint)
    def tensors(module):
        return {name: {"shape": list(value.shape), "values": value.detach().float().reshape(-1).tolist()}
                for name, value in module.state_dict().items()}
    root = {"format": FORMAT, "version": VERSION, "event": EVENT, "kernel": SE_KERNEL,
            "reference_sampler": model.reference_sampler,
            "distance_convention": "q=distance_from_entry/ell", "mean_convention": "b=m/sigma",
            "branch_order": [1, -1], "gradient_convention": "signed_derivative=-initial_sign*sigma/ell*W",
            "profile_mode": "cubic", "activation_dtype": "float32",
            "solver": "analytic_rqs" if model.config.architecture == "spline" else "rk4",
            "checkpoint_sha256": digest(checkpoint), "model_config": model.configuration()}
    if model.config.architecture == "spline":
        root["fixed_profile_convention"] = "uniform_q_over_Q_asinh_signed_b_db_u_log1p_Q"
    if model.config.architecture == "gru":
        root["gru"] = {"format": "macrofacet.renewal", "version": 1, "kernel": SE_KERNEL,
            "activation_dtype": "float32", "gru_convention": "pytorch_rzn_reset_after",
            "feature_transform": "asinh_first_four_log_dx_identity",
            "initial_transform": "asinh_mode_b0_known_z0_known_d0",
            "mixture_convention": "positive_truncated_components_residual_mean",
            "checkpoint_sha256": root["checkpoint_sha256"], "model_config": model.gru_model.configuration(),
            "weights": tensors(model.gru_model)}
    else:
        root["weights"] = tensors(model)
    write_json(Path(output), root)
    return {"bundle": str(output), "architecture": model.config.architecture}
