"""Train the joint censored distance/speed likelihood; select using validation only."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from . import FORMAT_VERSION, validate_kernel
from .collect import digest, resolve, write_json
from .data import SequenceDataset, collate, load_data, to_device
from .evaluate import likelihood, evaluate
from .model import ModelConfig, RenewalNetwork


def train(config_path: Path, resume: bool = False):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != FORMAT_VERSION:
        raise ValueError("unsupported training schema")
    torch.set_num_threads(config.get("cpu_threads", 4))
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable")
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dataset_path = resolve(config["dataset"])
    dataset_hash = digest(dataset_path)
    data = load_data(dataset_path)
    if "kernel" in config and validate_kernel(config["kernel"]) != data["kernel"]:
        raise ValueError("training kernel differs from dataset kernel")
    output = resolve(config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and not resume:
        raise ValueError("training output exists; pass --resume or choose a new output directory")
    model = RenewalNetwork(ModelConfig(**config.get("model", {}))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    train_set = SequenceDataset(data, "train")
    validation = DataLoader(SequenceDataset(data, "validation"), batch_size=config["batch_size"],
                            collate_fn=collate, num_workers=0)
    start_epoch, best, stale, history = 1, math.inf, 0, []
    if resume:
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if checkpoint["dataset_sha256"] != dataset_hash or checkpoint["training_config"] != config:
            raise ValueError("resume requires exactly the same dataset and training configuration")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"]+1
        best, stale, history = checkpoint["best_validation"], checkpoint["stale"], checkpoint["history"]
    else:
        initial = likelihood(model, validation, device)
        write_json(output / "initial_validation.json", initial)
        print(f"initial validation joint NLL={initial['all']['joint_nll']:.6f}", flush=True)
    write_json(output / "training_config.json", config)
    write_json(output / "environment.json", {"torch": str(torch.__version__), "device": device,
        "device_name": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else "CPU",
        "parameter_count": sum(p.numel() for p in model.parameters()), "dataset_sha256": dataset_hash})
    for epoch in range(start_epoch, config["epochs"]+1):
        began = time.perf_counter()
        # Epoch-specific streams make resume independent of checkpoint timing.
        torch.manual_seed(config["seed"]+epoch)
        generator = torch.Generator().manual_seed(config["seed"]+epoch)
        loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True,
                            generator=generator, collate_fn=collate, num_workers=0)
        model.train()
        total, count = 0.0, 0
        for step, original in enumerate(loader, 1):
            batch = to_device(original, device)
            optimizer.zero_grad(set_to_none=True)
            losses = model.losses(batch)["joint"]
            loss = losses.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss at epoch {epoch}, batch {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(losses)
            count += len(losses)
            if step % 200 == 0:
                print(f"epoch {epoch} batch {step}/{len(loader)} joint NLL={total/count:.6f}", flush=True)
        metrics = likelihood(model, validation, device)
        value = metrics["all"]["joint_nll"]
        if not math.isfinite(value):
            raise FloatingPointError("nonfinite validation likelihood")
        improved = value < best
        if improved:
            best, stale = value, 0
        else:
            stale += 1
        entry = {"epoch": epoch, "train_joint_nll": total/count, "validation": metrics,
                 "seconds": time.perf_counter()-began, "best": improved}
        history.append(entry)
        bundle = {"format_version": FORMAT_VERSION, "kernel": data["kernel"], "model_config": model.configuration(),
                  "model_state": model.state_dict(), "epoch": epoch, "dataset_sha256": dataset_hash,
                  "training_config": config, "validation": metrics}
        if improved:
            torch.save(bundle, output / "best.pt")
        torch.save({**bundle, "optimizer_state": optimizer.state_dict(), "best_validation": best,
                    "stale": stale, "history": history}, output / "last.pt")
        write_json(output / "history.json", history)
        print(f"epoch {epoch}: train={total/count:.6f} val={value:.6f} "
              f"seconds={entry['seconds']:.1f} best={improved}", flush=True)
        if stale >= config["patience"]:
            break
    # Test data are never used for optimizer steps, early stopping or selection.
    for split in ("validation", "test"):
        evaluate(output / "best.pt", dataset_path, output / f"{split}_metrics.json", split,
                 device, config["batch_size"])
    return output / "best.pt"
