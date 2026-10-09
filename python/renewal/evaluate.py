"""Held-out likelihood, survival calibration and conditional speed PIT diagnostics."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from . import FORMAT_VERSION, KERNEL
from .data import SPLITS, SequenceDataset, collate, load_data, to_device
from .model import ModelConfig, RenewalNetwork, cumulative
from .collect import digest, write_json


def load_model(path: Path, device):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("format_version") != FORMAT_VERSION or bundle.get("kernel") != KERNEL:
        raise ValueError("unsupported network checkpoint or kernel convention")
    model = RenewalNetwork(ModelConfig(**bundle["model_config"])).to(device)
    model.load_state_dict(bundle["model_state"])
    return model, bundle


def _ks_uniform(pit):
    if not len(pit):
        return None
    values = pit.sort().values.double()
    n = len(values)
    upper = torch.arange(1, n+1, dtype=torch.float64)/n
    return max(float((upper-values).max()), float((values-(upper-1/n)).max()))


@torch.no_grad()
def likelihood(model, loader, device, data=None):
    model.eval()
    all_values = {k: [] for k in ("joint", "distance", "speed", "pit", "hit", "mode", "profile")}
    for original in loader:
        batch = to_device(original, device)
        values = model.losses(batch)
        for key in ("joint", "distance", "speed", "pit"):
            all_values[key].append(values[key].detach().cpu())
        all_values["hit"].append(original["hit"])
        all_values["mode"].append(original["initial"][:, 0])
        all_values["profile"].append(original["profile"])
    if not all_values["joint"]:
        raise ValueError("empty evaluation split")
    v = {k: torch.cat(value) for k, value in all_values.items()}
    result = {}
    groups = [("all", torch.ones_like(v["hit"])), ("A", v["mode"] == 0), ("B", v["mode"] == 1)]
    if data is not None:
        scenes = [data["scene_ids"][int(p)] for p in v["profile"]]
        groups.extend((f"scene:{scene}", torch.tensor([s == scene for s in scenes])) for scene in sorted(set(scenes)))
    for name, mask in groups:
        if not mask.any():
            continue
        events = mask & v["hit"]
        result[name] = {"samples": int(mask.sum()), "hits": int(events.sum()),
                       "joint_nll": float(v["joint"][mask].double().mean()),
                       "distance_nll": float(v["distance"][mask].double().mean()),
                       "speed_nll_on_hits": float(v["speed"][events].double().mean()) if events.any() else None,
                       "speed_pit_ks": _ks_uniform(v["pit"][events])}
    return result


@torch.no_grad()
def survival_calibration(model, data, split, device, batch_size=128):
    profiles = torch.where(data["split"] == SPLITS[split])[0]
    fractions = torch.tensor([0.25, 0.5, 0.75, 1.0], device=device)
    predicted = torch.zeros((len(data["initial"]), 4), dtype=data["features"].dtype)
    for begin in range(0, len(profiles), batch_size):
        selected = profiles[begin:begin+batch_size]
        sequences = [data["features"][data["offsets"][p]:data["offsets"][p+1]] for p in selected]
        lengths = torch.tensor([len(f) for f in sequences])
        features = pad_sequence(sequences, batch_first=True).to(device)
        _, logits = model.encode(features, data["initial"][selected].to(device), lengths)
        valid = torch.arange(features.shape[1], device=device)[None, :] < lengths.to(device)[:, None]
        dx = features[..., 4].exp() * valid
        ends = dx.cumsum(-1)
        totals = dx * F.softplus(logits).sum(-1)/4
        H_before = F.pad(totals.cumsum(-1), (1, 0))[:, :-1]
        rows = torch.arange(len(selected), device=device)
        for j, fraction in enumerate(fractions):
            x = data["horizon"][selected].to(device) * fraction
            segment = ((ends < x[:, None]) & valid).sum(-1)
            segment = torch.minimum(segment, lengths.to(device)-1)
            left = ends[rows, segment]-dx[rows, segment]
            u = ((x-left)/dx[rows, segment]).clamp(0, 1)
            H = H_before[rows, segment] + cumulative(logits[rows, segment], dx[rows, segment], u)
            predicted[selected, j] = torch.exp(-H).cpu()
    p = data["sample_profile"]
    split_mask = data["split"][p] == SPLITS[split]
    report = {}
    for name, mask in (("all", split_mask), ("A", split_mask & (data["initial"][p, 0] == 0)),
                       ("B", split_mask & (data["initial"][p, 0] == 1))):
        if not mask.any():
            continue
        selected = p[mask]
        rows = []
        for j, fraction in enumerate(fractions.cpu()):
            observed = (~data["hit"][mask]) | (data["distance"][mask] > data["horizon"][selected] * fraction)
            prediction = predicted[selected, j]
            rows.append({"horizon_fraction": float(fraction), "predicted_survival": float(prediction.mean()),
                         "observed_survival": float(observed.float().mean()),
                         "absolute_gap": float(abs(prediction.mean()-observed.float().mean())),
                         "brier_score": float(((prediction-observed.float())**2).mean())})
        report[name] = rows
    return report


def evaluate(checkpoint: Path, dataset: Path, output: Path, split="test", device="cpu", batch_size=128):
    data = load_data(dataset)
    model, bundle = load_model(checkpoint, device)
    loader = DataLoader(SequenceDataset(data, split), batch_size=batch_size, collate_fn=collate)
    metrics = {"split": split, "checkpoint_sha256": digest(checkpoint), "dataset_sha256": digest(dataset),
               "selected_epoch": bundle["epoch"], "likelihood": likelihood(model, loader, device, data),
               "survival": survival_calibration(model, data, split, device, batch_size)}
    write_json(output, metrics)
    return metrics
