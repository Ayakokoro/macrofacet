"""Batch independent rays while retaining each ray's original cubic/RK4 grid.

Only geometry is cached. Hidden states and hazards are rebuilt with autograd on
every call; neither history nor gradients are detached between solver steps.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from renewal.model import mean_at, cumulative, log_hazard
from .model import validate_profile, combine, log_softplus, losses_from_prediction


def encoded(features, indices, positions, left):
    selected = features[indices]
    width = selected[..., 4].exp()
    u = ((positions-left[indices])/width).clamp(0, 1)
    b, db = mean_at(selected, u)
    signs = features.new_tensor([1., -1.])
    result = torch.stack((torch.asinh(b[..., None]*signs), torch.asinh(db[..., None]*signs),
                          positions.log1p()[..., None].expand(*b.shape, 2)), -1)
    return result, db[..., None]*signs


def make_plan(features, maximum_step, device="cpu"):
    features = features.detach().cpu().double()
    validate_profile(features)
    widths = features[:, 4].exp().tolist()
    starts, qleft, qright, indices = [], [], [], []
    left = 0.
    for i, width in enumerate(widths):
        starts.append(left)
        end, q = left+width, left
        while q < end:
            right = min(q+maximum_step, end)
            if right <= q:
                raise ValueError("integration step is below distance roundoff")
            qleft.append(q); qright.append(right); indices.append(i)
            q = right
        left = end
    lo, hi, start = (features.new_tensor(x) for x in (qleft, qright, starts))
    index = torch.tensor(indices)
    locations = torch.stack((lo, (lo+hi)/2, hi), -1)
    values, _ = encoded(features, index[:, None].expand(-1, 3), locations, start)
    plan = {"features": features, "left": start, "node_left": lo, "right": hi,
            "index": index, "dt": hi-lo, "inputs": values}
    # CPU geometry also prepares variable query locations without GPU round trips.
    return {**plan, "device_inputs": values.to(device), "device_dt": (hi-lo).to(device)}


def rk4(model, h, H, inputs, dt):
    def rhs(state, u):
        x = torch.cat((state, u), -1)
        return model.dynamics(x), F.softplus(model.hazard_head(x).squeeze(-1)).double()
    dh = dt.to(h.dtype)[:, None]
    k1, a1 = rhs(h, inputs[0])
    k2, a2 = rhs(h+(dh/2)*k1, inputs[1])
    k3, a3 = rhs(h+(dh/2)*k2, inputs[1])
    k4, a4 = rhs(h+dh*k3, inputs[2])
    return h+(dh/6)*(k1+2*k2+2*k3+k4), H+(dt/6)*(a1+2*a2+2*a3+a4)


class RNOIntegrator(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.initial_encoder = model.initial_encoder
        self.dynamics = model.dynamics
        self.hazard_head = model.hazard_head

    def forward(self, stage, dt):
        h = self.initial_encoder(stage[0, 0, :, :2])
        H = torch.zeros(len(h), dtype=torch.float64, device=h.device)
        states, hazards = [h], [H]
        for step in range(len(dt)):
            h, H = rk4(self, h, H, stage[step], dt[step])
            states.append(h); hazards.append(H)
        return torch.stack(states), torch.stack(hazards)


class CudaIntegrator:
    """Replay the unchanged solver/autograd graph; only static shapes are padded."""
    def __init__(self, model, maximum_steps, batch_size):
        parameter = next(model.parameters())
        if parameter.device.type != "cuda":
            raise ValueError("CUDA solver capture requires a CUDA model")
        self.maximum_steps, self.batch_size = maximum_steps, batch_size
        self.module = RNOIntegrator(model)
        sample = (torch.zeros((maximum_steps, 3, 2*batch_size, 3), device=parameter.device, dtype=parameter.dtype),
                  torch.zeros((maximum_steps, 2*batch_size), dtype=torch.float64, device=parameter.device))
        self.module = torch.cuda.make_graphed_callables(self.module, sample, num_warmup_iters=2)

    def __call__(self, stage, dt):
        if len(dt) > self.maximum_steps or dt.shape[1] > 2*self.batch_size:
            raise ValueError("batch exceeds the captured solver shape")
        values = stage.new_zeros((self.maximum_steps, 3, 2*self.batch_size, 3))
        steps = dt.new_zeros((self.maximum_steps, 2*self.batch_size))
        values[:len(stage), :, :stage.shape[2]] = stage
        steps[:len(dt), :dt.shape[1]] = dt
        return self.module(values, steps)


def query_many(model, features, queries, *, plans=None, with_mixture=True, integrator=None,
               with_mixture_context=False):
    if len(features) != len(queries) or not features:
        raise ValueError("one query vector is required per profile")
    if model.config.architecture == "spline":
        if with_mixture_context:
            raise ValueError("RNO mixture-context caching is not used by spline models")
        return [model.query(f, q, with_mixture=with_mixture) for f, q in zip(features, queries)]
    if with_mixture_context and model.config.architecture != 'rno':
        raise ValueError('cached mixture context currently requires an RNO')
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if plans is None:
        plans = [make_plan(f, model.config.maximum_step, device) for f in features]
    if len(plans) != len(features):
        raise ValueError("one geometry plan is required per profile")
    query_ids, node_ids, segment_ids, local_u, partial_inputs, partial_dt, dbs, counts = [], [], [], [], [], [], [], []
    for i, (plan, original) in enumerate(zip(plans, queries)):
        q = original.detach().cpu().double()
        if q.ndim != 1 or not len(q) or not torch.isfinite(q).all() or (q < 0).any() or (q > plan["right"][-1]+1e-10).any():
            raise ValueError("invalid batched query locations")
        q = q.clamp_max(plan["right"][-1])
        node = torch.searchsorted(plan["right"], q).clamp_max(len(plan["right"])-1)
        segment = plan["index"][node]
        lo = plan["node_left"][node]
        locations = torch.stack((lo, (lo+q)/2, q), -1)
        values, db = encoded(plan["features"], segment[:, None].expand(-1, 3), locations, plan["left"])
        query_ids.append(torch.full((len(q),), i, dtype=torch.long)); node_ids.append(node)
        segment_ids.append(segment)
        local_u.append(((q-plan["left"][segment])/plan["features"][segment, 4].exp()).clamp(0, 1))
        partial_inputs.append(values); partial_dt.append(q-lo); dbs.append(db[:, -1]); counts.append(len(q))
    query_ids, node_ids, segment_ids = (torch.cat(v) for v in (query_ids, node_ids, segment_ids))
    branch_ids = (2*query_ids[:, None]+torch.arange(2)[None]).to(device)
    partial = torch.cat(partial_inputs).permute(1, 0, 2, 3).reshape(3, -1, 3).to(device=device, dtype=dtype)
    db = torch.cat(dbs).to(device)
    if model.config.architecture == "rno":
        stage = pad_sequence([p["device_inputs"] for p in plans], batch_first=False)
        stage = stage.permute(0, 2, 1, 3, 4).reshape(stage.shape[0], 3, -1, 3).to(dtype)
        dt = pad_sequence([p["device_dt"] for p in plans], batch_first=False).repeat_interleave(2, dim=1)
        required = int(node_ids.max())
        # A zero-step batch still needs the initial encoder's inputs.
        if integrator is not None:
            states, hazards = integrator(stage, dt)
        else:
            states, hazards = RNOIntegrator(model)(stage[:max(1, required)], dt[:required])
        nodes = node_ids[:, None].expand(-1, 2).to(device)
        h = states[nodes, branch_ids].reshape(-1, model.config.hidden)
        H = hazards[nodes, branch_ids].reshape(-1)
        h, H = rk4(model, h, H, partial, torch.cat(partial_dt).repeat_interleave(2).to(device))
        x = torch.cat((h, partial[-1]), -1)
        result = {"H": H.reshape(-1, 2), "log_hazard": log_softplus(model.hazard_head(x).squeeze(-1)).double().reshape(-1, 2)}
        if with_mixture_context:
            result.update(mixture_inputs=x.reshape(-1, 2, model.config.hidden+3),
                          mixture_derivative=db)
        if with_mixture:
            weights, delta, scale = model.mixture_head(x).reshape(-1, 2, 3*model.config.components).chunk(3, -1)
            result.update(log_weights=F.log_softmax(weights, -1), means=-db[..., None]+delta,
                          scales=F.softplus(scale)+model.config.sigma_floor)
    else:
        lengths = torch.tensor([len(f) for f in features]).repeat_interleave(2)
        fs = pad_sequence([p["features"] for p in plans], batch_first=True).to(device)
        signs = fs.new_tensor([1., -1.])
        both = fs[:, None].expand(-1, 2, -1, -1)
        both = torch.cat((both[..., :4]*signs[None, :, None, None], both[..., 4:]), -1).reshape(-1, fs.shape[1], 5)
        initial = fs.new_zeros((len(features)*2, 4)); initial[:, 1] = both[:, 0, 0]
        context, logits = model.gru_model.encode(both, initial, lengths)
        widths = both[..., 4].exp()
        mass = widths*F.softplus(logits).sum(-1)/4
        prefix = F.pad(mass.cumsum(-1), (1, 0))[:, :-1]
        index = segment_ids[:, None].expand(-1, 2).to(device)
        u = torch.cat(local_u).to(device)[:, None].expand(-1, 2)
        rates = logits[branch_ids, index]
        result = {"H": prefix[branch_ids, index]+cumulative(rates, widths[branch_ids, index], u),
                  "log_hazard": log_hazard(rates, u).double()}
        if with_mixture:
            mix = model.gru_model.mixture(context[branch_ids, index], both[branch_ids, index], u)
            result.update(zip(("log_weights", "means", "scales"), mix))
    outputs, offset = [], 0
    for f, count in zip(features, counts):
        outputs.append(combine(f.to(device), {k: v[offset:offset+count].transpose(0, 1) for k, v in result.items()}))
        offset += count
    return outputs


def losses_many(model, profiles, plans=None, integrator=None):
    # Curve queries share the same solver replay as event queries. A second
    # replay before backward would overwrite the captured solver's live state.
    queries = [torch.cat((p['distance'], p['curve_queries'])) if 'curve_queries' in p else p['distance']
               for p in profiles]
    predictions = query_many(model, [p["features"] for p in profiles], queries, plans=plans, integrator=integrator)
    results = []
    for r, p in zip(predictions, profiles):
        count = len(p['distance'])
        event = {k: v if k == 'log_prior' else v[:, :count] if v.ndim >= 2 else v[:count]
                 for k, v in r.items()}
        result = losses_from_prediction(event, p['distance'], p['hit'], p['initial_sign'], p['speed'])
        if 'curve_queries' in p:
            result['curve_prediction'] = r['T'][count:]
        results.append(result)
    return results
