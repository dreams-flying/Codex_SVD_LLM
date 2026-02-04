#coding:utf8
import os
import sys
import argparse
import heapq
import torch.jit
from tqdm import tqdm
import torch
import torch.nn as nn

from utils.data_utils import *
from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
from component.svd_opt import SVDOPTDecoderLayer
from utils.model_utils import *
from evaluater import * 

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)



@torch.no_grad()
def profle_svdllm(name, model, calib_loader, dev):
    if "llama" in name or "mistral" in name or "vicuna" in name:
        layers = model.model.layers
    elif "opt" in name:
        layers = model.model.decoder.layers
    model = model.to(dev)
    print("Start obtaining the whitening matrix...")
    def hook(module, input, output):
        inp = input[0].detach().float()
        if inp.dim() == 2:   # for opt
            inp = inp.unsqueeze(0)
        adds = torch.matmul(inp.transpose(1,2), inp)
        adds_sum = torch.sum(adds, dim=0)
        module.raw_scaling_diag_matrix += adds_sum
        del inp, adds, adds_sum
        torch.cuda.empty_cache()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.raw_scaling_diag_matrix = 0
            module.register_forward_hook(hook)
    for batch in tqdm(calib_loader):
        batch = {k: v.to(dev) for k, v in batch.items()}
        model(**batch)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
    torch.cuda.empty_cache()
    model = model.cpu()
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        for name in subset:
            subset[name].raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.cpu()
    profiling_mat = {}
    print("Start Cholesky Decomposition...")
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        subset = find_layers(layers[i])
        for name in subset:
            raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.double().to(dev)
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix.cpu()
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        profiling_mat[i] = layer_profile
    return profiling_mat
        

@torch.no_grad()
def profle_svdllm_low_resource(model_name, model, calib_loader, dev):
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask'].cpu()
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids'].cpu()
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask'].cpu()), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids'].cpu()), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:  
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']
    profiling_mat = {}
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        layer = layers[i].to(dev)
        subset = find_layers(layer)        
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:  # for opt
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
        for j in range(inps.shape[0]):
            if "opt" not in model_name:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev), position_ids=position_ids[j].unsqueeze(0).to(dev))[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev))[0]
        for h in handles:
            h.remove()
        layer = layer.cpu()
        for name in subset:
            subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
        torch.cuda.empty_cache()
        for name in subset:
            raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().to(dev)
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix.cpu()
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        layers[i] = layer.cpu()
        profiling_mat[i] = layer_profile
        inps = outs
        torch.cuda.empty_cache()
    return profiling_mat


def _safe_cholesky(mat, eps=1e-6):
    try:
        return torch.linalg.cholesky(mat)
    except Exception:
        eig = torch.linalg.eigvalsh(mat)
        mat = mat + (-eig[0] + eps) * torch.eye(mat.shape[0], device=mat.device, dtype=mat.dtype)
        return torch.linalg.cholesky(mat)


def _get_block_size(name, model, attn_block_size, mlp_block_size):
    hidden_size = model.config.hidden_size
    num_heads = getattr(model.config, "num_attention_heads", None)
    head_dim = hidden_size // num_heads if num_heads else None
    if any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj", "out_proj")):
        if attn_block_size and attn_block_size > 0:
            return attn_block_size
        return head_dim if head_dim else 0
    if any(k in name for k in ("gate_proj", "up_proj", "down_proj", "fc1", "fc2")):
        if mlp_block_size and mlp_block_size > 0:
            return mlp_block_size
        return hidden_size
    return 0


def profile_grad_diag(model_name, model, calib_loader, dev, max_batches=None, grad_eps=1e-6, block_diag=False, attn_block_size=0, mlp_block_size=0, use_checkpointing=False):
    if "llama" in model_name or "mistral" in model_name or "vicuna" in model_name:
        layers = model.model.layers
    elif "opt" in model_name:
        layers = model.model.decoder.layers
    model = model.to(dev)
    prev_training = model.training
    prev_use_cache = getattr(model.config, "use_cache", None)
    prev_dropout_cfg = {}
    if prev_use_cache is not None:
        model.config.use_cache = False
    prev_grad_ckpt = getattr(model, "is_gradient_checkpointing", False) or getattr(model, "gradient_checkpointing", False)
    if use_checkpointing:
        for key in ("attention_dropout", "hidden_dropout", "dropout", "activation_dropout", "classifier_dropout"):
            if hasattr(model.config, key):
                prev_dropout_cfg[key] = getattr(model.config, key)
                setattr(model.config, key, 0.0)
    if use_checkpointing and hasattr(model, "gradient_checkpointing_enable") and not prev_grad_ckpt:
        model.gradient_checkpointing_enable()
    if use_checkpointing:
        model.train()
    else:
        model.eval()
    prev_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(True)

    handles = []
    def hook(module, grad_input, grad_output):
        if grad_output is None or grad_output[0] is None:
            return
        gout = grad_output[0].detach()
        if gout.dim() == 2:  # for opt
            gout = gout.unsqueeze(0)
        # gout: [bs, seq, out]
        if getattr(module, "grad_block_ranges", None) is not None:
            gout2d = gout.reshape(-1, gout.shape[-1])
            n = gout2d.shape[0]
            module.grad_blocks_count += n
            for bi, (s, e) in enumerate(module.grad_block_ranges):
                block = gout2d[:, s:e]
                module.grad_blocks_sum[bi] += block.t().matmul(block)
        else:
            sumsq = gout.pow(2).sum(dim=(0, 1))
            module.grad_squares_sum += sumsq
            module.grad_squares_count += gout.shape[0] * gout.shape[1]

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if block_diag:
                block_size = _get_block_size(name, model, attn_block_size, mlp_block_size)
                out_dim = module.weight.shape[0]
                if block_size and block_size > 0:
                    ranges = []
                    for s in range(0, out_dim, block_size):
                        e = min(s + block_size, out_dim)
                        ranges.append((s, e))
                    module.grad_block_ranges = ranges
                    module.grad_blocks_sum = [
                        torch.zeros((e - s, e - s), device=dev) for (s, e) in ranges
                    ]
                    module.grad_blocks_count = 0
                else:
                    module.grad_block_ranges = None
                    module.grad_squares_sum = torch.zeros(out_dim, device=dev)
                    module.grad_squares_count = 0
            else:
                module.grad_block_ranges = None
                module.grad_squares_sum = torch.zeros(module.weight.shape[0], device=dev)
                module.grad_squares_count = 0
            handles.append(module.register_full_backward_hook(hook))

    for i, batch in enumerate(tqdm(calib_loader)):
        if max_batches is not None and i >= max_batches:
            break
        model.zero_grad(set_to_none=True)
        batch = {k: v.to(dev) for k, v in batch.items()}
        labels = batch["input_ids"]
        outputs = model(**batch, labels=labels, use_cache=False, output_attentions=False, output_hidden_states=False)
        loss = outputs.loss
        loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    grad_diag = {}
    for i in range(len(layers)):
        layer_profile = {}
        subset = find_layers(layers[i])
        for name in subset:
            module = subset[name]
            if getattr(module, "grad_block_ranges", None) is not None and module.grad_block_ranges:
                count = max(module.grad_blocks_count, 1)
                blocks = []
                for bi, (s, e) in enumerate(module.grad_block_ranges):
                    mat = module.grad_blocks_sum[bi] / count
                    if grad_eps:
                        mat = mat + grad_eps * torch.eye(mat.shape[0], device=mat.device, dtype=mat.dtype)
                    blocks.append({"start": s, "end": e, "mat": mat.cpu()})
                layer_profile[name] = {"type": "block", "blocks": blocks}
                del module.grad_blocks_sum
                del module.grad_blocks_count
            else:
                count = max(module.grad_squares_count, 1)
                diag = (module.grad_squares_sum / count).clamp_min(grad_eps)
                layer_profile[name] = diag.cpu()
                del module.grad_squares_sum
                del module.grad_squares_count
        grad_diag[i] = layer_profile
    model = model.cpu()
    if use_checkpointing and hasattr(model, "gradient_checkpointing_disable") and not prev_grad_ckpt:
        model.gradient_checkpointing_disable()
    for key, value in prev_dropout_cfg.items():
        setattr(model.config, key, value)
    if prev_use_cache is not None:
        model.config.use_cache = prev_use_cache
    model.train(prev_training)
    torch.set_grad_enabled(prev_grad)
    return grad_diag


def compute_layer_ratios(model_name, model, grad_diag, base_ratio, min_ratio=0.01, max_ratio=0.99, eps=1e-6):
    if "llama" in model_name or "mistral" in model_name or "vicuna" in model_name:
        layers = model.model.layers
    elif "opt" in model_name:
        layers = model.model.decoder.layers
    else:
        return None
    layer_sizes = []
    layer_scores = []
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        size_i = 0
        score_i = 0.0
        for name in subset:
            W = subset[name].weight
            size_i += W.shape[0] * W.shape[1]
            if grad_diag is not None and i in grad_diag and name in grad_diag[i]:
                entry = grad_diag[i][name]
                if isinstance(entry, dict) and entry.get("type") == "block":
                    for b in entry["blocks"]:
                        mat = b["mat"]
                        score_i += float(torch.trace(mat).item())
                else:
                    score_i += float(entry.sum().item())
        if score_i <= 0:
            score_i = eps
        layer_sizes.append(size_i)
        layer_scores.append(score_i)
    total_size = sum(layer_sizes)
    if total_size == 0:
        return None
    weighted_mean = sum(s * w for s, w in zip(layer_sizes, layer_scores)) / total_size
    ratios = [base_ratio * (w / weighted_mean) for w in layer_scores]
    target = base_ratio * total_size

    # Iteratively clamp and rescale to hit the global budget.
    for _ in range(5):
        fixed = []
        free = []
        for i, r in enumerate(ratios):
            if r < min_ratio:
                ratios[i] = min_ratio
                fixed.append(i)
            elif r > max_ratio:
                ratios[i] = max_ratio
                fixed.append(i)
            else:
                free.append(i)
        fixed_size = sum(layer_sizes[i] for i in fixed)
        free_size = sum(layer_sizes[i] for i in free)
        if free_size <= 0:
            break
        fixed_budget = sum(layer_sizes[i] * ratios[i] for i in fixed)
        remain = target - fixed_budget
        if remain <= 0:
            break
        scale = remain / sum(layer_sizes[i] * ratios[i] for i in free)
        updated = False
        for i in free:
            new_r = ratios[i] * scale
            if new_r < min_ratio:
                ratios[i] = min_ratio
                updated = True
            elif new_r > max_ratio:
                ratios[i] = max_ratio
                updated = True
            else:
                ratios[i] = new_r
        if not updated:
            break
    return ratios


@torch.no_grad()
def profile_module_spectrum(model_name, model, profiling_mat, dev, grad_diag=None, grad_eps=1e-6):
    if "llama" in model_name or "mistral" in model_name or "vicuna" in model_name:
        layers = model.model.layers
    elif "opt" in model_name:
        layers = model.model.decoder.layers
    else:
        return None
    spectra = {}
    model.eval()
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        layer_spec = {}
        for name in subset:
            W = subset[name].weight.data.float().to(dev)
            if grad_diag is not None and i in grad_diag and name in grad_diag[i]:
                entry = grad_diag[i][name]
                if isinstance(entry, dict) and entry.get("type") == "block":
                    for b in entry["blocks"]:
                        s, e = b["start"], b["end"]
                        mat = b["mat"].to(dev)
                        chol = _safe_cholesky(mat, grad_eps)
                        W[s:e, :] = chol.matmul(W[s:e, :])
                else:
                    g_diag = entry.to(dev)
                    g_sqrt = torch.sqrt(torch.clamp(g_diag, min=grad_eps))
                    W = W * g_sqrt.unsqueeze(1)
            if profiling_mat is not None:
                scaling_diag_matrix = profiling_mat[i][name].to(dev).float()
                W_scale = torch.matmul(W, scaling_diag_matrix)
            else:
                W_scale = W
            try:
                svals = torch.linalg.svdvals(W_scale)
            except Exception:
                _, svals, _ = torch.linalg.svd(W_scale, full_matrices=False)
            layer_spec[name] = {
                "s2": (svals * svals).cpu(),
                "shape": tuple(W.shape),
            }
            W = W_scale = svals = None
            del W, W_scale, svals
        spectra[i] = layer_spec
        torch.cuda.empty_cache()
    return spectra


def allocate_module_ranks(spectra, target_ratio, min_rank=1, max_rank=None, eps=1e-12):
    if spectra is None:
        return None, 0.0
    entries = []
    total_full = 0
    min_total = 0
    max_score = 0.0
    for layer_id in spectra:
        for name, info in spectra[layer_id].items():
            m, n = info["shape"]
            s2 = info["s2"]
            k_max = int(min(m, n, s2.shape[0]))
            if k_max <= 0:
                continue
            cost = m + n
            total_full += m * n
            r_min = min_rank if min_rank is not None else 1
            r_min = max(1, min(r_min, k_max))
            min_total += r_min * cost
            if s2.numel() > 0:
                max_score = max(max_score, float((s2[0] / cost).item()))
            entries.append({
                "layer": layer_id,
                "name": name,
                "s2": s2,
                "cost": cost,
                "k_max": k_max,
                "r_min": r_min,
            })
    if total_full <= 0:
        return None, 0.0
    target_params = target_ratio * total_full
    if min_total > target_params + eps:
        print(f"Warning: min_rank budget {min_total} exceeds target params {target_params:.2f}. Using min_rank.")
        ranks = {}
        for entry in entries:
            ranks.setdefault(entry["layer"], {})[entry["name"]] = entry["r_min"]
        effective = min_total / total_full
        return ranks, effective
    low = 0.0
    high = max_score if max_score > 0 else 1.0
    best_ranks = None
    best_total = None
    for _ in range(40):
        mid = (low + high) / 2
        total = 0
        ranks = {}
        for entry in entries:
            s2 = entry["s2"]
            cost = entry["cost"]
            k_max = entry["k_max"]
            r_min = entry["r_min"]
            if s2.numel() == 0:
                k = r_min
            else:
                k = int((s2 / cost >= mid).sum().item())
                k = max(k, r_min)
                if max_rank is not None:
                    k = min(k, max_rank)
                k = min(k, k_max)
            total += k * cost
            ranks.setdefault(entry["layer"], {})[entry["name"]] = k
        if total > target_params:
            low = mid
        else:
            high = mid
            best_ranks = ranks
            best_total = total
    if best_ranks is None:
        return None, 0.0

    # Greedy fill to use remaining budget with highest marginal gains.
    total = best_total
    heap = []
    entry_map = {}
    for entry in entries:
        layer = entry["layer"]
        name = entry["name"]
        s2 = entry["s2"]
        cost = entry["cost"]
        k = best_ranks[layer][name]
        entry_map[(layer, name)] = entry
        if k < entry["k_max"]:
            score = float((s2[k] / cost).item())
            heapq.heappush(heap, (-score, layer, name))
    while heap and total + eps < target_params:
        neg_score, layer, name = heapq.heappop(heap)
        entry = entry_map[(layer, name)]
        cost = entry["cost"]
        if total + cost > target_params + eps:
            break
        k = best_ranks[layer][name] + 1
        if max_rank is not None:
            k = min(k, max_rank)
        k = min(k, entry["k_max"])
        if k == best_ranks[layer][name]:
            continue
        best_ranks[layer][name] = k
        total += cost
        if k < entry["k_max"]:
            next_score = float((entry["s2"][k] / cost).item())
            heapq.heappush(heap, (-next_score, layer, name))
    effective = total / total_full
    return best_ranks, effective


def summarize_module_ranks(spectra, module_ranks):
    if spectra is None or module_ranks is None:
        return 0.0, {}
    total_full = 0
    total_low = 0
    layer_stats = {}
    for layer_id in spectra:
        layer_full = 0
        layer_low = 0
        for name, info in spectra[layer_id].items():
            m, n = info["shape"]
            k = module_ranks.get(layer_id, {}).get(name, 0)
            layer_full += m * n
            layer_low += k * (m + n)
        total_full += layer_full
        total_low += layer_low
        if layer_full > 0:
            layer_stats[layer_id] = layer_low / layer_full
    effective = (total_low / total_full) if total_full > 0 else 0.0
    return effective, layer_stats


def _split_module_ranks(ranks_layer):
    attn_ranks = {}
    mlp_ranks = {}
    if not ranks_layer:
        return attn_ranks, mlp_ranks
    for name, r in ranks_layer.items():
        if "q_proj" in name:
            attn_ranks["q_proj"] = r
        elif "k_proj" in name:
            attn_ranks["k_proj"] = r
        elif "v_proj" in name:
            attn_ranks["v_proj"] = r
        elif "o_proj" in name:
            attn_ranks["o_proj"] = r
        elif "out_proj" in name:
            attn_ranks["out_proj"] = r
        elif "gate_proj" in name:
            mlp_ranks["gate_proj"] = r
        elif "up_proj" in name:
            mlp_ranks["up_proj"] = r
        elif "down_proj" in name:
            mlp_ranks["down_proj"] = r
        elif "fc1" in name:
            mlp_ranks["fc1"] = r
        elif "fc2" in name:
            mlp_ranks["fc2"] = r
    return attn_ranks, mlp_ranks


def _layer_param_sizes(model_name, model):
    if "llama" in model_name or "mistral" in model_name or "vicuna" in model_name:
        layers = model.model.layers
    elif "opt" in model_name:
        layers = model.model.decoder.layers
    else:
        return None
    sizes = []
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        size_i = 0
        for name in subset:
            W = subset[name].weight
            size_i += W.shape[0] * W.shape[1]
        sizes.append(size_i)
    return sizes


def _normalize_layer_ratios(ratios, layer_sizes, target_ratio, eps=1e-12):
    total = sum(layer_sizes)
    if total <= 0:
        return ratios, 0.0
    effective = sum(s * r for s, r in zip(layer_sizes, ratios)) / total
    if abs(effective - target_ratio) <= eps:
        return ratios, effective
    scale = target_ratio / max(effective, eps)
    ratios = [r * scale for r in ratios]
    effective = sum(s * r for s, r in zip(layer_sizes, ratios)) / total
    return ratios, effective


def _default_spectrum_path(save_path, model_name, dataset, nsamples, seed):
    safe_model = model_name.replace("/", "_").replace("-", "_")
    filename = f"{safe_model}_spectrum_{dataset}_{nsamples}_{seed}.pt"
    return os.path.join(save_path, filename)
     
 
@torch.no_grad()
def whitening(model_name, model, profiling_mat, ratio, dev, grad_diag=None, grad_eps=1e-6, layer_ratios=None, module_ranks=None):
    model.eval()
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("Start SVD decomposition after whitening...")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        ratio_i = layer_ratios[i] if layer_ratios is not None else ratio
        ranks_layer = module_ranks[i] if module_ranks is not None and i in module_ranks else None
        attn_ranks, mlp_ranks = _split_module_ranks(ranks_layer)
        subset = find_layers(layer)
        #### Replace Attn, MLP ####
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio, ranks=attn_ranks if attn_ranks else None)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio, ranks=mlp_ranks if mlp_ranks else None)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio, ranks=attn_ranks if attn_ranks else None)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio, ranks=mlp_ranks if mlp_ranks else None)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio, ranks=ranks_layer if ranks_layer else None)
        #### Replace Attn, MLP ####
        for name in subset:
            W = subset[name].weight.data.float().to(dev)
            dtype = W.dtype
            g_inv_sqrt = None
            g_blocks = None
            if grad_diag is not None:
                entry = grad_diag[i][name]
                if isinstance(entry, dict) and entry.get("type") == "block":
                    g_blocks = []
                    for b in entry["blocks"]:
                        s, e = b["start"], b["end"]
                        mat = b["mat"].to(dev)
                        chol = _safe_cholesky(mat, grad_eps)
                        inv_chol = torch.linalg.inv(chol)
                        g_blocks.append({"start": s, "end": e, "g_sqrt": chol, "g_inv_sqrt": inv_chol})
                        W[s:e, :] = chol.matmul(W[s:e, :])
                else:
                    g_diag = entry.to(dev)
                    g_sqrt = torch.sqrt(torch.clamp(g_diag, min=grad_eps))
                    g_inv_sqrt = 1.0 / g_sqrt
                    W = W * g_sqrt.unsqueeze(1)
            scaling_diag_matrix = profiling_mat[i][name].to(dev)
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
            if ranks_layer is not None and name in ranks_layer:
                num_s_after_trunc = int(ranks_layer[name])
            else:
                num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio_i / (W.shape[0] + W.shape[1]))
            num_s_after_trunc = max(1, min(num_s_after_trunc, min(W.shape[0], W.shape[1])))
            truc_s = S[:num_s_after_trunc]
            truc_u = U[:, :num_s_after_trunc]
            if g_blocks is not None:
                for b in g_blocks:
                    s, e = b["start"], b["end"]
                    truc_u[s:e, :] = b["g_inv_sqrt"].matmul(truc_u[s:e, :])
            elif g_inv_sqrt is not None:
                truc_u = g_inv_sqrt.unsqueeze(1) * truc_u
            truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
            truc_sigma = torch.diag(truc_s)
            #### Replace Attn, MLP ####
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
            svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)
            if 'opt' in model_name:
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data  # the linear layer in OPT has bias, which is different from LLaMA and Mistral
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[i] = svd_decoder
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn =  svd_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp
            W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT  = truc_s = truc_u = truc_v = sqrtSigma = None
            del  W, W_scale, scaling_matrix_inv, scaling_diag_matrix, U, S, VT, truc_s, truc_u, truc_v, sqrtSigma
        del layer
        torch.cuda.empty_cache()


@torch.no_grad()
def whitening_local_update(model_name, model, dataloader, profiling_mat, ratio, dev, direct_update=False, grad_diag=None, grad_eps=1e-6, layer_ratios=None, module_ranks=None):
    print("Start SVD decomposition then update...")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(dataloader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids']
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        ratio_i = layer_ratios[i] if layer_ratios is not None else ratio
        ranks_layer = module_ranks[i] if module_ranks is not None and i in module_ranks else None
        attn_ranks, mlp_ranks = _split_module_ranks(ranks_layer)
        subset = find_layers(layer)
        gpts = {}
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio, ranks=attn_ranks if attn_ranks else None)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio, ranks=mlp_ranks if mlp_ranks else None)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio, ranks=attn_ranks if attn_ranks else None)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio, ranks=mlp_ranks if mlp_ranks else None)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio, ranks=ranks_layer if ranks_layer else None)
        for name in subset:
            if profiling_mat is not None:
                scaling_diag_matrix = profiling_mat[i][name].to(dev)
            else: 
                scaling_diag_matrix = None
            if grad_diag is not None:
                g_diag = grad_diag[i][name].to(dev)
            else:
                g_diag = None
            rank_override = ranks_layer[name] if ranks_layer is not None and name in ranks_layer else None
            gpts[name] = local_update(subset[name], scaling_diag_matrix = scaling_diag_matrix, ratio=ratio_i, name=name, direct_update=direct_update, g_diag=g_diag, grad_eps=grad_eps, rank=rank_override)
        
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch_update_u(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        if "opt" not in model_name:
            outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        else:
            outs = layer(inps, attention_mask=attention_masks)[0]
        for h in handles:
            h.remove()
        for name in gpts:
            svd_u, svd_v = gpts[name].fasterprune()
            svd_u, svd_v = svd_u.to(dtype), svd_v.to(dtype)
            if 'opt' in model_name:
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data  # the linear layer in OPT has bias, which is different from LLaMA and Mistral
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[i] = svd_decoder
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn =  svd_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp
        layer = layer.to(dev)
        if "opt" not in model_name:
            outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        else:
            outs = layer(inps, attention_mask=attention_masks)[0]
        layers[i] = layer.cpu()
        del gpts
        torch.cuda.empty_cache()
        inps = outs
        outs = None
        del outs
    model.config.use_cache = use_cache


class local_update:
    def __init__(self, layer, scaling_diag_matrix, ratio, name, direct_update=False, g_diag=None, grad_eps=1e-6, rank=None):
        self.layer = layer
        self.name = name
        self.dev = self.layer.weight.device
        # W = layer.weight.data.clone()
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        g_inv_sqrt = None
        self.g_sqrt = None
        self.g_inv_sqrt = None
        self.g_blocks = None
        if g_diag is not None:
            if isinstance(g_diag, dict) and g_diag.get("type") == "block":
                self.g_blocks = []
                for b in g_diag["blocks"]:
                    s, e = b["start"], b["end"]
                    mat = b["mat"].to(self.dev)
                    chol = _safe_cholesky(mat, grad_eps)
                    inv_chol = torch.linalg.inv(chol)
                    self.g_blocks.append({"start": s, "end": e, "g_sqrt": chol, "g_inv_sqrt": inv_chol})
                    W[s:e, :] = chol.matmul(W[s:e, :])
            else:
                g_diag = g_diag.to(self.dev)
                g_sqrt = torch.sqrt(torch.clamp(g_diag, min=grad_eps))
                g_inv_sqrt = 1.0 / g_sqrt
                self.g_sqrt = g_sqrt
                self.g_inv_sqrt = g_inv_sqrt
                W = W * g_sqrt.unsqueeze(1)
        if direct_update:
            self.U, self.S, self.VT = torch.linalg.svd(W.data, full_matrices=False)
        else: 
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0])
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            self.U, self.S, self.VT = torch.linalg.svd(W_scale, full_matrices=False)  
        # truncation SVD
        if rank is not None:
            num_s_after_trunc = int(rank)
        else:
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
        num_s_after_trunc = max(1, min(num_s_after_trunc, min(W.shape[0], W.shape[1])))
        self.truc_s = self.S[:num_s_after_trunc].cuda()
        self.truc_u = self.U[:, :num_s_after_trunc].cuda()
        if self.g_blocks is not None:
            for b in self.g_blocks:
                s, e = b["start"], b["end"]
                self.truc_u[s:e, :] = b["g_inv_sqrt"].matmul(self.truc_u[s:e, :])
        elif g_inv_sqrt is not None:
            self.truc_u = g_inv_sqrt.unsqueeze(1) * self.truc_u
        if direct_update:
            self.truc_v = self.VT[:num_s_after_trunc, :].cuda()
        else:
            self.truc_v = torch.matmul(self.VT[:num_s_after_trunc, :].cuda(), scaling_matrix_inv)
        self.truc_sigma = torch.diag(self.truc_s)
        self.new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v[:num_s_after_trunc, :]))
        # intialize H for close form solution
        self.updated_err = self.error = 0

    def add_batch_update_u(self, inp, out):
        inps = inp.view(inp.shape[0] * inp.shape[1], inp.shape[2])
        outs = out.view(out.shape[0] * out.shape[1], out.shape[2])
        new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v))
        new_output = inps.matmul(new_w.t())
        if self.g_blocks is not None:
            outs_w = outs
            new_output_w = new_output
            for b in self.g_blocks:
                s, e = b["start"], b["end"]
                outs_w[:, s:e] = outs_w[:, s:e].matmul(b["g_sqrt"].t())
                new_output_w[:, s:e] = new_output_w[:, s:e].matmul(b["g_sqrt"].t())
            self.error = torch.sqrt(torch.sum((outs_w - new_output_w) ** 2)).item() / torch.norm(outs_w, p='fro').item()
            x = torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma)
            updated_uT_w = torch.linalg.lstsq(x, outs_w).solution
            updated_uT = updated_uT_w
            for b in self.g_blocks:
                s, e = b["start"], b["end"]
                updated_uT[:, s:e] = updated_uT[:, s:e].matmul(b["g_inv_sqrt"])
            self.updated_uT = updated_uT
            updated_output = torch.matmul(torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma), self.updated_uT)
            updated_output_w = updated_output
            for b in self.g_blocks:
                s, e = b["start"], b["end"]
                updated_output_w[:, s:e] = updated_output_w[:, s:e].matmul(b["g_sqrt"].t())
            self.updated_error = torch.sqrt(torch.sum((outs_w - updated_output_w) ** 2)).item() / torch.norm(outs_w, p='fro').item()
        elif self.g_sqrt is not None:
            outs_w = outs * self.g_sqrt
            new_output_w = new_output * self.g_sqrt
            self.error = torch.sqrt(torch.sum((outs_w - new_output_w) ** 2)).item() / torch.norm(outs_w, p='fro').item()
            # print(f"truncted error: {self.error}")
            x = torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma)
            updated_uT_w = torch.linalg.lstsq(x, outs_w).solution
            self.updated_uT = updated_uT_w * self.g_inv_sqrt.unsqueeze(0)
            updated_output = torch.matmul(torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma), self.updated_uT)
            updated_output_w = updated_output * self.g_sqrt
            self.updated_error = torch.sqrt(torch.sum((outs_w - updated_output_w) ** 2)).item() / torch.norm(outs_w, p='fro').item()
        else:
            self.error = torch.sqrt(torch.sum((outs - new_output)**2)).item() / torch.norm(outs, p='fro').item()
            # print(f"truncted error: {self.error}")
            x =  torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma)
            self.updated_uT = torch.linalg.lstsq(x,outs).solution
            updated_output = torch.matmul(torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma), self.updated_uT)
            self.updated_error = torch.sqrt(torch.sum((outs - updated_output)**2)).item() / torch.norm(outs, p='fro').item()
        # print(f"updated error: {self.updated_error}")
        inps = outs = new_output = updated_output = x = new_w = None
        del inps, outs, new_output, updated_output, x, new_w
        torch.cuda.empty_cache()
        # print(f"Finish {self.name}"
    
    def fasterprune(self):
        sqrtSigma = torch.sqrt(self.truc_sigma)
        self.appendU = self.updated_uT.t().matmul(sqrtSigma)
        self.appendV = sqrtSigma.matmul(self.truc_v)
        return self.appendU, self.appendV


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='jeffwan/llama-7b-hf', help='LLaMA model to load, pass `jeffwan/llama-7b-hf`')
    parser.add_argument('--model_path', type=str, default=None, help='local compressed model path or whitening information path')
    parser.add_argument('--ratio', type=float, default=0.2, help='Target compression ratio,(0,1), default=0.2, means only keeping about 20% of the params.')
    parser.add_argument('--run_low_resource', action='store_true', help='whether to run whitening in low resource, exp, compress LLaMA-7B below 15G gpu')
    parser.add_argument('--dataset', type=str, default='wikitext2',help='Where to extract calibration data from [wikitext2, ptb, c4]')
    parser.add_argument('--whitening_nsamples', type=int, default=256, help='Number of calibration data samples for whitening.')
    parser.add_argument('--updating_nsamples', type=int, default=16, help='Number of calibration data samples for udpating.')
    parser.add_argument('--save_path', type=str, default=None, help='the path to save the compressed model checkpoints.`')
    parser.add_argument('--profiling_mat_path', type=str, default=None, help='Local path to load the profiling matrices`')
    parser.add_argument('--use_grad_g', action='store_true', help='whether to use output-gradient diag weighting')
    parser.add_argument('--grad_nsamples', type=int, default=None, help='Number of calibration batches for grad diag estimation')
    parser.add_argument('--grad_path', type=str, default=None, help='Local path to load grad diag (G) matrices')
    parser.add_argument('--grad_eps', type=float, default=1e-6, help='Epsilon to avoid zero in grad diag')
    parser.add_argument('--grad_seq_len', type=int, default=None, help='Sequence length for grad diag profiling (defaults to model_seq_len)')
    parser.add_argument('--grad_batch_size', type=int, default=None, help='Batch size for grad diag profiling (defaults to 1)')
    parser.add_argument('--grad_checkpointing', action='store_true', help='Enable gradient checkpointing for grad diag profiling to reduce memory')
    parser.add_argument('--g_block_diag', action='store_true', help='use block-diagonal G (head blocks + MLP groups)')
    parser.add_argument('--attn_block_size', type=int, default=0, help='attention block size (0 uses head_dim)')
    parser.add_argument('--mlp_block_size', type=int, default=256, help='MLP block size')
    parser.add_argument('--use_layerwise_ratio', action='store_true', help='allocate per-layer ratios based on G importance')
    parser.add_argument('--layer_ratio_min', type=float, default=0.01, help='Minimum per-layer keep ratio')
    parser.add_argument('--layer_ratio_max', type=float, default=0.99, help='Maximum per-layer keep ratio')
    parser.add_argument('--layer_ratio_strict', action='store_true', help='Rescale layer ratios to exactly match the global keep ratio')
    parser.add_argument('--print_layer_ratios', action='store_true', help='Print per-layer keep ratios and effective global ratio')
    parser.add_argument('--use_module_rank_allocation', action='store_true', help='Allocate per-module ranks via spectrum + Lagrange')
    parser.add_argument('--spectrum_path', type=str, default=None, help='Path to load/save module spectrum cache')
    parser.add_argument('--module_rank_min', type=int, default=1, help='Minimum rank per module')
    parser.add_argument('--module_rank_max', type=int, default=None, help='Maximum rank per module (default: min(out,in))')
    parser.add_argument('--print_module_ranks', action='store_true', help='Print per-module ranks and per-layer effective ratios')
    parser.add_argument('--seed',type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--DEV', type=str, default="cuda", help='device')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='the default sequence length of the LLM')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='inference bactch size')
    parser.add_argument('--gen_seq_len', type=int, default=1024, help='generated sequence len for efficiency evaluation')
    parser.add_argument('--step', type=int, default=4, help='the step to run the compression')
    parser.add_argument('--lora', type=str, default=None, help='the lora updated weight path to run the accuracy evaluation')
    
    args = parser.parse_args()
    args.ratio = 1- args.ratio
    if args.step == 1:
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        model = model.eval()
        grad_diag = None
        layer_ratios = None
        module_ranks = None
        if args.profiling_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        else:
            profiling_mat = torch.load(args.profiling_mat_path)
        if args.use_grad_g or args.use_layerwise_ratio or args.use_module_rank_allocation:
            if args.grad_path is not None:
                grad_diag = torch.load(args.grad_path)
            else:
                grad_nsamples = args.grad_nsamples if args.grad_nsamples is not None else args.whitening_nsamples
                grad_seq_len = args.grad_seq_len if args.grad_seq_len is not None else args.model_seq_len
                grad_batch_size = args.grad_batch_size if args.grad_batch_size is not None else 1
                if "cali_white_data" in locals() and grad_seq_len == args.model_seq_len and grad_batch_size == 1:
                    cali_grad_data = cali_white_data
                else:
                    cali_grad_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=grad_seq_len, batch_size=grad_batch_size)
                grad_diag = profile_grad_diag(
                    args.model, model, cali_grad_data, args.DEV, max_batches=grad_nsamples, grad_eps=args.grad_eps,
                    block_diag=args.g_block_diag, attn_block_size=args.attn_block_size, mlp_block_size=args.mlp_block_size,
                    use_checkpointing=args.grad_checkpointing,
                )
                if args.save_path is not None:
                    torch.save(grad_diag, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_grad_diag_'+ args.dataset + '_' + str(grad_nsamples)  + '_' + str(args.seed)+ '.pt')
        if args.use_layerwise_ratio:
            layer_ratios = compute_layer_ratios(args.model, model, grad_diag, args.ratio, min_ratio=args.layer_ratio_min, max_ratio=args.layer_ratio_max, eps=args.grad_eps)
            layer_sizes = _layer_param_sizes(args.model, model)
            if layer_ratios is not None and layer_sizes is not None:
                if args.layer_ratio_strict:
                    layer_ratios, effective_ratio = _normalize_layer_ratios(layer_ratios, layer_sizes, args.ratio)
                else:
                    _, effective_ratio = _normalize_layer_ratios(layer_ratios, layer_sizes, args.ratio)
                if args.print_layer_ratios:
                    print(f"Layerwise keep ratios (target={args.ratio:.6f}, effective={effective_ratio:.6f})")
                    for i, r in enumerate(layer_ratios):
                        print(f"layer {i:02d} ratio={r:.6f} size={layer_sizes[i]}")
        if args.use_module_rank_allocation:
            grad_nsamples = args.grad_nsamples if args.grad_nsamples is not None else args.whitening_nsamples
            spectrum_path = args.spectrum_path
            if spectrum_path is None and args.save_path is not None:
                spectrum_path = _default_spectrum_path(args.save_path, args.model, args.dataset, grad_nsamples, args.seed)
            if spectrum_path is not None and os.path.exists(spectrum_path):
                spectrum = torch.load(spectrum_path)
            else:
                spectrum = profile_module_spectrum(
                    args.model, model, profiling_mat, args.DEV,
                    grad_diag=grad_diag, grad_eps=args.grad_eps,
                )
                if spectrum_path is not None:
                    torch.save(spectrum, spectrum_path)
            module_ranks, effective_ratio = allocate_module_ranks(
                spectrum, args.ratio, min_rank=args.module_rank_min, max_rank=args.module_rank_max
            )
            if args.print_module_ranks and module_ranks is not None:
                effective_ratio, layer_stats = summarize_module_ranks(spectrum, module_ranks)
                print(f"Module ranks (target={args.ratio:.6f}, effective={effective_ratio:.6f})")
                for layer_id in sorted(layer_stats.keys()):
                    print(f"layer {layer_id:02d} ratio={layer_stats[layer_id]:.6f}")
                for layer_id in sorted(module_ranks.keys()):
                    for name, k in module_ranks[layer_id].items():
                        shape = spectrum[layer_id][name]["shape"]
                        print(f"layer {layer_id:02d} {name} rank={k} shape={shape}")
            layer_ratios = None
        whitening(args.model, model, profiling_mat, args.ratio, args.DEV, grad_diag=grad_diag if args.use_grad_g else None, grad_eps=args.grad_eps, layer_ratios=layer_ratios, module_ranks=module_ranks)
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_only_' + str(args.ratio) + '.pt')   # fp32
    elif args.step == 2:
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        model = model.eval()
        model = model.float()  # need to set to float
        grad_diag = None
        layer_ratios = None
        module_ranks = None
        if args.profiling_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        else:
            profiling_mat = torch.load(args.profiling_mat_path)
        if args.use_grad_g or args.use_layerwise_ratio or args.use_module_rank_allocation:
            if args.grad_path is not None:
                grad_diag = torch.load(args.grad_path)
            else:
                grad_nsamples = args.grad_nsamples if args.grad_nsamples is not None else args.whitening_nsamples
                grad_seq_len = args.grad_seq_len if args.grad_seq_len is not None else args.model_seq_len
                grad_batch_size = args.grad_batch_size if args.grad_batch_size is not None else 1
                if "cali_white_data" in locals() and grad_seq_len == args.model_seq_len and grad_batch_size == 1:
                    cali_grad_data = cali_white_data
                else:
                    cali_grad_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=grad_seq_len, batch_size=grad_batch_size)
                grad_diag = profile_grad_diag(
                    args.model, model, cali_grad_data, args.DEV, max_batches=grad_nsamples, grad_eps=args.grad_eps,
                    block_diag=args.g_block_diag, attn_block_size=args.attn_block_size, mlp_block_size=args.mlp_block_size,
                    use_checkpointing=args.grad_checkpointing,
                )
                if args.save_path is not None:
                    torch.save(grad_diag, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_grad_diag_'+ args.dataset + '_' + str(grad_nsamples)  + '_' + str(args.seed)+ '.pt')
        if args.use_layerwise_ratio:
            layer_ratios = compute_layer_ratios(args.model, model, grad_diag, args.ratio, min_ratio=args.layer_ratio_min, max_ratio=args.layer_ratio_max, eps=args.grad_eps)
            layer_sizes = _layer_param_sizes(args.model, model)
            if layer_ratios is not None and layer_sizes is not None:
                if args.layer_ratio_strict:
                    layer_ratios, effective_ratio = _normalize_layer_ratios(layer_ratios, layer_sizes, args.ratio)
                else:
                    _, effective_ratio = _normalize_layer_ratios(layer_ratios, layer_sizes, args.ratio)
                if args.print_layer_ratios:
                    print(f"Layerwise keep ratios (target={args.ratio:.6f}, effective={effective_ratio:.6f})")
                    for i, r in enumerate(layer_ratios):
                        print(f"layer {i:02d} ratio={r:.6f} size={layer_sizes[i]}")
        if args.use_module_rank_allocation:
            grad_nsamples = args.grad_nsamples if args.grad_nsamples is not None else args.whitening_nsamples
            spectrum_path = args.spectrum_path
            if spectrum_path is None and args.save_path is not None:
                spectrum_path = _default_spectrum_path(args.save_path, args.model, args.dataset, grad_nsamples, args.seed)
            if spectrum_path is not None and os.path.exists(spectrum_path):
                spectrum = torch.load(spectrum_path)
            else:
                spectrum = profile_module_spectrum(
                    args.model, model, profiling_mat, args.DEV,
                    grad_diag=grad_diag, grad_eps=args.grad_eps,
                )
                if spectrum_path is not None:
                    torch.save(spectrum, spectrum_path)
            module_ranks, effective_ratio = allocate_module_ranks(
                spectrum, args.ratio, min_rank=args.module_rank_min, max_rank=args.module_rank_max
            )
            if args.print_module_ranks and module_ranks is not None:
                effective_ratio, layer_stats = summarize_module_ranks(spectrum, module_ranks)
                print(f"Module ranks (target={args.ratio:.6f}, effective={effective_ratio:.6f})")
                for layer_id in sorted(layer_stats.keys()):
                    print(f"layer {layer_id:02d} ratio={layer_stats[layer_id]:.6f}")
                for layer_id in sorted(module_ranks.keys()):
                    for name, k in module_ranks[layer_id].items():
                        shape = spectrum[layer_id][name]["shape"]
                        print(f"layer {layer_id:02d} {name} rank={k} shape={shape}")
            layer_ratios = None
        whitening_local_update(args.model, model, dataloader, profiling_mat, args.ratio, args.DEV, grad_diag=grad_diag if args.use_grad_g else None, grad_eps=args.grad_eps, layer_ratios=layer_ratios, module_ranks=module_ranks)
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_then_update_' + str(args.ratio) + '.pt')  # fp32
    elif args.step == 3:
        model, tokenizer = get_model_from_huggingface(args.model)
        model = model.eval()
        model = model.float()
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        grad_diag = None
        layer_ratios = None
        module_ranks = None
        if args.use_grad_g or args.use_layerwise_ratio or args.use_module_rank_allocation:
            if args.grad_path is not None:
                grad_diag = torch.load(args.grad_path)
            else:
                grad_nsamples = args.grad_nsamples if args.grad_nsamples is not None else args.whitening_nsamples
                grad_seq_len = args.grad_seq_len if args.grad_seq_len is not None else args.model_seq_len
                grad_batch_size = args.grad_batch_size if args.grad_batch_size is not None else 1
                if "cali_white_data" in locals() and grad_seq_len == args.model_seq_len and grad_batch_size == 1:
                    cali_grad_data = cali_white_data
                else:
                    cali_grad_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=grad_seq_len, batch_size=grad_batch_size)
                grad_diag = profile_grad_diag(
                    args.model, model, cali_grad_data, args.DEV, max_batches=grad_nsamples, grad_eps=args.grad_eps,
                    block_diag=args.g_block_diag, attn_block_size=args.attn_block_size, mlp_block_size=args.mlp_block_size,
                    use_checkpointing=args.grad_checkpointing,
                )
                if args.save_path is not None:
                    torch.save(grad_diag, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_grad_diag_'+ args.dataset + '_' + str(grad_nsamples)  + '_' + str(args.seed)+ '.pt')
        if args.use_layerwise_ratio:
            layer_ratios = compute_layer_ratios(args.model, model, grad_diag, args.ratio, min_ratio=args.layer_ratio_min, max_ratio=args.layer_ratio_max, eps=args.grad_eps)
            layer_sizes = _layer_param_sizes(args.model, model)
            if layer_ratios is not None and layer_sizes is not None:
                if args.layer_ratio_strict:
                    layer_ratios, effective_ratio = _normalize_layer_ratios(layer_ratios, layer_sizes, args.ratio)
                else:
                    _, effective_ratio = _normalize_layer_ratios(layer_ratios, layer_sizes, args.ratio)
                if args.print_layer_ratios:
                    print(f"Layerwise keep ratios (target={args.ratio:.6f}, effective={effective_ratio:.6f})")
                    for i, r in enumerate(layer_ratios):
                        print(f"layer {i:02d} ratio={r:.6f} size={layer_sizes[i]}")
        if args.use_module_rank_allocation:
            grad_nsamples = args.grad_nsamples if args.grad_nsamples is not None else args.whitening_nsamples
            spectrum_path = args.spectrum_path
            if spectrum_path is None and args.save_path is not None:
                spectrum_path = _default_spectrum_path(args.save_path, args.model, args.dataset, grad_nsamples, args.seed)
            if spectrum_path is not None and os.path.exists(spectrum_path):
                spectrum = torch.load(spectrum_path)
            else:
                spectrum = profile_module_spectrum(
                    args.model, model, profiling_mat=None, dev=args.DEV,
                    grad_diag=grad_diag, grad_eps=args.grad_eps,
                )
                if spectrum_path is not None:
                    torch.save(spectrum, spectrum_path)
            module_ranks, effective_ratio = allocate_module_ranks(
                spectrum, args.ratio, min_rank=args.module_rank_min, max_rank=args.module_rank_max
            )
            if args.print_module_ranks and module_ranks is not None:
                effective_ratio, layer_stats = summarize_module_ranks(spectrum, module_ranks)
                print(f"Module ranks (target={args.ratio:.6f}, effective={effective_ratio:.6f})")
                for layer_id in sorted(layer_stats.keys()):
                    print(f"layer {layer_id:02d} ratio={layer_stats[layer_id]:.6f}")
                for layer_id in sorted(module_ranks.keys()):
                    for name, k in module_ranks[layer_id].items():
                        shape = spectrum[layer_id][name]["shape"]
                        print(f"layer {layer_id:02d} {name} rank={k} shape={shape}")
            layer_ratios = None
        whitening_local_update(model_name=args.model, model=model, dataloader=dataloader, profiling_mat=None, ratio=args.ratio, dev=args.DEV, direct_update=True, grad_diag=grad_diag if args.use_grad_g else None, grad_eps=args.grad_eps, layer_ratios=layer_ratios, module_ranks=module_ranks)
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_update_only_' + str(args.ratio) + '.pt')   # fp32
    elif args.step >= 4:
        print(f"evaluating {args.model_path}...")
        if args.model_path == "original":
            model, tokenizer = get_model_from_huggingface(args.model)
        else:
            model, tokenizer = get_model_from_local(args.model_path)
            if args.lora is not None:
                from utils.peft import PeftModel
                model = PeftModel.from_pretrained(
                    model,
                    args.lora,
                    torch_dtype=torch.float16,
                )
                model = model.merge_and_unload()
                torch.save({'model': model, 'tokenizer': tokenizer}, args.lora + '/merge.pt')
        model.eval()
        model = model.float()
        model = model.to(args.DEV)
        if args.step == 4:
            ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        elif args.step == 5:
            eff_eval(model, tokenizer, generated_len=args.gen_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
