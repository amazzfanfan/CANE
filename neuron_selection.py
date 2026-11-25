"""Stage 1 & Stage 2 of CANE: Contrastive neuron screening and cross-layer co-activation clustering."""

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch_geometric.loader import DataLoader

ABLATION_MODES = (
    "full",
    "wo_coactivation",
    "wo_contrastive",
    "wo_both",
)


def list_target_linear_layers(model: nn.Module, hidden_dim: int = 64) -> Dict[str, nn.Linear]:
    """Return hidden linear modules with shape (hidden_dim x hidden_dim) in forward order."""
    target_layers: Dict[str, nn.Linear] = {}
    for name, module in model.named_modules():
        if (
            isinstance(module, nn.Linear)
            and module.in_features == hidden_dim
            and module.out_features == hidden_dim
        ):
            target_layers[name] = module
    return target_layers


def collect_specificity_scores(
    model: nn.Module,
    target_loader: DataLoader,
    background_loader: DataLoader,
    device: torch.device,
    hidden_dim: int = 64,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute per-neuron Cohen's d specificity scores between target and background graphs.

    Returns:
        specificity_scores: Dict mapping layer name to Cohen's d vector of length `hidden_dim`.
        target_activations: Dict mapping layer name to stacked target activations [N_nodes, hidden_dim].
    """
    model.eval().to(device)
    target_layers = list_target_linear_layers(model, hidden_dim=hidden_dim)

    batch_cache: Dict[str, torch.Tensor] = {}
    pos_acts = defaultdict(list)
    neg_acts = defaultdict(list)

    def _make_hook(layer_name: str):
        def _hook(_module, _inputs, output):
            batch_cache[layer_name] = output.detach().cpu()

        return _hook

    handles = [layer.register_forward_hook(_make_hook(name)) for name, layer in target_layers.items()]

    try:
        with torch.no_grad():
            for batch in target_loader:
                batch = batch.to(device)
                model(batch.x, batch.edge_index, getattr(batch, "batch", None))
                for name, act in batch_cache.items():
                    pos_acts[name].append(act.numpy())
                batch_cache.clear()

            for batch in background_loader:
                batch = batch.to(device)
                model(batch.x, batch.edge_index, getattr(batch, "batch", None))
                for name, act in batch_cache.items():
                    neg_acts[name].append(act.numpy())
                batch_cache.clear()
    finally:
        for handle in handles:
            handle.remove()

    specificity_scores: Dict[str, np.ndarray] = {}
    target_activations: Dict[str, np.ndarray] = {}

    for name in target_layers:
        pos = np.vstack(pos_acts[name]) if pos_acts[name] else np.empty((0, hidden_dim))
        neg = np.vstack(neg_acts[name]) if neg_acts[name] else np.empty((0, hidden_dim))
        target_activations[name] = pos

        if pos.size == 0 or neg.size == 0:
            specificity_scores[name] = np.zeros(hidden_dim)
            continue

        pos_mean = np.mean(pos, axis=0)
        neg_mean = np.mean(neg, axis=0)
        pos_std = np.std(pos, axis=0)
        neg_std = np.std(neg, axis=0)

        n1, n2 = pos.shape[0], neg.shape[0]
        denom = max(n1 + n2 - 2, 1)
        pooled_std = np.sqrt(((n1 - 1) * pos_std ** 2 + (n2 - 1) * neg_std ** 2) / denom)
        specificity_scores[name] = (pos_mean - neg_mean) / (pooled_std + 1e-8)

    return specificity_scores, target_activations


def build_coactivation_clusters(
    candidate_neurons: Dict[str, List[int]],
    target_activations: Dict[str, np.ndarray],
    layer_order: Sequence[str],
    top_p: float = 0.15,
    hidden_dim: int = 64,
) -> Dict[str, torch.Tensor]:
    """Extract cross-layer neuron clusters using positive adjacent-layer Pearson correlations.

    Retains the top-P fraction of positively correlated edges between each pair of adjacent
    hidden layers and keeps connected components spanning all hidden layers.
    """
    graph = nx.Graph()
    num_layers = len(layer_order)

    for layer_id, name in enumerate(layer_order):
        for neuron_idx in candidate_neurons.get(name, []):
            graph.add_node((layer_id, neuron_idx), layer=name)

    for layer_id in range(num_layers - 1):
        curr_layer = layer_order[layer_id]
        next_layer = layer_order[layer_id + 1]
        curr_acts = target_activations[curr_layer]
        next_acts = target_activations[next_layer]
        curr_neurons = candidate_neurons.get(curr_layer, [])
        next_neurons = candidate_neurons.get(next_layer, [])

        pair_corrs = []
        for i in curr_neurons:
            act_i = curr_acts[:, i]
            if len(act_i) < 2 or np.std(act_i) == 0:
                continue
            for j in next_neurons:
                act_j = next_acts[:, j]
                if len(act_j) < 2 or np.std(act_j) == 0:
                    continue
                corr, _ = pearsonr(act_i, act_j)
                if corr > 0:
                    pair_corrs.append(((i, j), corr))

        if not pair_corrs:
            continue

        pair_corrs.sort(key=lambda item: item[1], reverse=True)
        num_edges = max(int(len(pair_corrs) * top_p), min(5, len(pair_corrs)))
        for (u, v), corr in pair_corrs[:num_edges]:
            graph.add_edge((layer_id, u), (layer_id + 1, v), weight=float(corr))

    # Retain connected components that cover every hidden layer
    selected_by_layer = defaultdict(set)
    spanning_components = 0
    for component in nx.connected_components(graph):
        covered_layers = {node[0] for node in component}
        if len(covered_layers) == num_layers:
            spanning_components += 1
            for layer_id, neuron_idx in component:
                selected_by_layer[layer_order[layer_id]].add(neuron_idx)

    masks: Dict[str, torch.Tensor] = {}
    for name in layer_order:
        mask = torch.zeros(hidden_dim, dtype=torch.bool)
        if selected_by_layer[name]:
            mask[sorted(selected_by_layer[name])] = True
        masks[name] = mask

    print(
        f"[Co-activation Graph] Top-{top_p * 100:.0f}% edges -> "
        f"{graph.number_of_edges()} edges, {spanning_components} full-depth cluster(s)."
    )
    return masks


def select_neurons(
    model: nn.Module,
    target_loader: DataLoader,
    background_loaders: Sequence[DataLoader],
    device: torch.device,
    mode: str = "full",
    top_s: float = 0.2,
    top_p: float = 0.15,
    hidden_dim: int = 64,
) -> Dict[str, torch.Tensor]:
    """Select editing masks according to CANE or one of its ablation variants.

    Modes:
        - `full`: Contrastive Top-S screening + cross-layer Top-P co-activation clustering.
        - `wo_coactivation`: Contrastive Top-S screening only.
        - `wo_contrastive`: Cross-layer Top-P co-activation clustering over all neurons.
        - `wo_both`: Select all neurons in target hidden linear layers.
    """
    if mode not in ABLATION_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {', '.join(ABLATION_MODES)}")

    target_layers = list_target_linear_layers(model, hidden_dim=hidden_dim)
    layer_order = list(target_layers.keys())

    # Ablation: edit all neurons in target linear layers
    if mode == "wo_both":
        return {name: torch.ones(hidden_dim, dtype=torch.bool) for name in layer_order}

    # Ablation: remove contrastive screening, keep co-activation clustering
    if mode == "wo_contrastive":
        _, target_activations = collect_specificity_scores(
            model, target_loader, background_loaders[0], device=device, hidden_dim=hidden_dim
        )
        all_candidates = {name: list(range(hidden_dim)) for name in layer_order}
        return build_coactivation_clusters(
            all_candidates, target_activations, layer_order, top_p=top_p, hidden_dim=hidden_dim
        )

    # Stage 1: Contrastive Top-S screening (intersected across multiple background classes if present)
    all_score_dicts: List[Dict[str, np.ndarray]] = []
    target_activations: Dict[str, np.ndarray] = {}
    for idx, bg_loader in enumerate(background_loaders):
        scores, pos_acts = collect_specificity_scores(
            model, target_loader, bg_loader, device=device, hidden_dim=hidden_dim
        )
        all_score_dicts.append(scores)
        if idx == 0:
            target_activations = pos_acts

    candidate_neurons: Dict[str, List[int]] = {}
    k = max(int(hidden_dim * top_s), 1)
    for name in layer_order:
        common = None
        for score_dict in all_score_dicts:
            top_indices = set(np.argsort(score_dict[name])[-k:].tolist())
            common = top_indices if common is None else (common & top_indices)
        if common:
            candidate_neurons[name] = sorted(common)

    if not candidate_neurons:
        return {name: torch.zeros(hidden_dim, dtype=torch.bool) for name in layer_order}

    # Ablation: remove co-activation cluster extraction
    if mode == "wo_coactivation":
        masks = {}
        for name in layer_order:
            mask = torch.zeros(hidden_dim, dtype=torch.bool)
            if name in candidate_neurons:
                mask[candidate_neurons[name]] = True
            masks[name] = mask
        return masks

    # Stage 2: Cross-layer co-activation cluster extraction
    return build_coactivation_clusters(
        candidate_neurons, target_activations, layer_order, top_p=top_p, hidden_dim=hidden_dim
    )
