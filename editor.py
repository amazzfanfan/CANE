"""Stage 3 of CANE: Sparse delta knowledge editing with KL locality constraint and evaluation."""

from copy import deepcopy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader


def edit_model(
    model: nn.Module,
    knowledge_loader: DataLoader,
    background_loader: DataLoader,
    neuron_mask: Dict[str, torch.Tensor],
    device: torch.device,
    num_epochs: int = 120,
    lr: float = 3e-4,
    lambda_ce: float = 1.0,
    lambda_loc: float = 10.0,
    lambda_norm: float = 1e-4,
) -> nn.Module:
    """Apply localized sparse delta editing to selected neurons and merge updates into weights.

    Optimizes:
        L = lambda_ce * CE(target_knowledge) + lambda_loc * KL(P_edited(bg) || P_orig(bg))
            + lambda_norm * sum_l ||Delta^(l)||_2
    """
    model.to(device)

    # Freeze base parameters and create a frozen teacher copy for KL locality regularization
    for param in model.parameters():
        param.requires_grad = False
    teacher_model = deepcopy(model).eval().to(device)
    for param in teacher_model.parameters():
        param.requires_grad = False

    def _freeze_bn(module: nn.Module):
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.eval()

    model.eval()
    model.apply(_freeze_bn)
    teacher_model.apply(_freeze_bn)

    # Allocate trainable row increments only for masked neurons
    deltas: Dict[str, torch.Tensor] = {}
    target_modules: Dict[str, nn.Linear] = {}
    neuron_indices: Dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if name in neuron_mask and neuron_mask[name].any():
            mask = neuron_mask[name]
            num_selected = int(mask.sum().item())
            deltas[name] = torch.zeros(
                (num_selected, module.in_features),
                device=device,
                dtype=module.weight.dtype,
                requires_grad=True,
            )
            target_modules[name] = module
            neuron_indices[name] = mask.nonzero(as_tuple=True)[0].to(device)

    if not deltas:
        print("[Editor] No neurons selected; skipping editing.")
        return model

    def _make_delta_hook(delta_tensor: torch.Tensor, idx_tensor: torch.Tensor):
        def _hook(_module, inputs, output):
            out_delta = inputs[0].matmul(delta_tensor.T)
            updated = output.clone()
            updated[:, idx_tensor] += out_delta
            return updated

        return _hook

    optimizer = torch.optim.Adam(deltas.values(), lr=lr)

    for epoch in range(num_epochs):
        it_knowledge = iter(knowledge_loader)
        it_bg = iter(background_loader)
        steps = min(len(knowledge_loader), len(background_loader))

        total_loss_sum = 0.0
        ce_loss_sum = 0.0
        kl_loss_sum = 0.0

        for _ in range(steps):
            try:
                batch_k = next(it_knowledge).to(device)
                batch_bg = next(it_bg).to(device)
            except StopIteration:
                break

            optimizer.zero_grad()

            # Forward pass with sparse delta hooks active
            handles = [
                module.register_forward_hook(_make_delta_hook(deltas[name], neuron_indices[name]))
                for name, module in target_modules.items()
            ]
            try:
                out_k = model(batch_k.x, batch_k.edge_index, getattr(batch_k, "batch", None))
                out_bg = model(batch_bg.x, batch_bg.edge_index, getattr(batch_bg, "batch", None))
            finally:
                for handle in handles:
                    handle.remove()

            # Frozen pre-edit teacher predictions on background graphs
            with torch.no_grad():
                out_bg_teacher = teacher_model(
                    batch_bg.x, batch_bg.edge_index, getattr(batch_bg, "batch", None)
                )

            y_k = batch_k.y.view(-1).to(device, dtype=torch.long)
            loss_ce = F.cross_entropy(out_k, y_k)
            loss_kl = F.kl_div(
                F.log_softmax(out_bg, dim=-1),
                F.softmax(out_bg_teacher, dim=-1),
                reduction="batchmean",
            )
            loss_reg = sum(delta.norm(p=2) for delta in deltas.values())

            loss = lambda_ce * loss_ce + lambda_loc * loss_kl + lambda_norm * loss_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(deltas.values()), max_norm=1.0)
            optimizer.step()

            total_loss_sum += loss.item()
            ce_loss_sum += loss_ce.item()
            kl_loss_sum += loss_kl.item()

        if steps > 0 and (epoch == 0 or (epoch + 1) % 20 == 0 or epoch == num_epochs - 1):
            print(
                f"Epoch {epoch + 1:03d}/{num_epochs} | "
                f"Loss: {total_loss_sum / steps:.4f} | "
                f"CE: {ce_loss_sum / steps:.4f} | "
                f"KL: {kl_loss_sum / steps:.4f}"
            )

    # Merge learned row increments directly into the pretrained weight matrices
    with torch.no_grad():
        for name, delta_tensor in deltas.items():
            module = target_modules[name]
            idx = neuron_indices[name].to(module.weight.data.device)
            module.weight.data[idx, :] += delta_tensor.to(module.weight.data.device)

    return model


def _loader_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, getattr(batch, "batch", None))
            preds = logits.argmax(dim=1)
            labels = batch.y.view(-1).to(device)
            correct += int((preds == labels).sum().item())
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    test_by_class: Dict[int, DataLoader],
    target_class: int,
    class_names: Dict[int, str],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate overall and per-class classification accuracy on the test set."""
    model.eval().to(device)
    results = {"Overall": _loader_accuracy(model, test_loader, device)}
    for class_id, loader in sorted(test_by_class.items()):
        label = class_names[class_id]
        results[label] = _loader_accuracy(model, loader, device)

    parts = [f"Overall: {results['Overall'] * 100:.2f}%"]
    for class_id in sorted(test_by_class.keys()):
        label = class_names[class_id]
        tag = " (Target)" if class_id == target_class else ""
        parts.append(f"{label}{tag}: {results[label] * 100:.2f}%")
    print("  " + " | ".join(parts))
    return results
