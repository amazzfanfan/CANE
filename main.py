"""Main CLI entry point for running CANE and its ablation variants."""

import argparse
import os
import random
import time
from copy import deepcopy

# Default hyperparameters for each benchmark dataset
DATASET_CONFIGS = {
    "ba3": {
        "seed": 123,
        "reset_before_edit": 123,
        "top_s": 0.96,
        "top_p": 0.44,
        "num_epochs": 120,
        "lr": 1e-4,
        "lambda_ce": 1.0,
        "lambda_loc": 10.0,
        "lambda_norm": 1e-4,
    },
    "benzene": {
        "seed": 0,
        "reset_before_edit": None,
        "top_s": 0.20,
        "top_p": 0.60,
        "num_epochs": 120,
        "lr": 3e-4,
        "lambda_ce": 10.0,
        "lambda_loc": 150.0,
        "lambda_norm": 1e-4,
    },
    "mutagenicity": {
        "seed": 123,
        "reset_before_edit": 123,
        "top_s": 0.20,
        "top_p": 0.96,
        "num_epochs": 120,
        "lr": 1e-4,
        "lambda_ce": 4.0,
        "lambda_loc": 10.0,
        "lambda_norm": 1e-4,
    },
    "bbbp": {
        "seed": 12345,
        "reset_before_edit": None,
        "top_s": 0.10,
        "top_p": 0.30,
        "num_epochs": 120,
        "lr": 1e-3,
        "lambda_ce": 1.0,
        "lambda_loc": 5.0,
        "lambda_norm": 1e-4,
    },
}


def setup_device(device_str: str):
    """Configure CUDA_VISIBLE_DEVICES before importing torch and return torch.device."""
    if device_str.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str

    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" (CUDA_VISIBLE_DEVICES={device_str})" if device.type == "cuda" else ""))
    return device


def set_seed(seed: int):
    """Fix random seeds across Python, NumPy, and PyTorch."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="CANE: Collaborative Activation-based Network Editing for GNNs")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Benchmark dataset to edit (ba3, benzene, mutagenicity, bbbp).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "wo_coactivation", "wo_contrastive", "wo_both"],
        help="Neuron selection mode: full CANE or ablation variant.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU ID / UUID (e.g., '0', '1') or 'cpu'.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override editing epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Override editing learning rate.")
    parser.add_argument("--top-s", type=float, default=None, help="Override contrastive Top-S ratio.")
    parser.add_argument("--top-p", type=float, default=None, help="Override co-activation Top-P ratio.")
    parser.add_argument("--lambda-ce", type=float, default=None, help="Override CE loss weight.")
    parser.add_argument("--lambda-loc", type=float, default=None, help="Override KL locality loss weight.")
    parser.add_argument("--lambda-norm", type=float, default=None, help="Override L2 increment regularization weight.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = setup_device(args.device)

    from datasets import prepare_dataset
    from editor import edit_model, evaluate_model
    from neuron_selection import select_neurons

    cfg = dict(DATASET_CONFIGS[args.dataset])
    seed = args.seed if args.seed is not None else cfg["seed"]
    top_s = args.top_s if args.top_s is not None else cfg["top_s"]
    top_p = args.top_p if args.top_p is not None else cfg["top_p"]
    num_epochs = args.epochs if args.epochs is not None else cfg["num_epochs"]
    lr = args.lr if args.lr is not None else cfg["lr"]
    lambda_ce = args.lambda_ce if args.lambda_ce is not None else cfg["lambda_ce"]
    lambda_loc = args.lambda_loc if args.lambda_loc is not None else cfg["lambda_loc"]
    lambda_norm = args.lambda_norm if args.lambda_norm is not None else cfg["lambda_norm"]

    print("=" * 64)
    print(f"Dataset: {args.dataset} | Mode: {args.mode} | Top-S: {top_s} | Top-P: {top_p}")
    print("=" * 64)

    start_time = time.time()
    set_seed(seed)

    # 1. Load dataset, knowledge graphs, and pretrained GIN model
    print("\n[Stage 1/4] Loading dataset, knowledge graphs, and pretrained model...")
    ctx = prepare_dataset(args.dataset, device=device)

    # 2. Localize target-specific cross-layer neuron clusters
    print("\n[Stage 2/4] Localizing target neurons...")
    neuron_mask = select_neurons(
        model=ctx["model"],
        target_loader=ctx["knowledge_target"],
        background_loaders=ctx["knowledge_contrast"],
        device=device,
        mode=args.mode,
        top_s=top_s,
        top_p=top_p,
    )
    total_selected = 0
    for layer_name, mask in neuron_mask.items():
        indices = mask.nonzero(as_tuple=True)[0].tolist()
        if indices:
            print(f"  Layer '{layer_name}': {len(indices)} neurons -> {indices}")
            total_selected += len(indices)
    print(f"Total selected neurons: {total_selected}")

    # 3. Evaluate pre-edit model
    print("\n[Stage 3/4] Evaluating pre-edit model...")
    evaluate_model(
        ctx["model"],
        ctx["test_loader"],
        ctx["test_by_class"],
        ctx["target_class"],
        ctx["class_names"],
        device,
    )

    # 4. Perform localized sparse delta editing and evaluate post-edit model
    print("\n[Stage 4/4] Running CANE sparse delta editing...")
    model_to_edit = deepcopy(ctx["model"])
    if cfg["reset_before_edit"] is not None:
        set_seed(cfg["reset_before_edit"])

    edited_model = edit_model(
        model=model_to_edit,
        knowledge_loader=ctx["knowledge_target"],
        background_loader=ctx["locality_background"],
        neuron_mask=neuron_mask,
        device=device,
        num_epochs=num_epochs,
        lr=lr,
        lambda_ce=lambda_ce,
        lambda_loc=lambda_loc,
        lambda_norm=lambda_norm,
    )

    print("\nPost-edit evaluation:")
    evaluate_model(
        edited_model,
        ctx["test_loader"],
        ctx["test_by_class"],
        ctx["target_class"],
        ctx["class_names"],
        device,
    )
    print(f"\nCompleted in {time.time() - start_time:.2f}s.")


if __name__ == "__main__":
    main()
