# CANE

This repository contains the official implementation of **CANE (Collaborative Activation-based Network Editing)**, a cross-layer neuron-level knowledge editing framework for Graph Neural Networks (GNNs).

## Overview

CANE edits pretrained GNN graph classifiers in three stages to correct targeted predictions while preserving locality on unrelated classes:

1. **Contrastive Neuron Screening (Top-S):** Contrasts forward activations on target-knowledge graphs against background graphs using Cohen's $d$ specificity scores to filter out polysemantic neurons shared with non-target classes.
2. **Cross-Layer Co-Activation Cluster Extraction (Top-P):** Constructs an undirected neuron graph over adjacent hidden layers weighted by positive Pearson correlation on target activations, and extracts full-depth connected components spanning all hidden layers.
3. **Localized Sparse Delta Editing:** Freezes the pretrained backbone and learns sparse row increments $\Delta$ solely on the selected neuron clusters under a combined cross-entropy, KL-divergence locality, and $L_2$ regularization objective, then merges $\Delta$ back into the model weights.

## Environment Setup

The code has been tested with the following environment:

- Python 3.8.20
- torch==1.10.0+cu111
- torch-scatter==2.0.9
- torch-sparse==0.6.12
- torch-cluster==1.6.0
- torch-spline-conv==1.2.1
- torch_geometric==2.5.2
- rdkit==2024.3.5
- networkx==2.6.3
- numpy==1.24.4
- scipy==1.10.1
- scikit-learn==1.3.2
- tqdm==4.67.1

```shell
conda create -n cane python=3.8 -y
conda activate cane

pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==2.0.9 torch-sparse==0.6.12 torch-cluster==1.6.0 torch-spline-conv==1.2.1 \
    -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
pip install -r requirements.txt
```

## Project Structure

```text
CANE/
├── main.py              # Unified CLI entry point for main and ablation experiments
├── model.py             # GIN graph classifier and pretrained checkpoint loader
├── datasets.py          # Dataset loaders (BA3, Benzene, Mutagenicity, BBBP) and knowledge builders
├── neuron_selection.py  # Contrastive Top-S screening and Top-P co-activation cluster extraction
├── editor.py            # Sparse delta editing with KL locality regularization and evaluation
├── models/              # Pretrained GIN checkpoints
├── data/                # Benchmark datasets (BBBP is automatically processed from raw on first run)
├── requirements.txt
└── README.md
```

## Running the Code

### Main Experiments

Run CANE on each of the four benchmark datasets using the default hyperparameters from the paper:

```shell
python main.py --dataset ba3 --device 0
```

```shell
python main.py --dataset benzene --device 0
```

```shell
python main.py --dataset mutagenicity --device 0
```

```shell
python main.py --dataset bbbp --device 0
```

Set `--device cpu` to run on CPU, or pass a GPU index (e.g., `--device 0`).

### Ablation Experiments

Specify `--mode` to evaluate the three ablation variants (`wo_coactivation`, `wo_contrastive`, and `wo_both`) on any dataset:

```shell
# w/o Co-activation: keep Cohen's d contrastive screening, remove Pearson cross-layer clustering
python main.py --dataset benzene --mode wo_coactivation --device 0

# w/o Contrastive: keep Pearson cross-layer clustering, remove Cohen's d screening
python main.py --dataset benzene --mode wo_contrastive --device 0

# w/o Both: edit all neurons in the candidate hidden linear layers
python main.py --dataset benzene --mode wo_both --device 0
```

## Results

### Main Editing Results

Classification accuracy (%) before (`Original`) and after editing with **CANE** on the four benchmark datasets. Bold metric names denote the target class; remaining classes measure locality.

| Editing Target | Metric | Original | **CANE** |
|---|---|---:|---:|
| BA3: House | Overall | 91.25 | 95.50 |
| | **House** | 78.95 | 91.73 |
| | Grid | 100.00 | 100.00 |
| | Cycle | 94.40 | 94.40 |
| Benzene: Benzene | Overall | 92.25 | 91.75 |
| | **Target** | 96.92 | 97.41 |
| | Non-target | 87.31 | 85.76 |
| Mutagenicity: Mutagenic | Overall | 71.20 | 72.20 |
| | **Target** | 73.03 | 77.15 |
| | Non-target | 69.10 | 66.52 |
| BBBP: Impermeable | Overall | 84.80 | 85.78 |
| | **Target** | 65.52 | 68.97 |
| | Non-target | 92.47 | 92.47 |

### Ablation Study

Classification accuracy (%) for CANE and its ablation variants. Bold values indicate the best result per row (including ties).

| Editing Target | Metric | w/o Co-activation | w/o Contrastive | w/o Both | **CANE (Full)** |
|---|---|---:|---:|---:|---:|
| BA3: House | Overall | **95.50** | 95.00 | 95.25 | **95.50** |
| | **House** | **91.73** | 90.98 | **91.73** | **91.73** |
| | Grid | **100.00** | **100.00** | **100.00** | **100.00** |
| | Cycle | **94.40** | 93.60 | 93.60 | **94.40** |
| Benzene: Benzene | Overall | 88.58 | 85.33 | 85.75 | **91.75** |
| | **Target** | **97.73** | 87.68 | 88.17 | 97.41 |
| | Non-target | 78.90 | 82.85 | 83.19 | **85.76** |
| Mutagenicity: Mutagenic | Overall | **72.20** | 68.40 | 68.40 | **72.20** |
| | **Target** | 77.15 | **84.64** | **84.64** | 77.15 |
| | Non-target | **66.52** | 49.79 | 49.79 | **66.52** |
| BBBP: Impermeable | Overall | 85.29 | 84.80 | 84.80 | **85.78** |
| | **Target** | **68.97** | 65.52 | 67.24 | **68.97** |
| | Non-target | 91.78 | **92.47** | 91.78 | **92.47** |
