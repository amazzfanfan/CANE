"""Dataset classes, loaders, and knowledge-graph builders for CANE."""

import os
import os.path as osp
import random
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from numpy.random import RandomState
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
from sklearn import preprocessing
from torch.utils.data import ConcatDataset, Subset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import coalesce, to_networkx
from torch_scatter import scatter
from tqdm import tqdm

from model import load_pretrained_gin

RDLogger.DisableLog("rdApp.*")

BASE_DIR = osp.dirname(osp.abspath(__file__))
DATA_DIR = osp.join(BASE_DIR, "data")
MODEL_DIR = osp.join(BASE_DIR, "models")

# Atom and bond vocabularies for molecular datasets
MOLECULE_NUM_ATOM_TYPES = 14
BBBP_ATOM_TYPES = ["Cl", "C", "N", "O", "H", "F", "S", "Br", "I", "Na", "P", "Ca", "B"]
BBBP_ATOM_MAP = {sym: idx for idx, sym in enumerate(BBBP_ATOM_TYPES)}
BBBP_ATOMIC_NUMS = {
    "C": 6, "N": 7, "O": 8, "F": 9, "Cl": 17, "Br": 35,
    "S": 16, "P": 15, "I": 53, "Na": 11, "Ca": 20, "B": 5, "H": 1,
}
BBBP_BOND_MAP = {
    BT.SINGLE: 0,
    BT.DOUBLE: 1,
    BT.TRIPLE: 2,
    BT.AROMATIC: 3,
    BT.DATIVE: 4,
}


# ---------------------------------------------------------------------------
# PyG InMemoryDataset implementations
# ---------------------------------------------------------------------------

class BA3Motif(InMemoryDataset):
    """Synthetic 3-class motif dataset (House, Grid, Cycle)."""

    splits = ["training", "evaluation", "testing"]

    def __init__(self, root: str, mode: str = "testing", transform=None, pre_transform=None, pre_filter=None):
        assert mode in self.splits, f"Invalid mode: {mode}"
        self.mode = mode
        super().__init__(root, transform, pre_transform, pre_filter)
        idx = self.processed_file_names.index(f"{mode}.pt")
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self) -> List[str]:
        return ["BA-3motif.npy"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["training.pt", "evaluation.pt", "testing.pt"]

    def download(self):
        if all(osp.exists(p) for p in self.processed_paths):
            return
        raw_file = osp.join(self.raw_dir, self.raw_file_names[0])
        if not osp.exists(raw_file):
            raise FileNotFoundError(f"Missing raw BA3 dataset file: {raw_file}")

    def process(self):
        edge_index_list, label_list, ground_truth_list, role_id_list, pos = np.load(
            osp.join(self.raw_dir, self.raw_file_names[0]), allow_pickle=True
        )
        data_list = []
        alpha = 0.25
        for idx, (edge_index, y, ground_truth, z, p) in enumerate(
            zip(edge_index_list, label_list, ground_truth_list, role_id_list, pos)
        ):
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            node_idx = torch.unique(edge_index)
            num_nodes = node_idx.size(0)
            x = torch.zeros(num_nodes, 4)
            x[list(range(num_nodes)), z] = 1
            x = alpha * x + (1 - alpha) * torch.rand((num_nodes, 4))
            edge_attr = torch.ones(edge_index.size(1), 1)
            y_tensor = torch.tensor(y, dtype=torch.long).unsqueeze(dim=0)
            pos_arr = np.array(list(p.values()))

            data = Data(
                x=x,
                y=y_tensor,
                z=z,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=pos_arr,
                ground_truth_mask=ground_truth,
                name=f"BA-3motif{idx}",
                idx=idx,
            )
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        random.shuffle(data_list)
        torch.save(self.collate(data_list[800:]), self.processed_paths[0])
        torch.save(self.collate(data_list[400:800]), self.processed_paths[1])
        torch.save(self.collate(data_list[:400]), self.processed_paths[2])


class Benzene(InMemoryDataset):
    """Benzene molecular graph classification dataset (0: Non-benzene, 1: Benzene)."""

    def __init__(self, root: str, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices, self.index, self.sizes = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["benzene.npz"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self):
        if osp.exists(self.processed_paths[0]):
            return
        raw_path = osp.join(self.raw_dir, self.raw_file_names[0])
        if not osp.exists(raw_path):
            raise FileNotFoundError(
                f"Missing {raw_path}. Preprocessed data is expected at {self.processed_paths[0]}."
            )

    @staticmethod
    def _edge_mask_from_node_mask(node_mask: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        node_numbers = node_mask.nonzero(as_tuple=True)[0]
        iter_mask = torch.zeros((edge_index.shape[1],))
        for i in range(edge_index.shape[1]):
            iter_mask[i] = (edge_index[0, i] in node_numbers) and (edge_index[1, i] in node_numbers)
        return iter_mask.long()

    def process(self):
        raw = np.load(osp.join(self.raw_dir, self.raw_file_names[0]), allow_pickle=True)
        att, graphs_x, labels = raw["attr"], raw["X"][0], raw["y"]
        y_list = [labels[i][0] for i in range(labels.shape[0])]

        data_list = []
        for i in range(len(graphs_x)):
            x = torch.from_numpy(graphs_x[i]["nodes"])
            edge_attr = torch.from_numpy(graphs_x[i]["edges"])
            y = torch.tensor([y_list[i]], dtype=torch.long)
            edge_index = torch.stack([
                torch.from_numpy(graphs_x[i]["receivers"]).long(),
                torch.from_numpy(graphs_x[i]["senders"]).long(),
            ])

            node_imp = torch.from_numpy(att[i][0]["nodes"]).float()
            node_mask = torch.zeros(x.shape[0], dtype=torch.long)
            edge_mask = torch.zeros(edge_index.shape[1], dtype=torch.long)
            for j in range(node_imp.shape[1]):
                node_mask = torch.bitwise_or(node_mask, node_imp[:, j].long())
                edge_mask = torch.bitwise_or(
                    edge_mask, self._edge_mask_from_node_mask(node_imp[:, j].bool(), edge_index)
                )

            edge_index, edge_attrs = coalesce(edge_index, [edge_attr, edge_mask], x.shape[0])
            data = Data(
                x=x,
                y=y,
                edge_index=edge_index,
                edge_attr=edge_attrs[0],
                edge_mask=edge_attrs[1],
                node_mask=node_mask,
            )
            data_list.append(data)

        num_graphs = len(data_list)
        indices = RandomState(42).permutation(np.arange(num_graphs))
        data_list = [data_list[i] for i in indices]
        idx = torch.arange(num_graphs)
        index = {
            "train_idx": idx[: int(0.8 * num_graphs)],
            "test_idx": idx[int(0.8 * num_graphs) : int(0.9 * num_graphs)],
            "val_idx": idx[int(0.9 * num_graphs) :],
        }
        sizes = {
            "num_node_labels": data_list[0].x.size(-1),
            "num_edge_labels": data_list[0].edge_attr.size(-1),
            "num_classes": 2,
        }
        data, slices = self.collate(data_list)
        torch.save((data, slices, index, sizes), self.processed_paths[0])


class Mutagenicity(InMemoryDataset):
    """Mutagenicity molecular dataset (0: Mutagenic, 1: Non-mutagenic)."""

    splits = ["training", "evaluation", "testing"]

    def __init__(self, root: str, mode: str = "testing", transform=None, pre_transform=None, pre_filter=None):
        assert mode in self.splits, f"Invalid mode: {mode}"
        self.mode = mode
        super().__init__(root, transform, pre_transform, pre_filter)
        idx = self.processed_file_names.index(f"{mode}.pt")
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            osp.join("Mutagenicity", fname)
            for fname in [
                "Mutagenicity_A.txt",
                "Mutagenicity_edge_labels.txt",
                "Mutagenicity_graph_indicator.txt",
                "Mutagenicity_graph_labels.txt",
                "Mutagenicity_node_labels.txt",
            ]
        ]

    @property
    def processed_file_names(self) -> List[str]:
        return ["training.pt", "evaluation.pt", "testing.pt"]

    def download(self):
        if all(osp.exists(p) for p in self.processed_paths):
            return
        if not osp.exists(osp.join(self.raw_dir, "Mutagenicity")):
            raise FileNotFoundError(f"Missing raw Mutagenicity files in {self.raw_dir}")

    def process(self):
        edge_index = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[0]), delimiter=",").T
        edge_index = torch.from_numpy(edge_index - 1.0).to(torch.long)

        edge_label = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[1]))
        edge_enc = preprocessing.OneHotEncoder().fit(np.unique(edge_label).reshape(-1, 1))
        edge_attr = torch.Tensor(edge_enc.transform(edge_label.reshape(-1, 1)).toarray())

        node_label = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[-1]))
        node_enc = preprocessing.OneHotEncoder().fit(np.unique(node_label).reshape(-1, 1))
        x = torch.Tensor(node_enc.transform(node_label.reshape(-1, 1)).toarray())

        graph_indicator = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[2]), dtype=int)
        y = torch.unsqueeze(torch.LongTensor(np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[3]))), 1).long()

        total_edges = edge_index.size(1)
        begin = 0
        data_list = []
        for i in range(len(y)):
            perm = np.where(graph_indicator == i + 1)[0]
            bound = max(perm)
            end = begin
            for end in range(begin, total_edges):
                if int(edge_index[0, end]) > bound:
                    break

            data = Data(
                x=x[perm],
                y=y[i],
                edge_index=edge_index[:, begin:end] - int(min(perm)),
                edge_attr=edge_attr[begin:end],
            )
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            begin = end
            data_list.append(data)

        random.shuffle(data_list)
        torch.save(self.collate(data_list[1000:]), self.processed_paths[0])
        torch.save(self.collate(data_list[500:1000]), self.processed_paths[1])
        torch.save(self.collate(data_list[:500]), self.processed_paths[2])


class BBBP(InMemoryDataset):
    """Blood-Brain Barrier Penetration dataset (0: Impermeable, 1: Permeable)."""

    def __init__(self, root: str, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["rdkit_graph.sdf", "rdkit_graph.sdf.csv"]

    @property
    def processed_file_names(self) -> str:
        return "data_v3_rdkit2d.pt"

    def download(self):
        missing = [f for f in self.raw_file_names if not osp.exists(osp.join(self.raw_dir, f))]
        if missing:
            raise FileNotFoundError(f"Missing raw BBBP files in {self.raw_dir}: {missing}")

    def process(self):
        with open(self.raw_paths[1], "r", encoding="utf-8") as handle:
            lines = handle.read().split("\n")[1:-1]
            target = [[float(line.split(",")[0]) if line.split(",")[0] != "" else -1.0] for line in lines]
        y_all = torch.tensor(target, dtype=torch.float)

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)
        data_list = []
        for i, mol in enumerate(tqdm(suppl, desc="Processing BBBP molecules")):
            if mol is None:
                continue

            num_atoms = mol.GetNumAtoms()
            pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)
            if torch.unique(pos, dim=0).size(0) != num_atoms:
                continue

            type_idx, atomic_number, aromatic = [], [], []
            sp, sp2, sp3 = [], [], []
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                type_idx.append(BBBP_ATOM_MAP.get(symbol, BBBP_ATOM_MAP["C"]))
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hyb = atom.GetHybridization()
                sp.append(1 if hyb == HybridizationType.SP else 0)
                sp2.append(1 if hyb == HybridizationType.SP2 else 0)
                sp3.append(1 if hyb == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)

            row, col, edge_type_list = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type_list += 2 * [BBBP_BOND_MAP[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type_list, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(BBBP_BOND_MAP)).to(torch.float)

            if edge_index.numel() > 0:
                perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_attr = edge_attr[perm]
                row_idx, col_idx = edge_index
                hs = (z == 1).to(torch.float)
                num_hs = scatter(hs[row_idx], col_idx, dim_size=num_atoms, reduce="sum").tolist()
            else:
                num_hs = [0.0] * num_atoms

            x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(BBBP_ATOM_TYPES))
            x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs], dtype=torch.float).t().contiguous()
            x = torch.cat([x1, x2], dim=-1)

            y_val = y_all[i].unsqueeze(0) if i < len(y_all) else torch.tensor([[-1.0]])
            data_list.append(
                Data(
                    x=x,
                    pos=pos,
                    z=z,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    name=mol.GetProp("_Name"),
                    index=i,
                    smiles=Chem.MolToSmiles(mol, isomericSmiles=True),
                    y=y_val,
                )
            )

        torch.save(self.collate(data_list), self.processed_paths[0])


# ---------------------------------------------------------------------------
# Shared helper functions
# ---------------------------------------------------------------------------

def _split_by_class(dataset, n_classes: int) -> Tuple[Dict[int, List[Data]], Dict[int, DataLoader]]:
    """Group graphs by integer class label and wrap each group in a DataLoader."""
    by_class_data: Dict[int, List[Data]] = {}
    by_class_loaders: Dict[int, DataLoader] = {}
    for class_id in range(n_classes):
        graphs = [g for g in dataset if int(g.y.view(-1)[0].item()) == class_id]
        by_class_data[class_id] = graphs
        by_class_loaders[class_id] = DataLoader(graphs, batch_size=1, shuffle=False)
    return by_class_data, by_class_loaders


def _one_hot(idx: int, num_classes: int) -> torch.Tensor:
    vec = torch.zeros(num_classes, dtype=torch.float)
    if 0 <= idx < num_classes:
        vec[idx] = 1.0
    return vec


def _perturb_graph_add_carbon(
    graph_data: Data,
    num_nodes_to_add: int = 1,
    num_edges_to_add: int = 1,
) -> Data:
    """Attach random carbon nodes and edges to augment a molecular graph."""
    perturbed = graph_data.clone()
    original_num_nodes = perturbed.num_nodes
    if original_num_nodes == 0:
        return perturbed

    num_new_nodes = random.randint(0, num_nodes_to_add)
    if num_new_nodes == 0:
        return perturbed

    edge_dim = perturbed.edge_attr.size(1) if (perturbed.edge_attr is not None and perturbed.edge_attr.numel() > 0) else 4
    carbon_feat = _one_hot(0, perturbed.x.size(1))
    new_node_feats = torch.stack([carbon_feat.clone() for _ in range(num_new_nodes)], dim=0)
    perturbed.x = torch.cat([perturbed.x, new_node_feats], dim=0)

    total_num_nodes = perturbed.num_nodes
    candidate_old_nodes = list(range(original_num_nodes))
    new_edges, new_edge_attrs = [], []

    # Connect every new carbon node to at least one existing node
    for new_idx in range(original_num_nodes, total_num_nodes):
        old_idx = random.choice(candidate_old_nodes)
        new_edges.extend([[new_idx, old_idx], [old_idx, new_idx]])
        bond_attr = _one_hot(random.randint(0, 2), edge_dim)
        new_edge_attrs.extend([bond_attr, bond_attr])

    # Add optional extra edges
    max_extra_edges = max(0, random.randint(0, num_edges_to_add) - num_new_nodes)
    for _ in range(max_extra_edges):
        new_idx = random.randint(original_num_nodes, total_num_nodes - 1)
        old_idx = random.choice(candidate_old_nodes)
        new_edges.extend([[new_idx, old_idx], [old_idx, new_idx]])
        bond_attr = _one_hot(random.randint(0, 2), edge_dim)
        new_edge_attrs.extend([bond_attr, bond_attr])

    if new_edges:
        new_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
        new_edge_attr = torch.stack(new_edge_attrs, dim=0)
        perturbed.edge_index = torch.cat([perturbed.edge_index, new_edge_index], dim=1)
        if perturbed.edge_attr is not None:
            perturbed.edge_attr = torch.cat([perturbed.edge_attr, new_edge_attr], dim=0)

    return perturbed


def _augment_graphs(graphs: List[Data], p: float, num_nodes_to_add: int, num_edges_to_add: int) -> List[Data]:
    augmented = []
    for graph in graphs:
        if random.random() < p:
            augmented.append(_perturb_graph_add_carbon(graph, num_nodes_to_add, num_edges_to_add))
        else:
            augmented.append(graph)
    return augmented


# ---------------------------------------------------------------------------
# Dataset-specific knowledge builders and preparation entry points
# ---------------------------------------------------------------------------

def _extract_ba3_connected_motifs(data: Data) -> List[Data]:
    """Extract connected motif subgraphs (nodes with z != 0) from a BA3 graph."""
    z_tensor = data.z if torch.is_tensor(data.z) else torch.tensor(data.z)
    non_bg_indices = torch.where(z_tensor != 0)[0].tolist()
    if not non_bg_indices:
        return []

    g_motif = to_networkx(data, to_undirected=True).subgraph(non_bg_indices)
    motifs = []
    for comp in nx.connected_components(g_motif):
        comp_nodes = sorted(comp)
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(comp_nodes)}
        new_edges = []
        for i in range(data.edge_index.size(1)):
            src = data.edge_index[0, i].item()
            dst = data.edge_index[1, i].item()
            if src in node_map and dst in node_map:
                new_edges.append([node_map[src], node_map[dst]])
        if new_edges:
            motifs.append(
                Data(
                    x=data.x[comp_nodes],
                    edge_index=torch.tensor(new_edges, dtype=torch.long).t().contiguous(),
                    y=data.y,
                    z=z_tensor[comp_nodes],
                )
            )
    return motifs


def _sample_graphs(graphs: List[Data], sample_ratio: float = 0.7) -> List[Data]:
    num_samples = int(len(graphs) * sample_ratio)
    sampled_indices = random.sample(range(len(graphs)), num_samples)
    return [graphs[i] for i in sampled_indices]


def prepare_ba3(device: torch.device) -> Dict:
    """Prepare BA3 dataset, House target knowledge, contrast motifs, and pretrained GIN."""
    root = osp.join(DATA_DIR, "BA3")
    train_dataset = BA3Motif(root, mode="training")
    test_dataset = BA3Motif(root, mode="testing")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    train_by_class, train_loaders_by_class = _split_by_class(train_dataset, n_classes=3)
    _, test_loaders_by_class = _split_by_class(test_dataset, n_classes=3)

    # Extract motif subgraphs for non-target classes and sample 70% for knowledge sets
    grid_motifs = [m for g in train_by_class[1] for m in _extract_ba3_connected_motifs(g)]
    cycle_motifs = [m for g in train_by_class[2] for m in _extract_ba3_connected_motifs(g)]

    house_knowledge = _sample_graphs(train_by_class[0], sample_ratio=0.7)
    grid_knowledge = _sample_graphs(grid_motifs, sample_ratio=0.7)
    cycle_knowledge = _sample_graphs(cycle_motifs, sample_ratio=0.7)

    knowledge_target = DataLoader(house_knowledge, batch_size=32, shuffle=False)
    knowledge_contrast = [
        DataLoader(cycle_knowledge, batch_size=32, shuffle=False),
        DataLoader(grid_knowledge, batch_size=32, shuffle=False),
    ]
    locality_background = DataLoader(
        ConcatDataset([train_loaders_by_class[1].dataset, train_loaders_by_class[2].dataset]),
        batch_size=32,
        shuffle=True,
    )

    model = load_pretrained_gin(
        checkpoint_path=osp.join(MODEL_DIR, "BA3-GIN-fold3.pth"),
        n_features=4,
        n_classes=3,
        device=device,
    )

    return {
        "dataset": "ba3",
        "model": model,
        "knowledge_target": knowledge_target,
        "knowledge_contrast": knowledge_contrast,
        "locality_background": locality_background,
        "test_loader": test_loader,
        "test_by_class": test_loaders_by_class,
        "target_class": 0,
        "class_names": {0: "House", 1: "Grid", 2: "Cycle"},
    }


def _build_benzene_ring_graph() -> Data:
    """Construct a 6-carbon aromatic benzene ring graph (label 1)."""
    x = torch.stack([_one_hot(0, MOLECULE_NUM_ATOM_TYPES) for _ in range(6)])
    edges, edge_attrs = [], []
    for i in range(6):
        u, v = i, (i + 1) % 6
        edges.extend([[u, v], [v, u]])
        # Aromatic bond index = 3 in 4-dim bond encoding
        edge_attrs.extend([_one_hot(3, 4), _one_hot(3, 4)])
    return Data(
        x=x,
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr=torch.stack(edge_attrs).float(),
        y=torch.tensor([1], dtype=torch.long),
    )


def _build_random_nonbenzene_graph() -> Data:
    """Construct a random non-benzene molecular graph (tree, branched, non-6-cycle, or chain)."""
    graph_type = random.choice(["tree", "branched", "small_cycle", "chain"])
    atom_types = list(range(MOLECULE_NUM_ATOM_TYPES))
    edges, edge_attrs = [], []

    if graph_type == "tree":
        num_nodes = random.randint(4, 10)
        x = torch.stack([_one_hot(random.choice(atom_types), MOLECULE_NUM_ATOM_TYPES) for _ in range(num_nodes)])
        for i in range(1, num_nodes):
            parent = random.randint(0, i - 1)
            bond = _one_hot(random.choice([0, 1]), 3)
            edges.extend([[parent, i], [i, parent]])
            edge_attrs.extend([bond, bond])
    elif graph_type == "branched":
        num_nodes = random.randint(5, 12)
        x = torch.stack([_one_hot(random.choice(atom_types), MOLECULE_NUM_ATOM_TYPES) for _ in range(num_nodes)])
        num_branches = random.randint(2, min(4, num_nodes - 1))
        branch_heads = random.sample(range(1, num_nodes), num_branches)
        for head in branch_heads:
            bond = _one_hot(random.choice([0, 1]), 3)
            edges.extend([[0, head], [head, 0]])
            edge_attrs.extend([bond, bond])
        for node in [i for i in range(1, num_nodes) if i not in branch_heads]:
            parent = random.choice(list(range(node)))
            bond = _one_hot(random.choice([0, 1]), 3)
            edges.extend([[parent, node], [node, parent]])
            edge_attrs.extend([bond, bond])
    elif graph_type == "small_cycle":
        cycle_size = random.choice([3, 4, 5, 7])
        num_nodes = cycle_size + random.randint(0, 3)
        x = torch.stack([_one_hot(random.choice(atom_types), MOLECULE_NUM_ATOM_TYPES) for _ in range(num_nodes)])
        for i in range(cycle_size):
            u, v = i, (i + 1) % cycle_size
            bond = _one_hot(0, 3)
            edges.extend([[u, v], [v, u]])
            edge_attrs.extend([bond, bond])
        for i in range(cycle_size, num_nodes):
            parent = random.randint(0, i - 1)
            bond = _one_hot(0, 3)
            edges.extend([[parent, i], [i, parent]])
            edge_attrs.extend([bond, bond])
    else:
        num_nodes = random.randint(4, 10)
        x = torch.stack([_one_hot(0, MOLECULE_NUM_ATOM_TYPES) for _ in range(num_nodes)])
        for i in range(num_nodes - 1):
            bond = _one_hot(0, 3)
            edges.extend([[i, i + 1], [i + 1, i]])
            edge_attrs.extend([bond, bond])

    return Data(
        x=x,
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr=torch.stack(edge_attrs).float(),
        y=torch.tensor([0], dtype=torch.long),
    )


def prepare_benzene(device: torch.device) -> Dict:
    """Prepare Benzene dataset, constructed benzene/non-benzene knowledge, and pretrained GIN."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    dataset = Benzene(root=osp.join(DATA_DIR, "Benzene")).shuffle()
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    train_dataset = dataset[:n_train]
    test_dataset = dataset[n_train + n_val :]
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Construct augmented benzene target knowledge and non-benzene contrast knowledge
    benzene_graphs = [_build_benzene_ring_graph() for _ in range(400)]
    benzene_graphs = _augment_graphs(benzene_graphs, p=0.7, num_nodes_to_add=5, num_edges_to_add=4)
    knowledge_target = DataLoader(benzene_graphs[: int(0.8 * len(benzene_graphs))], batch_size=4, shuffle=True)

    nonbenzene_graphs = [_build_random_nonbenzene_graph() for _ in range(400)]
    knowledge_contrast = [
        DataLoader(nonbenzene_graphs[: int(0.8 * len(nonbenzene_graphs))], batch_size=32, shuffle=True)
    ]

    _, train_loaders_by_class = _split_by_class(train_dataset, n_classes=2)
    _, test_loaders_by_class = _split_by_class(test_dataset, n_classes=2)

    model = load_pretrained_gin(
        checkpoint_path=osp.join(MODEL_DIR, "BENZENE-GIN-fold0.pth"),
        n_features=14,
        n_classes=2,
        device=device,
    )

    return {
        "dataset": "benzene",
        "model": model,
        "knowledge_target": knowledge_target,
        "knowledge_contrast": knowledge_contrast,
        "locality_background": train_loaders_by_class[0],
        "test_loader": test_loader,
        "test_by_class": test_loaders_by_class,
        "target_class": 1,
        "class_names": {0: "Non-benzene", 1: "Benzene"},
    }


def _extract_three_membered_heterocycles(train_loader: DataLoader, batch_size: int = 32) -> DataLoader:
    """Extract training molecules containing a 3-membered heterocycle (cycle of length 3 with O, N, or S)."""
    matched_graphs = []
    for batch in train_loader:
        for data in batch.to_data_list():
            if data.edge_index is None or data.edge_index.numel() == 0:
                continue
            node_types = data.x.argmax(dim=1).cpu().numpy()
            edge_idx = data.edge_index.cpu().numpy()

            graph_nx = nx.Graph()
            for idx, atom_type in enumerate(node_types):
                graph_nx.add_node(idx, atom=int(atom_type))
            for i in range(edge_idx.shape[1]):
                graph_nx.add_edge(int(edge_idx[0, i]), int(edge_idx[1, i]))

            has_heterocycle = False
            for cycle in nx.cycle_basis(graph_nx):
                if len(cycle) == 3:
                    cycle_atoms = [graph_nx.nodes[n]["atom"] for n in cycle]
                    # O=1, N=4, S=7 in Mutagenicity atom map
                    if any(atom in (1, 4, 7) for atom in cycle_atoms):
                        has_heterocycle = True
                        break
            if has_heterocycle:
                matched_graphs.append(data)

    return DataLoader(matched_graphs, batch_size=batch_size, shuffle=True)


def _extract_non_nh2_loader(dataset, batch_size: int = 32) -> DataLoader:
    """Select training graphs that do not contain an NH2 group as contrastive background."""
    non_nh2_indices = []
    for idx in range(len(dataset)):
        data = dataset[idx]
        node_types = data.x.argmax(dim=1)
        n_nodes = (node_types == 4).nonzero(as_tuple=True)[0]
        has_nh2 = False
        for n_idx in n_nodes:
            neighbor_indices = data.edge_index[1][data.edge_index[0] == n_idx]
            num_h = (node_types[neighbor_indices] == 3).sum().item()
            if num_h >= 2:
                has_nh2 = True
                break
        if not has_nh2:
            non_nh2_indices.append(idx)

    return DataLoader(Subset(dataset, non_nh2_indices), batch_size=batch_size, shuffle=True)


def prepare_mutagenicity(device: torch.device) -> Dict:
    """Prepare Mutagenicity dataset, 3-membered heterocycle target knowledge, and pretrained GIN."""
    np.random.seed(1)
    torch.manual_seed(123)

    root = osp.join(DATA_DIR, "Mutagenicity")
    train_dataset = Mutagenicity(root, mode="training")
    test_dataset = Mutagenicity(root, mode="testing")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    knowledge_target = _extract_three_membered_heterocycles(train_loader, batch_size=32)
    knowledge_contrast = [_extract_non_nh2_loader(train_dataset, batch_size=32)]

    _, train_loaders_by_class = _split_by_class(train_dataset, n_classes=2)
    _, test_loaders_by_class = _split_by_class(test_dataset, n_classes=2)

    model = load_pretrained_gin(
        checkpoint_path=osp.join(MODEL_DIR, "MUTAGENICITY-GIN-fold1.pth"),
        n_features=14,
        n_classes=2,
        device=device,
    )

    return {
        "dataset": "mutagenicity",
        "model": model,
        "knowledge_target": knowledge_target,
        "knowledge_contrast": knowledge_contrast,
        "locality_background": train_loaders_by_class[1],
        "test_loader": test_loader,
        "test_by_class": test_loaders_by_class,
        "target_class": 0,
        "class_names": {0: "Mutagenic", 1: "Non-mutagenic"},
    }


def _fix_bbbp_target(data: Data) -> Data:
    if data.y is not None:
        data.y = data.y.view(-1).to(torch.long)
    return data


def _bbbp_atom_feature(symbol: str, is_aromatic: bool = False, num_hs: float = 0.0) -> torch.Tensor:
    one_hot = [0.0] * len(BBBP_ATOM_TYPES)
    one_hot[BBBP_ATOM_MAP[symbol]] = 1.0
    atomic_num = float(BBBP_ATOMIC_NUMS.get(symbol, 6))
    aro = 1.0 if is_aromatic else 0.0

    sp, sp2, sp3 = 0.0, 0.0, 0.0
    if is_aromatic:
        sp2 = 1.0
    elif symbol in ("C", "N", "O", "S", "P"):
        sp3 = 1.0
    if symbol == "O" and num_hs == 0:
        sp2, sp3 = 1.0, 0.0

    return torch.tensor(one_hot + [atomic_num, aro, sp, sp2, sp3, float(num_hs)], dtype=torch.float)


def _build_tertiary_amine_graph_bbbp(label: int = 1) -> Data:
    """Construct a trimethylamine N(CH3)3 graph with 19-dim BBBP atom features."""
    x_list = [_bbbp_atom_feature("N", is_aromatic=False, num_hs=0.0)]
    for _ in range(3):
        x_list.append(_bbbp_atom_feature("C", is_aromatic=False, num_hs=3.0))
    x = torch.stack(x_list)

    edges, edge_attrs = [], []
    for i in (1, 2, 3):
        edges.extend([[0, i], [i, 0]])
        edge_attrs.extend([_one_hot(0, 5), _one_hot(0, 5)])

    return Data(
        x=x,
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr=torch.stack(edge_attrs).float(),
        y=torch.tensor([label], dtype=torch.long),
    )


def _find_best_impermeable_prototype(model: torch.nn.Module, dataset, device: torch.device) -> Data:
    """Find the real Impermeable (class 0) molecule with highest predicted class-0 probability."""
    model.eval()
    best_prob = -1.0
    best_idx = -1
    with torch.no_grad():
        for idx in range(len(dataset)):
            data = dataset[idx]
            if int(data.y.view(-1)[0].item()) != 0:
                continue
            batch_vec = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
            logits = model(data.x.to(device), data.edge_index.to(device), batch_vec)
            prob_0 = F.softmax(logits, dim=1)[0, 0].item()
            if prob_0 > best_prob:
                best_prob = prob_0
                best_idx = idx
    return dataset[best_idx].clone().cpu()


def prepare_bbbp(device: torch.device) -> Dict:
    """Prepare BBBP dataset, prototype-augmented impermeable knowledge, and pretrained GIN."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    dataset = BBBP(root=osp.join(DATA_DIR, "BBBP"), transform=_fix_bbbp_target).shuffle()
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    train_dataset = dataset[:n_train]
    test_dataset = dataset[n_train + n_val :]
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Contrastive Permeable (class 1) tertiary amine graphs
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    amine_graphs = [_build_tertiary_amine_graph_bbbp(label=1) for _ in range(400)]
    amine_graphs = _augment_graphs(amine_graphs, p=0.5, num_nodes_to_add=3, num_edges_to_add=2)
    knowledge_contrast = [DataLoader(amine_graphs[: int(0.8 * len(amine_graphs))], batch_size=32, shuffle=True)]

    # Target Impermeable (class 0) prototype + carbon-attachment augmentation
    proto_model = load_pretrained_gin(
        checkpoint_path=osp.join(MODEL_DIR, "BBBP-GIN-fold0.pth"),
        n_features=19,
        n_classes=2,
        device=device,
    )
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    prototype = _find_best_impermeable_prototype(proto_model, dataset, device=device)
    np_graphs = [prototype.clone() for _ in range(400)]
    np_graphs = _augment_graphs(np_graphs, p=0.8, num_nodes_to_add=3, num_edges_to_add=5)
    knowledge_target = DataLoader(np_graphs[: int(0.8 * len(np_graphs))], batch_size=32, shuffle=True)

    _, train_loaders_by_class = _split_by_class(train_dataset, n_classes=2)
    _, test_loaders_by_class = _split_by_class(test_dataset, n_classes=2)

    model = load_pretrained_gin(
        checkpoint_path=osp.join(MODEL_DIR, "BBBP-GIN-fold1.pth"),
        n_features=19,
        n_classes=2,
        device=device,
    )

    return {
        "dataset": "bbbp",
        "model": model,
        "knowledge_target": knowledge_target,
        "knowledge_contrast": knowledge_contrast,
        "locality_background": train_loaders_by_class[1],
        "test_loader": test_loader,
        "test_by_class": test_loaders_by_class,
        "target_class": 0,
        "class_names": {0: "Impermeable", 1: "Permeable"},
    }


def prepare_dataset(name: str, device: torch.device) -> Dict:
    """Load dataset, knowledge graphs, and pretrained model by name."""
    name = name.lower()
    if name == "ba3":
        return prepare_ba3(device)
    if name == "benzene":
        return prepare_benzene(device)
    if name == "mutagenicity":
        return prepare_mutagenicity(device)
    if name == "bbbp":
        return prepare_bbbp(device)
    raise ValueError(f"Unsupported dataset: {name}. Choose from ba3, benzene, mutagenicity, bbbp.")
