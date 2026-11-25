"""Graph Isomorphism Network (GIN) classifier for graph classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_max_pool, global_mean_pool


class GINClassifier(nn.Module):
    """Multi-layer GIN model for graph-level classification.

    Parameter names (`convs.*`, `lin1.*`) match the pretrained checkpoints.
    """

    def __init__(
        self,
        n_features: int,
        n_units: int,
        n_layers: int,
        n_classes: int,
        aggr_type: str = "sum",
    ):
        super().__init__()
        self.n_features = n_features
        self.n_units = n_units
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.aggr_type = aggr_type

        self.convs = nn.ModuleList()
        self.convs.append(self._make_gin_layer(n_features, n_units))
        for _ in range(n_layers - 1):
            self.convs.append(self._make_gin_layer(n_units, n_units))

        self.lin1 = nn.Linear(n_units, n_classes)

    @staticmethod
    def _make_gin_layer(in_dim: int, out_dim: int) -> GINConv:
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )
        return GINConv(nn=mlp)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None,
    ) -> torch.Tensor:
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if self.aggr_type == "sum":
            x = global_add_pool(x, batch)
        elif self.aggr_type == "max":
            x = global_max_pool(x, batch)
        elif self.aggr_type == "mean":
            x = global_mean_pool(x, batch)
        else:
            raise ValueError(f"Unsupported aggregation type: {self.aggr_type}")

        return self.lin1(x)


def load_pretrained_gin(
    checkpoint_path: str,
    n_features: int,
    n_classes: int,
    device: torch.device,
    n_units: int = 64,
    n_layers: int = 3,
    aggr_type: str = "sum",
) -> GINClassifier:
    """Instantiate a GINClassifier and load pretrained weights onto `device`."""
    model = GINClassifier(
        n_features=n_features,
        n_units=n_units,
        n_layers=n_layers,
        n_classes=n_classes,
        aggr_type=aggr_type,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model
