from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class MolState:

    G: nx.Graph
    R: np.ndarray
    node_feat: np.ndarray

    def num_nodes(self):
        return int(self.R.shape[0])

    def copy_with(self,  G=None, R=None, node_feat=None):
        return MolState(
            G=G if G is not None else self.G.copy(),
            R=np.array(R if R is not None else self.R, copy=True),
            node_feat=np.array(node_feat if node_feat is not None else self.node_feat, copy=True),
        )

    def distance_to(self, other, weights=None):
        from utils.distances import d_mcm
        return float(d_mcm(self, other, weights=weights))
