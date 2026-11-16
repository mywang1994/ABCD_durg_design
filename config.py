"""Model, training, and sampling configuration."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from numbers import Integral, Real

HIDDEN = 512
N_LAYERS = 8
POCKET_HIDDEN = 512
POCKET_LAYERS = 8
PRIOR_HIDDEN = 512
SUB_LAYERS = 2
IMPLEMENTATION_REVISION = "public_v4"
DECODER_KIND = "graph_then_equivariant_attention_v1"
SEED_OBJECTIVE = "conditional_prior_state_distance_v1"
STATE_DISTANCE_SCHEMA = "gnn_euclidean_transport_mse_v1"
GRAPH_ENCODER_TRAINING = "fixed_initialization_v1"
LOSS_SCHEMA = "paper_chemistry_v4"
VALENCE_PENALTY_SCHEMA = "pair_normalized_smooth_l1_excess_v1"
SEED_PAIRING = "same_count_minibatch_assignment_v1"
GROUP_INPUT_SCHEMA = "member_balanced_set_attention_v1"
POCKET_SCORE_SCHEMA = "pharm_shape_vdw_repulsion_full_pocket_v2"
POCKET_RECEPTOR_SCOPE = "all_filtered_heavy_atoms_in_supplied_receptor"
POCKET_RADIUS_SOURCE = "matched_target_elements_v1"
POCKET_PHARM_SCHEMA = "matched_feature_centers_equal_channels_v1"
ANCHOR_KIND = "learned_topk_st_v1"


def _positive_integer(name, value, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _nonnegative(name, value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 4
    node_feat_dim: int = 16
    cond_dim: int = 64
    pocket_atoms: int = 2048
    test_pts: int = 64

    def __post_init__(self):
        for item in fields(self):
            _positive_integer(item.name, getattr(self, item.name))


@dataclass(frozen=True)
class ABCMConfig:
    schema_version: int = 2
    max_nodes: int = 128
    z_dim: int = 32
    edge_T: int = 20
    joint_steps: int = 3
    pocket_neighbors: int = 32
    pocket_step_scale: float = 0.2
    num_anchors: int = 4
    alpha_prior: float = 0.5
    initialization_group_fraction: float = 0.25
    group_train: bool = True
    tau: float = 0.92
    delta_edit: int = 2
    grp_min: int = 3
    grp_max: int = 6
    group_max_tries: int = 50
    pocket_min: float | None = None
    lambda_geom: float = 5.0
    w_target: float = 1.0
    single_target_fraction: float = 0.0
    w_basin: float = 0.3
    w_pocket: float = 0.1
    w_chem: float = 0.1
    b_seed: float = 0.15
    g_init: float = 0.15
    seed_match_samples: int = 2
    # Internal weights of L_chem.
    w_valence: float = 1.0
    w_atom: float = 1.0
    w_charge: float = 0.2
    w_hydrogen: float = 0.2
    w_adjacency: float = 0.2
    # Optional noisy-edge regularizers.
    w_edge: float = 0.0
    w_edge_op: float = 0.0
    w_bond: float = 1.0
    w_bond_length: float = 0.0
    w_count: float = 0.2
    w_feature: float = 0.0
    w_embedding: float = 0.0
    match_position_weight: float = 1.0
    match_atom_weight: float = 1.0
    task_dropout: float = 0.5
    molecular_distance: dict = field(default_factory=lambda: {
        "w_topo": 0.25, "w_scaffold": 0.25, "w_pharm": 0.25,
        "w_geom": 0.25, "fp_radius": 2, "fp_nbits": 2048,
    })
    # Independent Circles evaluation threshold.
    circles_threshold: float | None = None

    def __post_init__(self):
        if self.schema_version != 2:
            raise ValueError("Only schema_version=2 is supported")
        if self.group_train is not True:
            raise ValueError("Training requires single-step mapping groups")
        for name in ("max_nodes", "z_dim", "edge_T", "joint_steps", "pocket_neighbors", "num_anchors", "grp_min", "grp_max", "group_max_tries", "seed_match_samples"):
            _positive_integer(name, getattr(self, name))
        _positive_integer("delta_edit", self.delta_edit, minimum=0)
        if self.grp_min > self.grp_max:
            raise ValueError("grp_min cannot exceed grp_max")
        for item in fields(self):
            if item.name.startswith("w_") or item.name in (
                "lambda_geom", "b_seed", "g_init", "match_position_weight", "match_atom_weight", "pocket_step_scale",
            ):
                _nonnegative(item.name, getattr(self, item.name))
        if self.match_position_weight + self.match_atom_weight <= 0:
            raise ValueError("At least one matching cost must have positive weight")
        for name in ("tau", "task_dropout", "single_target_fraction", "alpha_prior", "initialization_group_fraction"):
            value = getattr(self, name)
            _nonnegative(name, value)
            if value > 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("pocket_min", "circles_threshold"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, Real) or not math.isfinite(value)):
                raise ValueError(f"{name} must be finite or None")
        if self.circles_threshold is not None and not 0 <= self.circles_threshold <= 1:
            raise ValueError("circles_threshold must lie in [0, 1]")
        expected = {"w_topo", "w_scaffold", "w_pharm", "w_geom", "fp_radius", "fp_nbits"}
        if not isinstance(self.molecular_distance, dict) or set(self.molecular_distance) != expected:
            raise ValueError(f"molecular_distance must contain exactly {sorted(expected)}")
        weights = [self.molecular_distance[key] for key in ("w_topo", "w_scaffold", "w_pharm", "w_geom")]
        for value in weights:
            _nonnegative("molecular_distance weight", value)
        if not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-8):
            raise ValueError("Molecular-distance weights must sum to one")
        _positive_integer("fp_radius", self.molecular_distance["fp_radius"], minimum=0)
        _positive_integer("fp_nbits", self.molecular_distance["fp_nbits"])
