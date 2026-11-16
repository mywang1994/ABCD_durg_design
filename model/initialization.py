"""Initial-group sampling and admission to the training attraction basin."""
from __future__ import annotations

from contextlib import contextmanager
import math

import numpy as np
import torch

from dataset.sampling import MapGroup, _key, edits_ok, is_valid
from utils.distances import d_mcm, molstate_to_mol


INITIALIZATION_GROUP_SCHEMA = "admitted_eq18_then_scaffold_pose_v1"
SCAFFOLD_POSE_MAX_DISPLACEMENT = .15
SCAFFOLD_POSE_MAX_ROTATION_DEGREES = 5.


def initialization_training_metadata(cfg):
    return {"schema": INITIALIZATION_GROUP_SCHEMA,
            "attempt_fraction": cfg.initialization_group_fraction,
            "raw_groups_per_attempt": 1,
            "raw_sampling": "conditional_seed_when_task_dropped_or_scaffold_absent_else_eq18_mixture",
            "admission": "distinct_chemical_intramolecular_geometry_tau_pairwise_edit_and_configured_pocket_fit",
            "fallback": "admitted_minimum_scaffold_rigid_pose_group_then_original_group",
            "scaffold_pose_max_displacement_angstrom": SCAFFOLD_POSE_MAX_DISPLACEMENT,
            "scaffold_pose_max_rotation_degrees": SCAFFOLD_POSE_MAX_ROTATION_DEGREES,
            "scaffold_pose_candidates_per_parent": cfg.grp_max - 1,
            "objective": "replace_group_input_without_changing_losses_or_weights",
            "raw_count_sampling": "predicted_detached",
            "operator_output_count": "target_teacher_for_existing_matched_supervision",
            "metric_denominator": "all_training_groups; divide_raw_sums_by_init_attempted_for_conditional_means",
            "scope": "admissible_subset_only_not_full_unfiltered_inference_distribution"}


def minimal_scaffold_prior(states):
    """Uniform prior over valid scaffolds with the fewest heavy atoms."""
    valid = []
    for state in states:
        if is_valid(state):
            count = molstate_to_mol(state).GetNumHeavyAtoms()
            if count > 0:
                valid.append((count, state))
    if not valid:
        raise ValueError("No valid scaffold-retention states are available for the structural prior; use seed mode explicitly")
    minimum = min(count for count, _ in valid)
    return [state for count, state in valid if count == minimum]


@contextmanager
def sampling_mode(*modules):
    """Restore each shared submodule's mode after detached sampling."""
    flags = {child: child.training for module in modules for child in module.modules()}
    try:
        for child in flags:
            child.training = False
        with torch.no_grad():
            yield
    finally:
        for child, flag in flags.items():
            child.training = flag


def sample_initial_group(seed, prior, hp, hc, cfg, rng, *, anchors=None,
                         num_atoms=None, structural_prior=None, alpha_prior=0.):
    if not 0 <= alpha_prior <= 1:
        raise ValueError("alpha_prior must lie in [0,1]")
    if alpha_prior > 0 and not structural_prior:
        raise ValueError("A nonzero structural mixture requires scaffold-retention states")
    size = rng.randint(cfg.grp_min, cfg.grp_max)
    states, sources = [], []
    with torch.no_grad():
        for _ in range(size):
            if rng.random() < alpha_prior:
                state = rng.choice(structural_prior)
                sources.append("scaffold_prior")
            else:
                z = prior.sample(hp, hc, num_samples=1)[0]
                state = seed.forward(z, anchors, hp, hc, num_nodes=num_atoms)
                sources.append("conditional_prior")
            states.append(state)
    return states, {"initial_group_size": size, "initial_sources": sources,
                    "initial_atom_counts": [state.num_nodes() for state in states],
                    "alpha_prior": alpha_prior}


def state_admission_reason(state, target, cfg, pocket):
    from utils.geometry_validation import evaluate_geometry
    from utils.pkt_fit import pkt_fit

    if not is_valid(state):
        return "chemical_or_nonfinite"
    if state.num_nodes() > cfg.max_nodes:
        return "atom_count"
    if not evaluate_geometry(molstate_to_mol(state))["intramolecular_valid"]:
        return "geometry"
    if d_mcm(state, target, cfg.molecular_distance) > cfg.tau:
        return "target_distance"
    if cfg.pocket_min is not None:
        if pocket is None:
            raise ValueError("Initialization admission requires the configured receptor/reference pocket")
        score = pkt_fit(state, pocket).tot
        if not math.isfinite(score) or score < cfg.pocket_min:
            return "pocket_fit"
    return None


def group_admission_reason(states, target, cfg, pocket):
    if not cfg.grp_min <= len(states) <= cfg.grp_max:
        return "group_size"
    if len({_key(state) for state in states}) != len(states):
        return "duplicate_member"
    for state in states:
        reason = state_admission_reason(state, target, cfg, pocket)
        if reason is not None:
            return reason
    if not edits_ok(states, cfg.delta_edit):
        return "pairwise_edit"
    return None


def _scaffold_pose(state, rng):
    axis = np.asarray([rng.gauss(0., 1.) for _ in range(3)])
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    angle = math.radians(rng.uniform(-SCAFFOLD_POSE_MAX_ROTATION_DEGREES,
                                     SCAFFOLD_POSE_MAX_ROTATION_DEGREES))
    cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    centered = state.R - state.R.mean(axis=0)
    radius = float(np.linalg.norm(centered, axis=-1).max())
    angle = math.copysign(min(abs(angle), SCAFFOLD_POSE_MAX_DISPLACEMENT / (2 * max(radius, 1e-12))), angle)
    rotation = np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)
    translation = axis * rng.uniform(.1, .5) * SCAFFOLD_POSE_MAX_DISPLACEMENT
    coordinates = centered @ rotation.T + state.R.mean(axis=0) + translation
    return state.copy_with(R=coordinates.astype(state.R.dtype))


def scaffold_pose_groups(scaffolds, target, cfg, pocket, rng):
    """Bounded geometric variants; the chemical graph and chirality are retained."""
    report = {"source_kind": "geo", "source_note": "minimum_scaffold_rigid_pose",
              "scaffold_candidates": len(scaffolds), "pose_candidates": 0,
              "accepted_pose_states": 0, "rejections": {}, "groups": 0}
    groups = []
    for scaffold in scaffolds:
        reason = state_admission_reason(scaffold, target, cfg, pocket)
        if reason is not None:
            report["rejections"][reason] = report["rejections"].get(reason, 0) + 1
            continue
        members, keys = [scaffold], {_key(scaffold)}
        for _ in range(cfg.grp_max - 1):
            pose = _scaffold_pose(scaffold, rng)
            report["pose_candidates"] += 1
            reason = state_admission_reason(pose, target, cfg, pocket)
            if reason is not None:
                report["rejections"][reason] = report["rejections"].get(reason, 0) + 1
            elif _key(pose) not in keys:
                members.append(pose)
                keys.add(_key(pose))
        if len(members) >= cfg.grp_min:
            groups.append(MapGroup(tuple(members), target))
            report["accepted_pose_states"] += len(members) - 1
    report["groups"] = len(groups)
    return groups, report
