"""Generate conditional molecules and record outcomes."""
from __future__ import annotations

import argparse
from collections import Counter
import dataclasses
import hashlib
import json
import math
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from rdkit import Chem

from config import (ABCMConfig, IMPLEMENTATION_REVISION, PRIOR_HIDDEN, SUB_LAYERS,
                    DECODER_KIND, SEED_OBJECTIVE, STATE_DISTANCE_SCHEMA, ANCHOR_KIND, GROUP_INPUT_SCHEMA,
                    POCKET_SCORE_SCHEMA, POCKET_RECEPTOR_SCOPE)
from dataset.protein_data import (CrossDockComplex, pocket_atoms, pocket_coords, task_vec,
                                  PROTEIN_FEATURE_DIM, PROTEIN_FEATURE_SCHEMA, TASK_VECTOR_REPRESENTATION)
from dataset.state_pool import build_crossdock_bundle
from dataset.pool_core import LocalMMPConfig, PoolCfg, load_reference_mol_from_sdf, mol_to_molstate
from dataset.pool_builders import validate_mmp_backend
from mol_state import MolState
from model import PocketEncoder, TaskEncoder, build_model
from model.anchor import top_anchors
from model.consistency_operator import Route, infer_grp_routes, infer_routes, is_recoverable_generation_error, state_from_step
from utils.chemistry import CHEMISTRY_SCHEMA_VERSION, encode_validated_state
from utils.distances import molstate_to_mol, tanimoto_molstate
from utils.geometry_validation import (GeometryValidationConfig, evaluate_geometry, load_receptor_geometry,
                                       not_evaluated_geometry, summarize_geometry)
from utils.pkt_fit import PktWt, pkt_fit
from model.initialization import minimal_scaffold_prior, sample_initial_group

STATE_POOL_TYPES = ("local", "pharm", "scaffold", "rxn", "geo")


def _clamp_basin(n):
    return max(0, min(20, int(n)))


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _json_safe(value):
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(path, manifest):
    """Publish a complete JSON snapshot atomically, including interrupted runs."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _cfg_from_ckpt(d: dict) -> ABCMConfig:
    if not isinstance(d, dict) or d.get("schema_version") != CHEMISTRY_SCHEMA_VERSION:
        raise ValueError("Checkpoint lacks the complete configuration required by the prediction heads")
    d = {"seed_match_samples": 1, "initialization_group_fraction": 0., **d}
    required = {field.name for field in dataclasses.fields(ABCMConfig)}
    legacy = (required - {"w_valence"}) | {"w_aux"}
    if set(d) == legacy:
        d = dict(d)
        auxiliary = d.pop("w_aux")
        weights = (auxiliary, d["w_chem"], d["w_edge"], d["w_edge_op"])
        if any(isinstance(w, bool) or not isinstance(w, (int, float))
               or not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("Checkpoint chemical loss weights must be finite and nonnegative")
        scale = max(weights)
        divisor = scale or 1.0
        # Normalize stored coefficients while preserving their effective weights.
        d["w_valence"], d["w_chem"] = d["w_chem"] / divisor, scale
        for name in ("w_atom", "w_charge", "w_hydrogen", "w_adjacency", "w_bond",
                     "w_count", "w_bond_length", "w_feature", "w_embedding"):
            d[name] *= auxiliary / divisor
        for name in ("w_edge", "w_edge_op"):
            d[name] /= divisor
    if set(d) != required:
        raise ValueError(f"Checkpoint configuration keys differ: missing={sorted(required - set(d))}, unknown={sorted(set(d) - required)}")
    return ABCMConfig(**d)


def _validate_shared_weights(checkpoint, modules):
    """Reject contradictory copies of parameters/buffers tied by the architecture."""
    aliases = {}
    for module_name, module in modules:
        if module_name not in checkpoint:
            raise ValueError(f"Checkpoint is missing the trained {module_name} module")
        for name, value in module.state_dict(keep_vars=True).items():
            if name not in checkpoint[module_name]:
                continue  # strict load_state_dict supplies the missing-key diagnostic.
            saved = checkpoint[module_name][name]
            previous = aliases.get(id(value))
            if previous is not None:
                prev_module, prev_name, prev_saved = previous
                if (not isinstance(saved, torch.Tensor) or not isinstance(prev_saved, torch.Tensor)
                        or saved.dtype != prev_saved.dtype or saved.shape != prev_saved.shape
                        or not torch.equal(saved, prev_saved)):
                    raise ValueError(f"Checkpoint has conflicting shared weights: {prev_module}.{prev_name} and {module_name}.{name}")
            else:
                aliases[id(value)] = (module_name, name, saved)


def load_ckpt(path, dev):
    try:
        ck = torch.load(path, map_location=dev, weights_only=False)
    except TypeError:
        ck = torch.load(path, map_location=dev)
    if not isinstance(ck, dict) or ck.get("schema_version") != CHEMISTRY_SCHEMA_VERSION:
        raise ValueError("Only schema_version=2 checkpoints are supported")
    meta = ck.get("meta", {})
    if meta.get("schema_version") != CHEMISTRY_SCHEMA_VERSION:
        raise ValueError("Checkpoint metadata schema is missing or inconsistent")
    if meta.get("implementation_revision") != IMPLEMENTATION_REVISION:
        raise ValueError(f"Checkpoint implementation_revision must be {IMPLEMENTATION_REVISION!r} to match the model architecture")
    expected = {"task_feature_schema": TASK_VECTOR_REPRESENTATION,
                "pocket_feature_dim": PROTEIN_FEATURE_DIM, "pocket_feature_schema": PROTEIN_FEATURE_SCHEMA,
                "decoder_kind": DECODER_KIND, "seed_objective": SEED_OBJECTIVE,
                "state_distance_schema": STATE_DISTANCE_SCHEMA, "anchor_kind": ANCHOR_KIND,
                "group_input_schema": GROUP_INPUT_SCHEMA}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Checkpoint {key} must be {value!r}; current architecture metadata is required")
    cfg = _cfg_from_ckpt(ck.get("abcm_config"))
    required = ("node_feat_dim", "cond_dim", "z_dim", "num_nodes", "max_nodes", "edge_T", "hidden", "n_layers", "pocket_hidden", "pocket_layers", "pocket_r", "pocket_max_atoms")
    missing = [name for name in required if name not in meta]
    if missing:
        raise ValueError(f"Checkpoint is missing architecture metadata: {missing}")
    if int(meta["max_nodes"]) != cfg.max_nodes or int(meta["edge_T"]) != cfg.edge_T or int(meta["z_dim"]) != cfg.z_dim:
        raise ValueError("Checkpoint architecture metadata disagrees with its configuration")
    core_kw = dict(meta.get("core_kwargs") or {
        "hidden": int(meta["hidden"]), "num_layers": int(meta["n_layers"]),
        "mpnn_sub_layers": int(meta.get("mpnn_sub_layers", SUB_LAYERS)),
    })
    model, seed, prior = build_model(
        node_feat_dim=int(meta["node_feat_dim"]), cond_dim=int(meta["cond_dim"]),
        z_dim=cfg.z_dim, num_nodes=int(meta["num_nodes"]), max_nodes=cfg.max_nodes,
        shared_core=True, core_kwargs=core_kw, edge_T=cfg.edge_T,
        prior_hidden=int(meta.get("prior_hidden", PRIOR_HIDDEN)),
        joint_steps=cfg.joint_steps, pocket_neighbors=cfg.pocket_neighbors,
        pocket_step_scale=cfg.pocket_step_scale,
        num_anchors=cfg.num_anchors,
    )
    pe = PocketEncoder(
        out_dim=int(meta["cond_dim"]), radius=float(meta["pocket_r"]),
        hidden_nf=int(meta["pocket_hidden"]), n_layers=int(meta["pocket_layers"]),
        feat_dim=PROTEIN_FEATURE_DIM,
    )
    te = TaskEncoder(in_dim=int(meta["cond_dim"]), out_dim=int(meta["cond_dim"]))
    modules = (("model", model), ("seed", seed), ("prior", prior), ("pocket_encoder", pe), ("task_encoder", te))
    _validate_shared_weights(ck, modules)
    for key, module in modules:
        module.load_state_dict(ck[key], strict=True)
        module.to(dev).eval()
    model.distance_config = cfg.molecular_distance
    return model, seed, prior, pe, te, cfg, meta, int(meta["pocket_max_atoms"])


def _validated_mol(state):
    mol = molstate_to_mol(state)
    if mol is None:
        return None, "invalid_chemistry_or_coordinates"
    if len(Chem.GetMolFrags(mol)) != 1:
        return None, "disconnected_molecule"
    return mol, None


def _write_sdf(path, X):
    mol, _ = _validated_mol(X)
    if mol is None:
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    try:
        writer.write(mol)
    finally:
        writer.close()
    return True


def _check_pocket_xyz(xyz, pocket_pdb):
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] < 4 or not np.isfinite(xyz).all():
        raise ValueError(f"Pocket has no usable set of at least four finite atomic coordinates: {pocket_pdb}")
    return xyz


def _pocket_xyz(pocket_pdb, *, max_atoms, seed):
    return _check_pocket_xyz(pocket_coords(pocket_pdb, max_atoms=max_atoms, seed=seed), pocket_pdb)


def _typed_input(state, *, node_feat_dim, max_nodes=None):
    return encode_validated_state(state, node_feat_dim=node_feat_dim, max_nodes=max_nodes)


def mk_cond(pocket_pdb, *, n_feat, c_dim, pe, te, dev, pkt_max, seed, ref_sdf=None, max_nodes=None):
    xyz, features = pocket_atoms(pocket_pdb, max_atoms=pkt_max, seed=seed)
    _check_pocket_xyz(xyz, pocket_pdb)
    tvec = np.zeros(c_dim, dtype=np.float32)
    if ref_sdf is not None:
        ref = mol_to_molstate(load_reference_mol_from_sdf(ref_sdf, require_3d=True), node_feat_dim=n_feat)
        ref = _typed_input(ref, node_feat_dim=n_feat, max_nodes=max_nodes)
        tvec = task_vec(ref, c_dim)
    with torch.no_grad():
        context = pe(torch.from_numpy(xyz).to(device=dev, dtype=torch.float32),
                     feat=torch.from_numpy(features).to(device=dev, dtype=torch.float32), return_context=True)
        return context, te(torch.from_numpy(tvec).to(device=dev, dtype=torch.float32))


def mk_pocket(pdb: Path, ref_sdf: Path | None) -> dict:
    pkt = {"receptor_pdb": pdb.resolve()}
    if ref_sdf is not None:
        pkt["reference_mol"] = load_reference_mol_from_sdf(ref_sdf, require_3d=True)
    return pkt


def pkt_dict(X: MolState, pocket: dict) -> dict:
    mol, _ = _validated_mol(X)
    if mol is None:
        return {name: None for name in ("spharm", "sshape", "sfit", "sclash", "svina", "pkt_fit")}
    score = pkt_fit(X, pocket)
    return _json_safe({"spharm": score.spharm, "sshape": score.sshape, "sfit": score.sfit, "sclash": score.sclash, "svina": score.svina, "pkt_fit": score.tot})


def pick_anchors(pdb, ref_sdf, *, n_anchors, max_sites, n_feat, dev, seed):
    pkt = mk_pocket(pdb, ref_sdf)
    xyz = _pocket_xyz(pdb, max_atoms=max_sites, seed=seed)
    if n_anchors <= 0:
        return torch.from_numpy(xyz.mean(axis=0, keepdims=True)).to(device=dev, dtype=torch.float32)
    selected = top_anchors(xyz, pkt, k=n_anchors, score_kw={"wt": PktWt(), "clash_r": 3.4, "probe_z": 7, "feat_dim": n_feat})
    if selected.numel() == 0:
        selected = torch.from_numpy(xyz.mean(axis=0, keepdims=True))
    return selected.to(device=dev, dtype=torch.float32)


def _smi(X):
    mol, _ = _validated_mol(X)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _vs_ref(X, X_ref, pkt=None, *, w_pkt=0.5, distance_config=None):
    if _smi(X) is None:
        return -float("inf"), None, None, None
    similarity = tanimoto_molstate(X, X_ref)
    distance = float(X.distance_to(X_ref, weights=distance_config))
    ref_score = float(similarity) if similarity is not None else (1.0 / (1.0 + distance) if np.isfinite(distance) else -float("inf"))
    pocket_score = pkt_dict(X, pkt)["pkt_fit"] if pkt is not None else None
    if pocket_score is not None:
        if not np.isfinite(ref_score):
            return pocket_score, similarity, distance, pocket_score
        weight = float(w_pkt)
        return (1.0 - weight) * ref_score + weight * pocket_score, similarity, distance, pocket_score
    return ref_score, similarity, distance, pocket_score


def basin_from_routes(routes, X_star, bmax, pkt=None, w_pkt=0.5, distance_config=None):
    rows, seen = [], set()
    for route in routes:
        for step, state in enumerate(route.trajectory[:-1]):
            smiles = _smi(state)
            if smiles is None or smiles in seen:
                continue
            seen.add(smiles)
            key, sim, distance, _ = _vs_ref(state, X_star, pkt, w_pkt=w_pkt, distance_config=distance_config)
            if np.isfinite(key):
                extra = {"route_init_index": route.init_index, "trajectory_step": step, "role": "initial_or_intermediate"}
                rows.append((key, state, sim, distance, extra))
    rows.sort(key=lambda row: row[0], reverse=True)
    rows = rows[:_clamp_basin(bmax)]
    return ([r[1] for r in rows], [r[0] for r in rows], [r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows])


def basin_from_traj(traj, bmax, ref=None, distance_config=None):
    if len(traj) < 2:
        return [], [], [], [], []
    route = Route(traj[0], traj[-1], traj)
    basin, scores, similarity, _, extra = basin_from_routes([route], ref if ref is not None else traj[-1], bmax, distance_config=distance_config)
    # Output slots have no indexwise correspondence to initialization atoms.
    return basin, scores, similarity, [None] * len(basin), extra


class _InferenceCountOverride:
    """Apply an explicit output atom count across inference routes."""
    def __init__(self, model, num_atoms):
        self.model, self.num_atoms = model, int(num_atoms)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def step(self, X, hP, hc, **kwargs):
        return self.model.step(X, hP, hc, num_nodes=self.num_atoms)

    def forward_step_trainable(self, X, hP, hc, device=None, *, num_nodes=None, **kwargs):
        return self.model.forward_step_trainable(
            X, hP, hc, device=device, num_nodes=self.num_atoms, **kwargs)

    def forward_group_step_trainable(self, group, hP, hc, device=None, *, num_nodes=None, **kwargs):
        return self.model.forward_group_step_trainable(
            group, hP, hc, device=device, num_nodes=self.num_atoms, **kwargs)

    def group_step_trainable_infer(self, group, hP, hc, device=None, *, X_star=None,
                                   edge_threshold=None, num_nodes=None, group_context=True, slot_noise=None):
        with torch.no_grad():
            out = self.forward_group_step_trainable(
                group, hP, hc, device=device, X_star=X_star, num_nodes=num_nodes,
                predict_count=True, group_context=group_context, slot_noise=slot_noise)
            return state_from_step(out["primary"], out, edge_threshold=self.model.edge_threshold if edge_threshold is None else edge_threshold)

    def group_step(self, group, hP, hc, *, X_star=None, X_current=None, num_nodes=None,
                   group_context=True, slot_noise=None):
        return self.group_step_trainable_infer(
            group, hP, hc, X_star=X_star, num_nodes=num_nodes,
            group_context=group_context, slot_noise=slot_noise)


class _InferenceWithoutGroupContext:
    """Disable group-summary conditioning while retaining every source member.

    Uses fixed trained weights; this is not a retrained pointwise baseline.
    """
    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)

    def forward_group_step_trainable(self, group, hP, hc, device=None, *, X_star=None,
                                     num_nodes=None, predict_count=True, group_context=True, **kwargs):
        return self.model.forward_group_step_trainable(
            group, hP, hc, device=device, X_star=X_star, num_nodes=num_nodes,
            predict_count=predict_count, group_context=False, **kwargs)

    def group_step_trainable_infer(self, group, hP, hc, device=None, *, X_star=None,
                                   edge_threshold=None, num_nodes=None, group_context=True, slot_noise=None):
        with torch.no_grad():
            out = self.forward_group_step_trainable(
                group, hP, hc, device=device, X_star=X_star, num_nodes=num_nodes,
                predict_count=True, slot_noise=slot_noise)
            threshold = self.model.edge_threshold if edge_threshold is None else edge_threshold
            return state_from_step(out["primary"], out, edge_threshold=threshold)

    def group_step(self, group, hP, hc, *, X_star=None, X_current=None, num_nodes=None,
                   group_context=True, slot_noise=None):
        return self.group_step_trainable_infer(
            group, hP, hc, X_star=X_star, num_nodes=num_nodes, slot_noise=slot_noise)


def explore_pool(model, pool, X_star, hP, hc, cfg, n_routes, bmax, seed, pkt=None, w_pkt=0.5, *, use_grp=None, failures=None, completed_routes=None):
    grouped = cfg.group_train if use_grp is None else bool(use_grp)
    args = dict(model=model, pool=pool, hP=hP, hc=hc, cfg=cfg, num_starts=n_routes, X_star=X_star, seed=seed, early_stop=True)
    if failures is not None:
        args["on_error"] = failures.append
    if completed_routes is not None:
        args["on_route"] = completed_routes.append
    routes = infer_grp_routes(**args) if grouped else infer_routes(**args)
    def rank(route):
        if _smi(route.final) is None:
            return -float("inf")
        return _vs_ref(route.final, X_star, pkt, w_pkt=1.0, distance_config=cfg.molecular_distance)[0]
    routes.sort(key=rank, reverse=True)
    return (routes, *basin_from_routes(routes, X_star, bmax, pkt=pkt, w_pkt=w_pkt, distance_config=cfg.molecular_distance))


def _inference_pool_config(seed, pocket_min, mmp_db):
    config = PoolCfg(seed=seed, pocket_min=pocket_min)
    if mmp_db is not None:
        backend = LocalMMPConfig(fragment_db_path=Path(mmp_db).expanduser().resolve(), require_backend=True)
        validate_mmp_backend(backend)
        config = dataclasses.replace(config, local_mmp=backend)
    return config


def mk_infer_pool(pdb, ref_sdf, *, seed=0, mmp_db=None, pocket_min=None, node_feat_dim=16, max_nodes=128, return_report=False,
                  exclude_kinds=()):
    excluded = _state_exclusions(exclude_kinds)
    cx = CrossDockComplex(complex_id="infer", receptor_pdb=pdb.resolve(), reference_ligand_sdf=ref_sdf.resolve())
    config = _inference_pool_config(seed, pocket_min, mmp_db)
    bundle = build_crossdock_bundle(cx, config)
    pool = bundle.all_molstates()
    source_exclusion = None
    if excluded:
        items = bundle.all_states()
        kept = [item for item in items if item.pool_kind not in excluded]
        pool = [item.X for item in kept]
        source_exclusion = {"excluded_types": sorted(excluded),
                            "policy": "filter_actual_sources_after_all_building_and_backfill; no_refill; not_size_matched",
                            "before_counts": dict(Counter(item.pool_kind for item in items)),
                            "after_exclusion_counts": dict(Counter(item.pool_kind for item in kept)),
                            "removed_states": len(items) - len(kept)}
    if not pool:
        raise ValueError("The reference-derived state pool is empty; no synthetic replacement pool was generated")
    target = _typed_input(bundle.X_star, node_feat_dim=node_feat_dim, max_nodes=max_nodes)
    clean, rejected, accepted_counts = [], Counter(), Counter()
    for index, state in enumerate(pool):
        try:
            clean.append(_typed_input(state, node_feat_dim=node_feat_dim, max_nodes=max_nodes))
            if source_exclusion is not None:
                accepted_counts[kept[index].pool_kind] += 1
        except ValueError as error:
            rejected[str(error)] += 1
    report = {"considered_states": len(pool), "accepted_states": len(clean),
              "mmp_backend_policy": "required_explicit" if mmp_db is not None else "optional_default",
              "rejected_states": sum(rejected.values()), "rejection_reasons": dict(rejected),
              "build_rejections": dict(getattr(bundle, "build_rejections", {})),
              "pool_sources_before_input_validation": {
                  kind: len(states) for kind, states in getattr(bundle, "by_kind", {}).items()}}
    if source_exclusion is not None:
        source_exclusion["accepted_counts"] = dict(accepted_counts)
        report["source_exclusion"] = source_exclusion
    if not clean:
        raise ValueError(f"All {len(pool)} state-pool candidates were rejected: {dict(rejected)}")
    return (clean, target, report) if return_report else (clean, target)


def mk_scaffold_prior(pdb, ref_sdf, *, seed=0, mmp_db=None, pocket_min=None, node_feat_dim=16, max_nodes=128):
    spec = CrossDockComplex("initialization", pdb.resolve(), ref_sdf.resolve())
    pool_cfg = _inference_pool_config(seed, pocket_min, mmp_db)
    bundle = build_crossdock_bundle(spec, pool_cfg)
    candidates, rejected = [], 0
    for candidate in bundle.by_kind.get("scaffold", []):
        if candidate.pool_kind != "scaffold":
            rejected += 1
            continue
        try:
            candidates.append(_typed_input(candidate.X, node_feat_dim=node_feat_dim, max_nodes=max_nodes))
        except ValueError:
            rejected += 1
    selected = minimal_scaffold_prior(candidates)
    return selected, {"scaffold_candidates": len(candidates), "rejected_states": rejected,
                      "mmp_backend_policy": "required_explicit" if mmp_db is not None else "optional_default",
                      "selected_states": len(selected), "minimum_heavy_atoms":
                      _validated_mol(selected[0])[0].GetNumHeavyAtoms(),
                      "selection_rule": "minimum_heavy_atom_count_within_scaffold_retention_states"}


def gen_seed(model, seed, prior, hP, hc, cfg, bmax, run_seed, anchors=None, num_atoms=None,
             *, structural_prior=None, alpha_prior=0.0, initialization_record=None):
    """Sample conditional or scaffold initial states and map the group to a molecule."""
    _set_seed(run_seed)
    rng = random.Random(run_seed)
    with torch.no_grad():
        initial_group, record = sample_initial_group(seed, prior, hP, hc, cfg, rng,
            anchors=anchors, num_atoms=num_atoms, structural_prior=structural_prior, alpha_prior=alpha_prior)
        if initialization_record is not None:
            initialization_record.update(record)
        kwargs = {"X_star": None}
        # Override output size; keep scaffold inputs intact.
        if num_atoms is not None:
            out = model.forward_group_step_trainable(initial_group, hP, hc,
                    num_nodes=num_atoms, predict_count=True, **kwargs)
            final = state_from_step(out["primary"], out)
        else:
            final = model.group_step_trainable_infer(initial_group, hP, hc, **kwargs)
    trajectory = [initial_group[0], final]
    basin, scores, similarity, rmsd, _ = basin_from_traj(trajectory, bmax=bmax, distance_config=cfg.molecular_distance)
    return final, basin, scores, similarity, rmsd, trajectory


def _dump_basin(out_dir, mols, prefix="rank"):
    files = []
    for index, state in enumerate(mols):
        path = Path(out_dir) / f"{prefix}_{index:02d}.sdf"
        if _write_sdf(path, state):
            files.append(path.name)
    return files


def _count_summary(requested, attempts, valid):
    return {"requested": int(requested), "attempted": int(attempts), "valid_final": int(valid),
            "invalid_final": int(attempts - valid), "not_attempted": max(0, int(requested - attempts)),
            "validity_rate": valid / attempts if attempts else None}


def _final_geometry(state, receptor_geometry=None):
    mol, _ = _validated_mol(state)
    receptor = {} if receptor_geometry is None else {
        "receptor_positions": receptor_geometry[0], "receptor_atomic_numbers": receptor_geometry[1]}
    return evaluate_geometry(mol, **receptor)


def _score_output(row, state, pocket):
    try:
        scores = pkt_dict(state, pocket)
    except (Exception, KeyboardInterrupt) as error:
        row["scoring_error"] = {"error_type": type(error).__name__, "error": str(error)}
        raise
    row.update(scores)
    return scores


def _dump_routes(out_root, routes, basin, sc, tani, dist, extra, X_star, pkt=None, requested=None, distance_config=None, failures=(), receptor_geometry=None, report=None):
    out_root = Path(out_root)
    rows, valid_routes = [], []
    for index, route in enumerate(routes):
        tag = f"route_{index:03d}"
        init_path = out_root / "routes" / f"{tag}_init.sdf"
        final_path = out_root / "routes" / f"{tag}_final.sdf"
        init_ok = _write_sdf(init_path, route.X_init)
        final_ok = _write_sdf(final_path, route.final)
        _, failure = _validated_mol(route.final)
        row = {"id": tag, "pool_init_index": route.init_index,
               "init_sdf": init_path.relative_to(out_root).as_posix() if init_ok else None,
               "final_sdf": final_path.relative_to(out_root).as_posix() if final_ok else None,
               "status": "valid" if final_ok else "rejected", "rejection_reason": failure,
               "geometry": _final_geometry(route.final, receptor_geometry),
               "trajectory_len": len(route.trajectory), "group_size": route.group_size,
               "d_to_reference": float(route.final.distance_to(X_star, weights=distance_config)) if final_ok else None}
        if final_ok:
            valid_routes.append(route)
            row["smiles"] = _smi(route.final)
        if route.group_states:
            group_dir = out_root / "routes" / f"{tag}_group"
            names = _dump_basin(group_dir, route.group_states, prefix="member")
            row["group_sdfs"] = [(group_dir / name).relative_to(out_root).as_posix() for name in names]
        rows.append(row)
    for index, failure in enumerate(failures):
        rows.append({"id": f"failed_route_{index:03d}", **failure,
                     "status": "generation_error", "final_sdf": None,
                     "geometry": not_evaluated_geometry("generation_error")})
    best_path = out_root / "best_final.sdf"
    best = valid_routes[0] if valid_routes else None
    best_ok = best is not None and _write_sdf(best_path, best.final)
    chosen = [i for i, state in enumerate(basin) if _smi(state) is not None]
    basin_files = _dump_basin(out_root / "basin", [basin[i] for i in chosen])
    result = {} if report is None else report
    result.update({"best_final_sdf": best_path.name if best_ok else None,
              "final_counts": _count_summary(len(rows) if requested is None else requested, len(rows), len(valid_routes)),
              "geometry_counts": summarize_geometry([row["geometry"] for row in rows]),
              "num_routes": len(routes), "routes": rows,
              "global_basin_role": "initial_or_intermediate_not_generated_final",
              "global_basin_dir": "basin", "global_basin_count": len(basin_files),
              "global_basin_sdfs": basin_files,
              "global_basin_scores": [sc[i] for i in chosen],
              "global_basin_tanimoto_to_ref": [tani[i] for i in chosen],
              "global_basin_d_to_reference": [dist[i] for i in chosen],
              "global_basin_meta": [extra[i] for i in chosen]})
    if pkt is not None:
        for route, row in zip(routes, rows):
            if row["status"] == "valid":
                scores = _score_output(row, route.final, pkt)
                if route is best:
                    result["best_pkt_fit"] = scores
    return _json_safe(result)


def _chemistry_histograms(states):
    elements, bonds, sizes = Counter(), Counter(), Counter()
    valid = 0
    for state in states:
        mol, _ = _validated_mol(state)
        if mol is None:
            continue
        valid += 1
        sizes[str(mol.GetNumAtoms())] += 1
        elements.update(atom.GetSymbol() for atom in mol.GetAtoms())
        # Report aromaticity after sanitization.
        bonds.update(str(bond.GetBondType()).lower() for bond in mol.GetBonds())
    return {"valid_final_count": valid, "element_atom_counts": dict(sorted(elements.items())),
            "bond_type_counts": dict(sorted(bonds.items())), "molecule_atom_count_histogram": dict(sorted(sizes.items(), key=lambda item: int(item[0])))}


def build_parser():
    parser = argparse.ArgumentParser(description="Generate with a group operator and conditional/scaffold-mixture initialization.")
    parser.add_argument("--pocket-pdb", type=Path, required=True)
    parser.add_argument("--num-generate", type=int, required=True, help="Total requested final-generation attempts, without replacement retries.")
    parser.add_argument("--num-atoms", type=int, default=None, help="Override predicted output atom count in both the seed and operator branches.")
    parser.add_argument("--init-mode", choices=("seed", "pool", "mixture", "both"), default="mixture",
                        help="mixture implements the scaffold/conditional initialization mixture; both is its alias")
    parser.add_argument("--alpha-prior", type=float, default=None, help="Structural mixture probability; defaults to checkpoint alpha_prior")
    parser.add_argument("--output-dir", type=Path, default=_ROOT / "output")
    parser.add_argument("--basin-max", type=int, default=20)
    parser.add_argument("--checkpoint", type=Path, default=_ROOT / "output" / "abcm.pt")
    parser.add_argument("--reference-sdf", type=Path,
                        help="3D ligand pose in the same coordinate frame as the pocket PDB")
    parser.add_argument("--mmp-db", type=Path,
                        help="Require this CReM database for pool/scaffold construction; unused in seed-only initialization")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-anchors", type=int, default=None,
                        help="Seed/mixture only: anchor count for learned/heuristic modes; defaults to the checkpoint")
    parser.add_argument("--anchor-mode", choices=("learned", "center", "heuristic"), default="learned",
                        help="Seed/mixture only: learned pocket anchors, pocket center, or heuristic probes")
    parser.add_argument("--anchor-sites", type=int, default=128,
                        help="Maximum candidate pocket atoms for heuristic anchors (seed/mixture only)")
    parser.add_argument("--pocket-min", type=float,
                        help="Pool threshold on the current score scale; required to replace a threshold from an unverified checkpoint score schema")
    parser.add_argument("--basin-pocket-weight", type=float, default=0.5)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--group-inference", action="store_true")
    group.add_argument("--no-group-inference", action="store_true")
    return parser


def _state_exclusions(kinds):
    excluded = set(kinds)
    unknown = excluded - set(STATE_POOL_TYPES)
    if unknown:
        raise ValueError(f"Unknown state types: {sorted(unknown)}")
    return excluded


def _validate_inference_options(args, *, group_size=None, group_context=True, exclude_state_types=()):
    """Cross-option rules shared by the CLI and the Python generation API."""
    if args.alpha_prior is not None and args.init_mode != "mixture":
        raise ValueError("--alpha-prior is only used by mixture mode")
    if args.no_group_inference and args.init_mode != "pool":
        raise ValueError("--no-group-inference is an explicit pool-mode ablation; seed/mixture use initial groups")
    if group_size is not None and (isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1):
        raise ValueError("group_size must be a positive integer")
    excluded = _state_exclusions(exclude_state_types)
    if excluded and args.init_mode != "pool":
        raise ValueError("--exclude-state-types requires --init-mode pool")
    if len(excluded) == len(STATE_POOL_TYPES):
        raise ValueError("Cannot exclude all five state types")
    if args.no_group_inference and (group_size is not None or not group_context):
        raise ValueError("Group-size or group-context interventions require grouped sampling; omit --no-group-inference")


def validate_args(args, parser):
    if args.init_mode == "both":
        args.init_mode = "mixture"
    try:
        _validate_inference_options(
            args, group_size=getattr(args, "group_size", None),
            group_context=getattr(args, "group_context", "on") == "on",
            exclude_state_types=getattr(args, "exclude_state_types", ()),
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def parse_args(argv=None):
    parser = build_parser()
    return validate_args(parser.parse_args(argv), parser)


def run_generation(args, *, group_size=None, group_context=True, exclude_state_types=(), evaluation_entry=False):
    """Shared generation path; optional interventions never alter saved weights."""
    _validate_inference_options(args, group_size=group_size, group_context=group_context,
                                exclude_state_types=exclude_state_types)
    _set_seed(args.seed)
    pdb = args.pocket_pdb.resolve()
    reference = args.reference_sdf.resolve() if args.reference_sdf else None
    for name, path in (("checkpoint", args.checkpoint), ("pocket PDB", pdb), ("reference SDF", reference)):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    if args.init_mode in ("pool", "mixture") and reference is None:
        raise ValueError("--reference-sdf is required for pool or mixture; use seed mode without a structural reference")
    if reference is not None:
        load_reference_mol_from_sdf(reference, require_3d=True)
    output_dir = args.output_dir.resolve()
    reserved = ("generate_manifest.json", "generate_manifest.json.tmp", "best_final.sdf", "routes", "basin")
    if any((output_dir / name).exists() for name in reserved) or any(output_dir.glob("seed_*")):
        raise FileExistsError("Output already contains generation artifacts; select a fresh --output-dir")
    dev = torch.device(args.device)
    model, seed, prior, pe, te, cfg, meta, pkt_max = load_ckpt(args.checkpoint, dev)
    checkpoint_configuration = dataclasses.asdict(cfg)
    alpha = (cfg.alpha_prior if args.alpha_prior is None else args.alpha_prior) if args.init_mode == "mixture" else 0.
    uses_pool = args.init_mode == "pool" or (args.init_mode == "mixture" and alpha > 0)
    uses_seed = args.init_mode != "pool" and alpha < 1.
    effective_pocket_min = cfg.pocket_min if args.pocket_min is None else args.pocket_min
    checkpoint_score_schema = meta.get("pocket_score_schema") or "unknown"
    if (uses_pool and effective_pocket_min is not None and args.pocket_min is None
            and checkpoint_score_schema != POCKET_SCORE_SCHEMA):
        raise ValueError(f"Checkpoint pocket_score_schema is {checkpoint_score_schema!r}; "
                         f"set --pocket-min explicitly for score schema {POCKET_SCORE_SCHEMA!r}. "
                         "The saved threshold cannot be inherited across unverified score scales.")
    if group_size is not None:
        cfg = dataclasses.replace(cfg, grp_min=group_size, grp_max=group_size)
    if args.num_atoms is not None:
        if args.num_atoms > cfg.max_nodes:
            raise ValueError(f"--num-atoms exceeds checkpoint max_nodes={cfg.max_nodes}")
        model = _InferenceCountOverride(model, args.num_atoms)
    if not group_context:
        model = _InferenceWithoutGroupContext(model)
    hp, hc = mk_cond(pdb, n_feat=int(meta["node_feat_dim"]), c_dim=int(meta["cond_dim"]), pe=pe, te=te, dev=dev, pkt_max=pkt_max, seed=args.seed, ref_sdf=reference, max_nodes=cfg.max_nodes)
    pocket = mk_pocket(pdb, reference)
    receptor_geometry = load_receptor_geometry(pdb)
    anchor_count = (cfg.num_anchors if args.num_anchors is None else args.num_anchors) if uses_seed else None
    output_dir.mkdir(parents=True, exist_ok=True)
    use_group = cfg.group_train if not args.no_group_inference else False
    if args.group_inference:
        use_group = True
    manifest = {
        "schema_version": CHEMISTRY_SCHEMA_VERSION, "implementation": "ABCD",
        "implementation_revision": IMPLEMENTATION_REVISION, "run_status": "running",
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": _file_sha256(args.checkpoint),
        "checkpoint_metadata": meta, "configuration": dataclasses.asdict(cfg), "arguments": vars(args),
        "checkpoint_initialization_training": meta.get("initialization_training") or "not_recorded",
        "random_seed": args.seed, "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "pocket_pdb": str(pdb), "reference_sdf": str(reference) if reference else None,
        "reference_conditioning": {"task_features": reference is not None, "pool_construction": uses_pool,
                                  "group_selection_and_ranking": reference is not None and args.init_mode == "pool"},
        "pocket_scoring": {"schema": POCKET_SCORE_SCHEMA, "weights": dataclasses.asdict(PktWt()),
                           "receptor_scope": POCKET_RECEPTOR_SCOPE,
                           "radius_source": "generated_molecule_elements",
                           "checkpoint_score_schema": checkpoint_score_schema,
                           "checkpoint_radius_source": meta.get("pocket_radius_source") or "unknown",
                           "checkpoint_pharm_schema": meta.get("pocket_pharm_schema") or "unknown"},
        "mode_semantics": (
            "reference_conditioned_pool_then_group_mapping" if use_group else "reference_conditioned_pool_then_single_mapping"
            ) if args.init_mode == "pool" else "per_initial_state_bernoulli_scaffold_conditional_mixture_then_group_mapping",
        "initial_group_semantics": (
            "distinct_pool_members_under_target_distance_and_pairwise_edit_constraints" if use_group else "single_sampled_pool_member"
            ) if args.init_mode == "pool" else "iid_initial_samples_may_repeat_same_scaffold_no_resampling_to_force_a_quota",
        "initial_group_selection": "reference_neighborhood_constraints_for_pool_only_unknown_target_not_used_for_seed_or_mixture",
        "validity_policy": "finite_coordinates_RDKit_sanitization_and_single_connected_component",
        "geometry_policy": {"schema": "rdkit_bounds_geometry_v1", "mode": "report_only_raw_coordinates",
                            "thresholds": dataclasses.asdict(GeometryValidationConfig()),
                            "scope": "intramolecular_geometry_and_heavy_atom_receptor_contacts",
                            "receptor_contact_scope": "vdw_overlap_diagnostic_without_metal_coordination_model",
                            "rate_denominator": "all_generation_attempts"},
        "stereochemistry": "not_predicted", "retries_to_replace_invalid_samples": 0,
        "atom_count_mode": "explicit_override" if args.num_atoms is not None else "learned",
        "num_atoms_override": args.num_atoms, "group_inference": use_group,
        "basin_role": "initial_or_intermediate_states_excluded_from_final_counts",
    }
    if evaluation_entry or group_size is not None or not group_context or exclude_state_types:
        manifest["inference_experiment"] = {
            "scope": "fixed_checkpoint_inference_conditions; not_a_retrained_pointwise_baseline",
            "checkpoint_configuration": checkpoint_configuration,
            "group_size_override": group_size,
            "group_context": "on" if group_context else "off_group_summary",
            "group_context_intervention": "none" if group_context else "disable_pooled_group_conditioning_keep_all_source_members",
            "sampled_group_range": [cfg.grp_min, cfg.grp_max] if args.init_mode != "pool" or use_group else None,
            "mapping_uses_group_context": bool((args.init_mode != "pool" or use_group) and group_context),
            "mapping_uses_all_group_members": bool(args.init_mode != "pool" or use_group),
            "group_size_outside_training_range": group_size is not None and not (
                checkpoint_configuration["grp_min"] <= group_size <= checkpoint_configuration["grp_max"]),
            "excluded_state_types": sorted(set(exclude_state_types)),
            "training_w_basin": checkpoint_configuration["w_basin"],
            "randomness": "original_generation_stream; same_seed_repeats_same_condition; different_group_sizes_or_initialization_settings_are_not_strictly_noise_paired",
        }
    requested, attempts, valid = args.num_generate, 0, 0
    pool_requested = args.num_generate if args.init_mode == "pool" else 0
    seed_requested = args.num_generate - pool_requested
    manifest["requested_by_mode"] = {"pool": pool_requested,
                                     "seed": seed_requested if args.init_mode == "seed" else 0,
                                     "mixture": seed_requested if args.init_mode == "mixture" else 0}
    manifest["effective_pool_pocket_min"] = effective_pocket_min
    manifest["effective_pool_pocket_min_source"] = (
        "unused" if not uses_pool else "cli" if args.pocket_min is not None else "checkpoint")
    manifest["pool_pocket_filter_enabled"] = uses_pool and effective_pocket_min is not None
    manifest["anchor_usage"] = (
        "conditional_seed_initialization" if uses_seed else
        "not_applicable_pool" if args.init_mode == "pool" else "not_applicable_scaffold_only")
    manifest["anchor_mode"] = args.anchor_mode if uses_seed else None
    manifest["anchor_count_requested"] = anchor_count
    manifest["effective_alpha_prior"] = alpha
    finals = []
    bmax = _clamp_basin(args.basin_max)
    def snapshot():
        manifest["final_counts"] = _count_summary(requested, attempts, valid)
        geometry_rows = [*manifest.get("seed_runs", []),
                         *manifest.get("pool_exploration", {}).get("routes", [])]
        manifest["geometry_counts"] = summarize_geometry(
            [row["geometry"] for row in geometry_rows], attempted=attempts)
        manifest["successful_final_chemistry"] = _chemistry_histograms(finals)
        _write_manifest(output_dir / "generate_manifest.json", manifest)

    snapshot()
    try:
        structural_prior = None
        if args.init_mode == "mixture" and alpha > 0:
            structural_prior, prior_report = mk_scaffold_prior(
                pdb, reference, seed=args.seed, mmp_db=args.mmp_db, pocket_min=effective_pocket_min,
                node_feat_dim=int(meta["node_feat_dim"]), max_nodes=cfg.max_nodes)
            manifest["structural_prior"] = prior_report
            snapshot()
        if pool_requested:
            pool_options = {"exclude_kinds": exclude_state_types} if exclude_state_types else {}
            pool, target, pool_report = mk_infer_pool(pdb, reference, seed=args.seed, mmp_db=args.mmp_db, pocket_min=effective_pocket_min, node_feat_dim=int(meta["node_feat_dim"]), max_nodes=cfg.max_nodes, return_report=True, **pool_options)
            manifest["pool_size"] = len(pool)
            manifest["pool_validation"] = pool_report
            partial_routes, failures = [], []
            routes, basin, scores, similarity, distance, extra = partial_routes, [], [], [], [], []
            try:
                routes, basin, scores, similarity, distance, extra = explore_pool(
                    model, pool, target, hp, hc, cfg, pool_requested, bmax, args.seed,
                    pkt=pocket, w_pkt=args.basin_pocket_weight, use_grp=use_group,
                    failures=failures, completed_routes=partial_routes,
                )
            finally:
                # Keep completed routes if a later attempt fails.
                details = {}
                manifest["pool_exploration"] = details
                try:
                    _dump_routes(output_dir, routes, basin, scores, similarity, distance, extra, target, pkt=pocket, requested=pool_requested, distance_config=cfg.molecular_distance, failures=failures, receptor_geometry=receptor_geometry, report=details)
                finally:
                    counts = details.get("final_counts", _count_summary(pool_requested, 0, 0))
                    attempts += counts["attempted"]
                    valid += counts["valid_final"]
                    finals.extend(route.final for route in routes)
                    snapshot()
        if seed_requested:
            anchors = None
            if uses_seed:
                seed.anchor_scorer.num_anchors = anchor_count
                anchors = None if args.anchor_mode == "learned" else pick_anchors(
                    pdb, reference, n_anchors=0 if args.anchor_mode == "center" else anchor_count,
                    max_sites=pkt_max if args.anchor_mode == "center" else args.anchor_sites,
                    n_feat=int(meta["node_feat_dim"]), dev=dev, seed=args.seed)
            rows = manifest["seed_runs"] = []
            for index in range(seed_requested):
                attempts += 1
                tag = f"seed_{index:04d}"
                run_seed = (args.seed + 1000 * index) % 2**32
                initialization = {}
                try:
                    final, basin, scores, similarity, _, trajectory = gen_seed(
                        model, seed, prior, hp, hc, cfg, bmax, run_seed, anchors=anchors, num_atoms=args.num_atoms,
                        structural_prior=structural_prior, alpha_prior=alpha, initialization_record=initialization,
                    )
                except (ValueError, RuntimeError) as error:
                    recoverable = is_recoverable_generation_error(error)
                    rows.append({"id": tag, "run_seed": run_seed, "status": "generation_error", "error_type": type(error).__name__,
                                 "error": str(error), "recoverable": recoverable, "final_sdf": None,
                                 "geometry": not_evaluated_geometry("generation_error"),
                                 "initialization": initialization})
                    snapshot()
                    if not recoverable:
                        raise
                    continue
                finals.append(final)
                final_path = output_dir / f"{tag}_final.sdf"
                ok = _write_sdf(final_path, final)
                _, failure = _validated_mol(final)
                init_path = output_dir / f"{tag}_init.sdf"
                init_ok = _write_sdf(init_path, trajectory[0])
                row = {"id": tag, "run_seed": run_seed, "status": "valid" if ok else "rejected",
                       "initialization": initialization,
                       "rejection_reason": failure, "final_sdf": final_path.name if ok else None,
                       "geometry": _final_geometry(final, receptor_geometry),
                       "init_sdf": init_path.name if init_ok else None, "trajectory_len": len(trajectory),
                       "basin_role": "initial_or_intermediate_not_generated_final", "basin_scores": scores,
                       "basin_tanimoto": similarity, "basin_sdfs": [f"{tag}_basin/{name}" for name in _dump_basin(output_dir / f"{tag}_basin", basin)]}
                row["basin_count"] = len(row["basin_sdfs"])
                rows.append(row)
                if ok:
                    valid += 1
                    row["smiles"] = _smi(final)
                    _score_output(row, final, pocket)
                snapshot()
    except (Exception, KeyboardInterrupt) as error:
        manifest["run_status"] = "failed"
        manifest["run_error"] = {"error_type": type(error).__name__, "error": str(error)}
        raise
    else:
        manifest["run_status"] = "completed"
    finally:
        snapshot()
    manifest = _json_safe(manifest)
    print(f"Completed {attempts}/{requested} requested attempts; {valid} valid final molecules. Manifest: {output_dir / 'generate_manifest.json'}")
    return manifest


def main(argv=None):
    return run_generation(parse_args(argv))


if __name__ == "__main__":
    main()
