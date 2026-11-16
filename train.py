"""Train group mapping, basin consistency, and conditional seed generation."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace

from config import ABCMConfig, HIDDEN, N_LAYERS, POCKET_HIDDEN, POCKET_LAYERS, PRIOR_HIDDEN, TrainConfig

_ROOT = Path(__file__).resolve().parent


@dataclass
class TrainData:
    X_star: object
    pool: list
    cid: str
    coords: object
    pkt_ctx: object | None = None
    pdb: str | None = None
    dropped_pool_states: int = 0
    pool_source_counts: dict | None = None
    pool_build_rejections: dict | None = None
    pocket_features: object | None = None
    pool_cache: dict | None = None
    mmp_database_states: int = 0
    structural_prior: tuple = ()
    scaffold_groups: tuple = ()
    initialization_pocket: dict | None = None
    initialization_report: dict | None = None
    _map_sampler: object | None = dataclasses.field(default=None, init=False, repr=False, compare=False)


def _parse(argv=None, *, defaults=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("test", "crossdock"), default="crossdock")
    parser.add_argument("--pocket10-root", type=Path,
                        help="Default: dataset/data/crossdocked_v1.1_rmsd1.0_pocket10, or the manifest's root")
    parser.add_argument("--split-file", type=Path,
                        help="Name-pair split; integer IDs from another model's LMDB are not interchangeable")
    parser.add_argument("--split", choices=("train", "val", "validation", "test"), default="train")
    parser.add_argument("--data-manifest", type=Path,
                        help="Manifest produced by python -m dataset.prepare_crossdock")
    parser.add_argument("--allow-unsplit", action="store_true",
                        help="Explicit directory-scan ablation for fixtures/private data; no held-out guarantee")
    parser.add_argument("--max-complexes", type=int, help="Limit complexes after split selection")
    parser.add_argument("--mmp-db", type=Path, default=_ROOT / "dataset" / "data" / "mmp.db")
    parser.add_argument("--mmp-mode", choices=("required", "optional", "off"), default="required",
                        help="required fails on unavailable CReM; optional/off are recorded pool ablations")
    parser.add_argument("--pool-cache-dir", type=Path,
                        help="Optional persistent CPU state pools; mapping groups are still resampled each epoch")
    parser.add_argument("--allow-unfiltered-pool", action="store_true",
                        help="Explicit ablation: permit no PocketFit admission threshold in CrossDocked mode")
    parser.add_argument("--output-dir", type=Path, default=_ROOT / "output")
    parser.add_argument("--checkpoint", default="abcm.pt")
    parser.add_argument("--device", default="auto", help="auto, cpu or an explicit torch device")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1.4e-3)
    parser.add_argument("--num-pairs", type=int, default=512, help="Requested mapping groups per complex per epoch")
    parser.add_argument("--pool-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-clusters", type=int, default=None)
    parser.add_argument("--max-pairs-per-cluster", type=int, default=None)
    parser.add_argument("--complex-index", type=int, default=None, help="Optional single-complex fine-tuning selection")
    parser.add_argument("--test-sizes", type=int, nargs="+", default=[6, 9, 12])
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--layers", type=int, default=N_LAYERS)
    parser.add_argument("--pocket-hidden", type=int, default=POCKET_HIDDEN)
    parser.add_argument("--pocket-layers", type=int, default=POCKET_LAYERS)
    parser.add_argument("--prior-hidden", type=int, default=PRIOR_HIDDEN)
    parser.add_argument("--pocket-r", type=float, default=6.0)
    parser.add_argument("--print-config", action="store_true", help="Print resolved configuration; no data or models are loaded")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and one objective forward per complex; no updates or checkpoint")
    for field in dataclasses.fields(TrainConfig):
        parser.add_argument("--" + field.name.replace("_", "-"), type=int, default=None,
                            help="Mapping groups averaged per optimizer update" if field.name == "batch_size" else None)
    integer_fields = {"max_nodes", "z_dim", "edge_T", "joint_steps", "pocket_neighbors", "num_anchors", "delta_edit", "grp_min", "grp_max", "group_max_tries", "seed_match_samples"}
    for field in dataclasses.fields(ABCMConfig):
        if field.name in {"schema_version", "group_train", "molecular_distance"}:
            continue
        flag = "--" + field.name.replace("_", "-")
        aliases = [flag] if flag == flag.lower() else [flag, flag.lower()]
        parser.add_argument(*aliases, dest=field.name, type=int if field.name in integer_fields else float, default=None)
    for component in ("topo", "scaffold", "pharm", "geom"):
        parser.add_argument(f"--distance-{component}-weight", type=float, default=None)
    parser.add_argument("--distance-fp-radius", type=int, default=None)
    parser.add_argument("--distance-fp-nbits", type=int, default=None)
    if defaults is not None:
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    args._pocket10_root_explicit = args.pocket10_root is not None
    if args.pocket10_root is None:
        args.pocket10_root = _ROOT / "dataset" / "data" / "crossdocked_v1.1_rmsd1.0_pocket10"
    args._split_file_explicit = args.split_file is not None
    if args.split_file is None:
        args.split_file = _ROOT / "dataset" / "data" / "split_by_name.pt"
    if args.allow_unsplit and (args.data_manifest is not None or args.split != "train"):
        parser.error("--allow-unsplit cannot be combined with --data-manifest or a held-out --split")
    return args


def _cfg(args):
    kwargs = {field.name: getattr(args, field.name) for field in dataclasses.fields(ABCMConfig)
              if hasattr(args, field.name) and getattr(args, field.name) is not None}
    distance = dict(ABCMConfig().molecular_distance)
    for component in ("topo", "scaffold", "pharm", "geom"):
        value = getattr(args, f"distance_{component}_weight")
        if value is not None:
            distance[f"w_{component}"] = value
    for name in ("fp_radius", "fp_nbits"):
        value = getattr(args, f"distance_{name}")
        if value is not None:
            distance[name] = value
    kwargs["molecular_distance"] = distance
    return ABCMConfig(**kwargs)


def _train_cfg(args):
    return TrainConfig(**{field.name: getattr(args, field.name) for field in dataclasses.fields(TrainConfig)
                          if getattr(args, field.name) is not None})


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _dev(name):
    import torch
    if name.strip().lower() in {"auto", "gpu"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _uniq_params(*modules):
    seen = set()
    params = []
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                params.append(parameter)
    return params


def _specs(args):
    if args.mode == "test":
        specs = list(dict.fromkeys(args.test_sizes))
        args._data_report = {"source": "synthetic_fixture", "scientific_dataset": False}
    else:
        from dataset.crossdock_index import load_crossdock_manifest, load_crossdock_split
        from dataset.protein_data import discover_crossdock_pocket10
        if args.split != "train" and not args.dry_run:
            raise ValueError("Optimizer updates are restricted to --split train; use --dry-run for held-out data")
        if args.data_manifest is not None:
            specs, report = load_crossdock_manifest(args.data_manifest, split=args.split,
                                                   dataset_root=args.pocket10_root if args._pocket10_root_explicit else None,
                                                   split_file=args.split_file if args._split_file_explicit else None)
            args.pocket10_root = Path(report["dataset_root"])
        elif args.allow_unsplit:
            if not args.pocket10_root.is_dir():
                raise ValueError(f"Dataset directory does not exist: {args.pocket10_root}")
            specs = discover_crossdock_pocket10(args.pocket10_root)
            report = {"source": "explicit_unsplit_directory_scan", "split": None,
                      "held_out_guarantee": False, "discovered_complexes": len(specs)}
        else:
            specs, report = load_crossdock_split(args.pocket10_root, args.split_file, split=args.split)
        if args.max_clusters is not None or args.max_pairs_per_cluster is not None:
            counts, limited = {}, []
            for spec in specs:
                cluster = (Path(spec.pocket_fn_rel).parts[0] if spec.pocket_fn_rel
                           else spec.complex_id.split("__", 1)[0])
                if cluster not in counts:
                    if args.max_clusters is not None and len(counts) >= args.max_clusters:
                        continue
                    counts[cluster] = 0
                if args.max_pairs_per_cluster is None or counts[cluster] < args.max_pairs_per_cluster:
                    limited.append(spec)
                    counts[cluster] += 1
            specs = limited
        args._data_report = report
    if args.max_complexes is not None:
        specs = specs[:args.max_complexes]
    if not specs:
        raise ValueError("No training complexes found")
    if args.complex_index is not None:
        if args.complex_index >= len(specs):
            raise ValueError("--complex-index is outside the discovered dataset")
        specs = [specs[args.complex_index]]
    args._data_report = {**args._data_report, "training_entry_selected_complexes": len(specs),
                        "entry_limits": {name: getattr(args, name) for name in
                                         ("max_complexes", "max_clusters", "max_pairs_per_cluster", "complex_index")}}
    return specs


def _local_edit_config(args):
    from dataset.pool_core import LocalMMPConfig
    return LocalMMPConfig(fragment_db_path=None if args.mmp_mode == "off" else args.mmp_db,
                          require_backend=args.mmp_mode == "required")


def _data_preflight(args, cfg):
    """Check data backends before allocating the model."""
    if args.mode == "test":
        return {"source": "synthetic_fixture"}
    from dataset.pool_builders import validate_mmp_backend
    from dataset.scaffold_tree import SCAFFOLD_TREE_BACKEND, _load_scaffold_backend
    from config import POCKET_SCORE_SCHEMA
    if cfg.pocket_min is None and not args.allow_unfiltered_pool:
        raise ValueError("CrossDocked state pools require an explicit --pocket-min threshold. "
                         "Use --allow-unfiltered-pool only for a declared ablation.")
    _, scaffold_version = _load_scaffold_backend()
    return {"mmp_mode": args.mmp_mode, "mmp": validate_mmp_backend(_local_edit_config(args)),
            "scaffold": {"backend": SCAFFOLD_TREE_BACKEND, "version": scaffold_version},
            "pocket_filter": {"enabled": cfg.pocket_min is not None, "threshold": cfg.pocket_min,
                              "score_schema": POCKET_SCORE_SCHEMA,
                              "provenance": "training_configuration"}}


def _complex_seed(args, spec):
    """Keep augmentation seeds independent of dataset ordering."""
    identity = f"synthetic_{spec}" if args.mode == "test" else spec.complex_id
    return (args.seed + int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "little")) % (2 ** 31)


def _typed_state(state, cfg, train_cfg):
    from utils.chemistry import encode_validated_state
    return encode_validated_state(state, node_feat_dim=train_cfg.node_feat_dim, max_nodes=cfg.max_nodes,
                                  require_connected=True, context="Training")


def _load_one(spec, args, cfg, train_cfg, device, *, data_seed):
    import numpy as np
    from dataset.pool_core import PoolCfg, load_reference_mol_from_sdf
    from dataset.protein_data import pocket_atoms, protein_atom_features
    from dataset.sampling import is_valid
    from dataset.state_pool import build_crossdock_bundle, build_state_pool_test
    from utils.distances import molstate_to_mol
    from utils.pkt_fit import mk_pkt_ctx
    from model.initialization import minimal_scaffold_prior, scaffold_pose_groups
    cache_info, database_states = {"status": "disabled"}, 0

    if args.mode == "test":
        target = build_state_pool_test.make_target_state(spec, train_cfg.node_feat_dim)
        pool = build_state_pool_test.build_pool(target, args.pool_size)
        pool_kinds = ["synthetic"] * len(pool)
        pool_notes = ["synthetic"] * len(pool)
        build_rejections = {}
        rng = np.random.default_rng(data_seed)
        coords = rng.normal(size=(train_cfg.test_pts, 3)).astype(np.float32)
        coords *= 5.0
        coords += target.R.mean(axis=0)
        elements = ["C", "N", "O", "S"] * ((len(coords) + 3) // 4)
        pocket_features = protein_atom_features(elements[:len(coords)], ["ALA"] * len(coords),
                                                elements[:len(coords)])
        cid, pdb = f"synthetic_{spec}_atoms", None
        pocket = {"receptor_coords": coords, "reference_mol": molstate_to_mol(target)}
    else:
        from dataset.crossdock_index import validate_crossdock_complex
        validate_crossdock_complex(spec, max_nodes=cfg.max_nodes)
        reference = load_reference_mol_from_sdf(spec.reference_ligand_sdf, require_3d=True)
        pool_cfg = PoolCfg(seed=data_seed, pocket_min=cfg.pocket_min, max_pool=args.pool_size,
                           local_mmp=_local_edit_config(args),
                           mmp_db=args.mmp_db.resolve() if args.mmp_mode != "off" and args.mmp_db else None)
        if args.pool_cache_dir is None:
            bundle = build_crossdock_bundle(spec, pool_cfg)
        else:
            from dataset.pool_cache import load_or_build_pool
            if not hasattr(args, "_pool_implementation_key"):
                args._pool_implementation_key = _source_revision()["source_sha256"]
            bundle, cache_info = load_or_build_pool(spec, pool_cfg, args.pool_cache_dir,
                                                   implementation_key=args._pool_implementation_key)
        build_rejections = dict(bundle.build_rejections)
        source_states = bundle.all_states()
        database_states = sum("fragdb_mutate" in state.source_note or "fragdb_grow" in state.source_note
                              for state in source_states)
        target, pool = bundle.X_star, [state.X for state in source_states]
        pool_kinds = [state.pool_kind for state in source_states]
        pool_notes = [state.source_note for state in source_states]
        coords, pocket_features = pocket_atoms(spec.receptor_pdb, max_atoms=train_cfg.pocket_atoms, seed=data_seed)
        if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) < 4 or not np.isfinite(coords).all():
            raise ValueError(f"{spec.complex_id}: pocket must contain at least four finite 3D atom positions")
        cid, pdb = spec.complex_id, str(spec.receptor_pdb.resolve())
        # Scoring uses the full receptor even when the encoder subsamples atoms.
        pocket = {"receptor_pdb": spec.receptor_pdb, "reference_mol": reference}
    target = _typed_state(target, cfg, train_cfg)
    clean_pool = []
    dropped = 0
    for state, kind, note in zip(pool, pool_kinds, pool_notes):
        try:
            typed = _typed_state(state, cfg, train_cfg)
            if not is_valid(typed, require_geometry=True):
                raise ValueError("Pool state fails intramolecular geometry admission")
            clean_pool.append((typed, kind, note))
        except ValueError:
            dropped += 1
    scaffolds = [(state, note) for state, kind, note in clean_pool if kind == "scaffold"]
    structural_prior = minimal_scaffold_prior([state for state, _ in scaffolds]) if scaffolds else []
    scaffold_groups, initialization_report = [], None
    if cfg.initialization_group_fraction > 0:
        scaffold_groups, initialization_report = scaffold_pose_groups(
            structural_prior, target, cfg, pocket, random.Random(data_seed + 7_919))
        initialization_report["pose_seed"] = data_seed + 7_919
        initialization_report["parent_source_notes"] = [note for state, note in scaffolds
                                                        if any(state is prior for prior in structural_prior)]
    if len(clean_pool) > args.pool_size:
        clean_pool = random.Random(data_seed).sample(clean_pool, args.pool_size)
    if len(clean_pool) < cfg.grp_min:
        raise ValueError(f"{cid}: only {len(clean_pool)} supported pool states, need at least {cfg.grp_min}")
    source_counts = dict(Counter(kind for _, kind, _ in clean_pool))
    clean_pool = [state for state, _, _ in clean_pool]
    context = mk_pkt_ctx(pocket, target, device=device) if cfg.w_pocket > 0 else None
    if cfg.w_pocket > 0 and context is None:
        raise ValueError(f"{cid}: pocket objective requested but no receptor context could be built")
    return TrainData(target, clean_pool, cid, coords, context, pdb, dropped, source_counts,
                     build_rejections, pocket_features, cache_info, database_states,
                     tuple(structural_prior), tuple(scaffold_groups), pocket, initialization_report)


def _map_cfg(cfg):
    from dataset.sampling import MapCfg
    return MapCfg(tau=cfg.tau, d_edit=cfg.delta_edit, g_min=cfg.grp_min,
                  g_max=cfg.grp_max, max_tries=cfg.group_max_tries,
                  distance_config=dict(cfg.molecular_distance))


def _groups(data, cfg, num_pairs, random_seed):
    from dataset.sampling import build_map_pool
    groups = build_map_pool(data.pool, data.X_star, num_pairs, seed=random_seed,
                            sampler=_mapping_sampler(data, cfg))
    if not len(groups):
        raise ValueError(f"{data.cid}: no mapping groups satisfy the configured basin/edit constraints")
    return groups


def _mapping_sampler(data, cfg):
    from dataset.sampling import MapSampler
    mapping_cfg = _map_cfg(cfg)
    if data._map_sampler is None or data._map_sampler.cfg != mapping_cfg:
        data._map_sampler = MapSampler(mapping_cfg)
    return data._map_sampler


def _target_embedding(model, target, hP, hc, device):
    import torch
    with torch.no_grad():
        return model.forward(target, hP, hc, device)["node_emb"].detach()


def _pocket_loss(out, target, order, ctx):
    """Pocket loss with matched reference elements and coordinates."""
    import torch
    from rdkit import Chem
    from utils.chemistry import ATOMIC_NUMBERS, atom_targets
    from utils.pkt_fit import pkt_fit_train
    positions = out["R_pred"]
    radii = positions.new_tensor([Chem.GetPeriodicTable().GetRvdw(z) for z in ATOMIC_NUMBERS])
    labels = atom_targets(target, positions.device)["atom"][order]
    reference = torch.as_tensor(target.R, device=positions.device, dtype=positions.dtype)[order]
    aligned_context = replace(ctx, ref_lig=reference, lig_r=radii[labels],
                              pharm_atom_weights=ctx.pharm_atom_weights[:, order])
    return -pkt_fit_train(positions, aligned_context).tot


def _seed_pairs(seed, prior, hp, hc, pool, cfg, graph_encoder, rng):
    """Bijection between equally sized seed batches under the same condition."""
    import torch
    from scipy.optimize import linear_sum_assignment
    from utils.losses import paper_state_distance

    first_index = rng.randrange(len(pool))
    first = pool[first_index]
    count = first.num_nodes()
    candidates = [i for i, state in enumerate(pool) if i != first_index and state.num_nodes() == count]
    indices = [first_index, *rng.sample(candidates, min(cfg.seed_match_samples - 1, len(candidates)))]
    targets = [pool[i] for i in indices]
    outputs = [seed.forward_trainable(prior.rsample(hp, hc), None, hp, hc, num_nodes=count)
               for _ in targets]
    with torch.no_grad():
        costs = torch.stack([torch.stack([
            paper_state_distance(out, target, cfg, graph_encoder)[0]["total"]
            for target in targets]) for out in outputs])
    rows, columns = linear_sum_assignment(costs.cpu().numpy())
    return [(outputs[i], targets[j]) for i, j in zip(rows, columns)]


def _initialization_group(seed, prior, hp, hc, data, group, cfg, rng, *, task_dropped):
    from dataset.sampling import MapGroup
    from model.initialization import group_admission_reason, sample_initial_group, sampling_mode

    metrics = {"init_attempted": 0., "init_raw_accepted": 0., "init_scaffold_pose_accepted": 0.,
               "init_original_fallback": 0.}
    if (cfg.initialization_group_fraction == 0 or seed is None or prior is None
            or rng.random() >= cfg.initialization_group_fraction):
        return group, metrics
    alpha = cfg.alpha_prior if data.structural_prior and not task_dropped else 0.
    with sampling_mode(seed, prior):
        states, record = sample_initial_group(seed, prior, hp, hc, cfg, rng,
            structural_prior=data.structural_prior, alpha_prior=alpha)
    source = "mixed" if len(set(record["initial_sources"])) > 1 else record["initial_sources"][0]
    metrics.update(init_attempted=1., init_raw_alpha_sum=alpha,
                   init_raw_group_size_sum=len(states),
                   init_raw_count_min_sum=min(record["initial_atom_counts"]),
                   init_raw_count_max_sum=max(record["initial_atom_counts"]))
    metrics["init_raw_source_" + source] = 1.
    metrics["init_mode_" + ("mixture" if alpha > 0 else "conditional_seed")] = 1.
    reason = group_admission_reason(states, data.X_star, cfg, data.initialization_pocket)
    if reason is None:
        metrics["init_raw_accepted"] = 1.
        metrics["init_raw_accepted_source_" + source] = 1.
        return MapGroup(tuple(states), data.X_star), metrics
    metrics["init_raw_rejected_" + reason] = 1.
    if data.scaffold_groups:
        anchor = rng.choice(data.scaffold_groups)
        size = rng.randint(cfg.grp_min, min(cfg.grp_max, len(anchor.states)))
        members = [anchor.states[0], *rng.sample(list(anchor.states[1:]), size - 1)]
        metrics["init_scaffold_pose_accepted"] = 1.
        return MapGroup(tuple(members), data.X_star), metrics
    metrics["init_original_fallback"] = 1.
    return group, metrics


def group_loss(model, seed, prior, pe, te, data, group, cfg, device, rng):
    """Target, basin, pocket, chemistry, and initialization objectives.

    Seed training samples the conditional prior; there is no posterior or KL term.
    """
    import torch
    from dataset.protein_data import task_vec
    from dataset.sampling import _key
    from model.edge_diffusion import edge_loss
    from model.pocket_context import with_summary, summary_of
    from utils.losses import aligned_edge_target, chemical_loss, paper_state_distance

    coords = torch.as_tensor(data.coords, device=device, dtype=torch.float32)
    task = torch.as_tensor(task_vec(data.X_star, model.cond_dim), device=device, dtype=torch.float32)
    task_dropped = rng.random() < cfg.task_dropout
    if task_dropped:
        task = torch.zeros_like(task)
    if data.pocket_features is None:
        raise ValueError("Typed pocket atom features are required for training")
    features = torch.as_tensor(data.pocket_features, device=device, dtype=torch.float32)
    hp, hc = pe(coords, feat=features, return_context=True), te(task)
    group, initialization_metrics = _initialization_group(
        seed, prior, hp, hc, data, group, cfg, rng, task_dropped=task_dropped)
    out = model.forward_group_step_trainable(list(group.states), hp, hc, device=device,
                                            num_nodes=group.X_star.num_nodes())
    metrics, paths = {"chem_map_total": 0., **initialization_metrics}, []
    total = out["R_pred"].sum() * 0.
    chemistry_total = total * 0.

    def record(prefix, values):
        metrics.update({prefix + key: float(value.detach()) for key, value in values.items()})

    fraction = cfg.single_target_fraction
    group_hp = with_summary(hp, summary_of(hp) + model.group_to_hp(out["group_vec"]))
    group_hc = hc + model.group_to_hc(out["group_vec"])
    paths.append(("group", 1. - fraction, out, group_hp, group_hc))
    if fraction > 0:
        single = model.forward_step_trainable(rng.choice(group.states), hp, hc, device=device,
                                              num_nodes=group.X_star.num_nodes())
        paths.append(("single", fraction, single, hp, hc))
    aggregates = {}
    for name, weight, prediction, path_hp, path_hc in paths:
        state, transport = paper_state_distance(prediction, group.X_star, cfg, model.graph_encoder)
        embedding = (_target_embedding(model, group.X_star, path_hp, path_hc, device)
                     if cfg.w_embedding > 0 else None)
        chemistry, order = chemical_loss(prediction, group.X_star, cfg, target_embedding=embedding,
                                        order=transport.argmax(dim=1))
        if cfg.w_chem > 0 and cfg.w_edge_op > 0 and weight > 0:
            adjacency = aligned_edge_target(group.X_star, order, prediction["node_emb"].dtype)
            value = edge_loss(model.edge_denoiser, adjacency, prediction["node_emb"], path_hp, path_hc, cfg.edge_T)
            chemistry["edge_denoising"] = value
            chemistry["total"] = chemistry["total"] + cfg.w_edge_op * value
        record("target_" + name + "_", state)
        record("chem_" + name + "_", chemistry)
        for key, value in state.items():
            aggregates[key] = aggregates.get(key, total * 0.) + weight * value
        total = total + weight * cfg.w_target * state["total"]
        chemistry_total = chemistry_total + weight * chemistry["total"]
        metrics["chem_map_total"] += weight * float(chemistry["total"].detach())
        if cfg.w_pocket > 0 and weight > 0:
            if data.pkt_ctx is None:
                raise ValueError("Pocket objective requested without a context")
            value = _pocket_loss(prediction, group.X_star, order, data.pkt_ctx)
            total = total + cfg.w_pocket * weight * value
            metrics["pocket_" + name] = float(value.detach())
            metrics["pocket"] = metrics.get("pocket", 0.) + weight * float(value.detach())
    record("target_", aggregates)

    if cfg.w_basin > 0:
        sampler = _mapping_sampler(data, cfg)
        members = {_key(state) for state in group.states}
        second = None
        for _ in range(cfg.group_max_tries):
            candidate = sampler.sample_one(data.pool, data.X_star, rng)
            if candidate is not None and {_key(state) for state in candidate.states} != members:
                second = candidate
                break
        if second is None:
            raise ValueError(f"{data.cid}: no distinct second mapping group found after {cfg.group_max_tries} attempts; "
                             "increase the eligible pool or explicitly set --w-basin 0")
        out_b = model.forward_group_step_trainable(list(second.states), hp, hc, device=device,
                                                  num_nodes=data.X_star.num_nodes())
        basin, _ = paper_state_distance(out, out_b, cfg, model.graph_encoder)
        total = total + cfg.w_basin * basin["total"]
        record("basin_", basin)
        metrics["basin"] = float(basin["total"].detach())

    if seed is not None and prior is not None:
        pairs = _seed_pairs(seed, prior, hp, hc, data.pool, cfg, model.graph_encoder, rng)
        weight = 1. / len(pairs)
        for seed_out, seed_target in pairs:
            state, transport = paper_state_distance(seed_out, seed_target, cfg, model.graph_encoder)
            embedding = _target_embedding(model, seed_target, hp, hc, device) if cfg.w_embedding > 0 else None
            chemistry, seed_order = chemical_loss(seed_out, seed_target, cfg, target_embedding=embedding,
                                                 order=transport.argmax(dim=1))
            if cfg.w_chem > 0 and cfg.w_edge > 0:
                adjacency = aligned_edge_target(seed_target, seed_order, seed_out["node_emb"].dtype)
                value = edge_loss(seed.edge_denoiser, adjacency, seed_out["node_emb"], hp, hc, cfg.edge_T)
                chemistry["edge_denoising"] = value
                chemistry["total"] = chemistry["total"] + cfg.w_edge * value
            total = total + weight * cfg.b_seed * state["total"]
            chemistry_total = chemistry_total + weight * chemistry["total"]
            terms = [("seed_state_", state), ("chem_seed_", chemistry)]
            if cfg.g_init > 0:
                initial, _ = paper_state_distance(seed_out, data.X_star, cfg, model.graph_encoder)
                total = total + weight * cfg.g_init * initial["total"]
                terms.append(("initial_", initial))
                metrics["seed_neighborhood"] = metrics.get("seed_neighborhood", 0.) + weight * float(initial["total"].detach())
            for prefix, values in terms:
                for key, value in values.items():
                    name = prefix + key
                    metrics[name] = metrics.get(name, 0.) + weight * float(value.detach())
        metrics["seed"] = metrics["seed_state_total"]
    total = total + cfg.w_chem * chemistry_total
    metrics["chem_total"] = float(chemistry_total.detach())
    if not bool(torch.isfinite(total)):
        raise FloatingPointError(f"{data.cid}: objective is not finite")
    metrics["total"] = float(total.detach())
    return total, metrics


def build_training_modules(args, cfg, train_cfg, device):
    from model import PocketEncoder, TaskEncoder, build_model
    from dataset.protein_data import PROTEIN_FEATURE_DIM
    model, seed, prior = build_model(
        train_cfg.node_feat_dim, train_cfg.cond_dim, cfg.z_dim,
        num_nodes=min(12, cfg.max_nodes), max_nodes=cfg.max_nodes,
        edge_T=cfg.edge_T, prior_hidden=args.prior_hidden,
        joint_steps=cfg.joint_steps, pocket_neighbors=cfg.pocket_neighbors,
        pocket_step_scale=cfg.pocket_step_scale,
        num_anchors=cfg.num_anchors,
        core_kwargs={"hidden": args.hidden, "num_layers": args.layers},
    )
    pe = PocketEncoder(train_cfg.cond_dim, radius=args.pocket_r, feat_dim=PROTEIN_FEATURE_DIM,
                       hidden_nf=args.pocket_hidden, n_layers=args.pocket_layers)
    te = TaskEncoder(train_cfg.cond_dim, train_cfg.cond_dim)
    for module in (model, seed, prior, pe, te):
        module.to(device)
    model.distance_config = dict(cfg.molecular_distance)
    return model, seed, prior, pe, te


def _source_revision():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True, text=True, timeout=5)
        revision = result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        revision = None
    digest = hashlib.sha256()
    paths = [*_ROOT.glob("*.py")]
    for directory in ("model", "dataset", "utils"):
        paths.extend((_ROOT / directory).glob("*.py"))
    for path in sorted(paths):
        digest.update(path.relative_to(_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return {"git_commit": revision, "source_sha256": digest.hexdigest()}


def _versions():
    versions = {"python": sys.version.split()[0]}
    for package in ("torch", "numpy", "networkx", "rdkit", "scipy", "egnn-pytorch", "torch-geometric",
                    "crem", "ScaffoldGraph", "plip", "openbabel-wheel"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        from openbabel import openbabel
        versions["openbabel_runtime"] = str(openbabel.OBReleaseVersion())
    except (ImportError, OSError, AttributeError):
        versions["openbabel_runtime"] = None
    return versions


def _metadata(args, cfg, train_cfg, specs, summaries, updates):
    from config import IMPLEMENTATION_REVISION, DECODER_KIND, SEED_OBJECTIVE, STATE_DISTANCE_SCHEMA, ANCHOR_KIND, LOSS_SCHEMA, GRAPH_ENCODER_TRAINING, SEED_PAIRING, GROUP_INPUT_SCHEMA
    from dataset.protein_data import TASK_VECTOR_REPRESENTATION, PROTEIN_FEATURE_DIM, PROTEIN_FEATURE_SCHEMA
    from config import POCKET_SCORE_SCHEMA, POCKET_RECEPTOR_SCOPE, POCKET_RADIUS_SOURCE, POCKET_PHARM_SCHEMA, VALENCE_PENALTY_SCHEMA
    from utils.pkt_fit import PktWt
    from model.initialization import initialization_training_metadata
    from utils.geometry_validation import GEOMETRY_VALIDATION_SCHEMA, GeometryValidationConfig
    from dataset.state_pool import PoolCfg, POOL_CAP_POLICY
    return {
        "schema_version": 2, "implementation": "ABCD",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "task_feature_schema": TASK_VECTOR_REPRESENTATION,
        "pocket_feature_dim": PROTEIN_FEATURE_DIM, "pocket_feature_schema": PROTEIN_FEATURE_SCHEMA,
        "decoder_kind": DECODER_KIND, "seed_objective": SEED_OBJECTIVE,
        "state_distance_schema": STATE_DISTANCE_SCHEMA, "anchor_kind": ANCHOR_KIND,
        "graph_encoder_training": GRAPH_ENCODER_TRAINING,
        "graph_encoder_initialization": "torch_default_from_training_rng",
        "graph_encoder_initialization_global_seed": args.seed,
        "loss_schema": LOSS_SCHEMA,
        "valence_penalty_schema": VALENCE_PENALTY_SCHEMA,
        "seed_pairing": SEED_PAIRING,
        "group_input_schema": GROUP_INPUT_SCHEMA,
        "initialization_training": initialization_training_metadata(cfg),
        "state_pool_geometry": {"schema": GEOMETRY_VALIDATION_SCHEMA,
                                "thresholds": dataclasses.asdict(GeometryValidationConfig()),
                                "scope": "intramolecular_candidate_admission"},
        "pool_builder_defaults": dataclasses.asdict(PoolCfg()),
        "pool_cap_policy": POOL_CAP_POLICY,
        "pocket_score_schema": POCKET_SCORE_SCHEMA,
        "pocket_receptor_scope": POCKET_RECEPTOR_SCOPE,
        "pocket_radius_source": POCKET_RADIUS_SOURCE,
        "pocket_pharm_schema": POCKET_PHARM_SCHEMA,
        "pocket_score_weights": dataclasses.asdict(PktWt()),
        "training_status": "optimized_unvalidated" if updates else "untrained",
        "optimizer_steps": updates, "epochs": args.epochs if updates else 0,
        "mode": args.mode, "scientific_evaluation": "not_performed",
        "node_feat_dim": train_cfg.node_feat_dim, "cond_dim": train_cfg.cond_dim,
        "z_dim": cfg.z_dim, "max_nodes": cfg.max_nodes, "num_nodes": min(12, cfg.max_nodes),
        "edge_T": cfg.edge_T, "hidden": args.hidden, "n_layers": args.layers,
        "core_kwargs": {"hidden": args.hidden, "num_layers": args.layers,
                        "radius_cutoff": 6.0, "use_radius_edges": True},
        "pocket_hidden": args.pocket_hidden, "pocket_layers": args.pocket_layers,
        "prior_hidden": args.prior_hidden, "pocket_r": args.pocket_r,
        "pocket_max_atoms": train_cfg.pocket_atoms, "molecular_distance": cfg.molecular_distance,
        "dataset_ids": [f"synthetic_{spec}_atoms" if args.mode == "test" else spec.complex_id for spec in specs],
        "complex_summaries": summaries,
        "cli_arguments": _jsonable({key: value for key, value in vars(args).items() if not key.startswith("_")}),
        "dataset_provenance": getattr(args, "_data_report", None),
        "data_backend_provenance": getattr(args, "_data_backend_report", None),
        "source_revision": _source_revision(), "package_versions": _versions(),
        "pocket_surrogate": "pharmacophore_shape_and_vdw_repulsion_with_matched_target_elements",
        "defaults_provenance": "config.py_and_command_line",
    }


def _save_ckpt(path, *, modules, cfg, train_cfg, meta):
    import os
    import tempfile
    import torch
    if meta.get("optimizer_steps", 0) < 1:
        raise ValueError("Refusing to save an untrained/dry-run artifact as a training checkpoint")
    model, seed, prior, pe, te = modules
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "model": model.state_dict(), "seed": seed.state_dict(), "prior": prior.state_dict(),
        "pocket_encoder": pe.state_dict(), "task_encoder": te.state_dict(),
        "abcm_config": dataclasses.asdict(cfg),
        "train_config": dataclasses.asdict(train_cfg), "meta": meta,
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _move_context(data, device):
    """Move pocket context tensors to the training device."""
    import torch
    if data.pkt_ctx is not None:
        context = data.pkt_ctx
        data.pkt_ctx = replace(context, **{
            field.name: getattr(context, field.name).to(device)
            for field in dataclasses.fields(context)
            if isinstance(getattr(context, field.name), torch.Tensor)
        })
    return data


def _run_training(args, cfg, train_cfg, device):
    import numpy as np
    import torch

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args._data_backend_report = _data_preflight(args, cfg)
    specs = _specs(args)
    print(json.dumps({"dataset": args._data_report,
                      "data_backends": args._data_backend_report}), flush=True)
    modules = build_training_modules(args, cfg, train_cfg, device)
    parameters = _uniq_params(*modules)
    optimizer = None if args.dry_run else torch.optim.AdamW(parameters, lr=args.lr)
    updates, summaries = 0, {}
    for epoch in range(1 if args.dry_run else args.epochs):
        epoch_specs = list(specs)
        if not args.dry_run:
            random.Random(args.seed + epoch).shuffle(epoch_specs)
        for spec in epoch_specs:
            data_seed = _complex_seed(args, spec)
            group_seed = data_seed + epoch * 1_000_003
            random.seed(data_seed)
            np.random.seed(data_seed)
            data = _load_one(spec, args, cfg, train_cfg, torch.device("cpu"), data_seed=data_seed)
            groups = list(_groups(data, cfg, args.num_pairs, group_seed))
            random.Random(group_seed).shuffle(groups)
            data = _move_context(data, device)
            rng = random.Random(group_seed)
            torch.manual_seed(group_seed)
            summaries[data.cid] = {"target_atoms": data.X_star.num_nodes(), "pool_states": len(data.pool),
                                   "pool_atom_counts": sorted({state.num_nodes() for state in data.pool}),
                                   "pool_source_counts": data.pool_source_counts,
                                   "pool_build_rejections": data.pool_build_rejections,
                                   "state_pool_seed": data_seed,
                                   "pool_cache": data.pool_cache,
                                   "initialization_scaffold_poses": data.initialization_report,
                                   "mmp_database_states_before_typed_filter": data.mmp_database_states,
                                   "groups": len(groups), "dropped_pool_states": data.dropped_pool_states}
            print(json.dumps({"epoch": epoch, "complex": data.cid, **summaries[data.cid]}), flush=True)
            for module in modules:
                module.train(not args.dry_run)
            if args.dry_run:
                with torch.no_grad():
                    _, metrics = group_loss(*modules, data, groups[0], cfg, device, rng)
                print(json.dumps({"dry_run": True, "complex": data.cid, "objective": metrics}), flush=True)
                continue
            metrics_sum = defaultdict(float)
            for start in range(0, len(groups), train_cfg.batch_size):
                batch = groups[start:start + train_cfg.batch_size]
                optimizer.zero_grad(set_to_none=True)
                for group in batch:
                    loss, metrics = group_loss(*modules, data, group, cfg, device, rng)
                    (loss * (1.0 / len(batch))).backward()
                    for key, value in metrics.items():
                        metrics_sum[key] += value
                if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                       for parameter in parameters):
                    raise FloatingPointError("Nonfinite training gradients")
                optimizer.step()
                if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
                    raise FloatingPointError("Nonfinite model parameters")
                updates += 1
            print(json.dumps({"epoch": epoch, "complex": data.cid,
                              "mean_objective": {key: value / len(groups)
                                                 for key, value in metrics_sum.items()}}), flush=True)
        if not args.dry_run:
            meta = _metadata(args, cfg, train_cfg, specs, summaries, updates)
            meta["epochs"] = epoch + 1
            _save_ckpt((args.output_dir / args.checkpoint).resolve(), modules=modules,
                       cfg=cfg, train_cfg=train_cfg, meta=meta)
    if args.dry_run:
        print("Dry run complete: no optimizer updates or checkpoint saved.")
    return 0


def main(argv=None, *, defaults=None):
    args = _parse(argv, defaults=defaults)
    cfg, train_cfg = _cfg(args), _train_cfg(args)
    if args.print_config:
        from config import IMPLEMENTATION_REVISION, LOSS_SCHEMA, GRAPH_ENCODER_TRAINING, SEED_PAIRING, GROUP_INPUT_SCHEMA
        from config import POCKET_SCORE_SCHEMA, POCKET_RECEPTOR_SCOPE, POCKET_RADIUS_SOURCE, POCKET_PHARM_SCHEMA, VALENCE_PENALTY_SCHEMA
        from utils.pkt_fit import PktWt
        from model.initialization import initialization_training_metadata
        from utils.geometry_validation import GEOMETRY_VALIDATION_SCHEMA, GeometryValidationConfig
        from dataset.state_pool import PoolCfg, POOL_CAP_POLICY
        print(json.dumps({"abcm_config": dataclasses.asdict(cfg), "train_config": dataclasses.asdict(train_cfg),
                          "implementation_revision": IMPLEMENTATION_REVISION,
                          "loss_schema": LOSS_SCHEMA,
                          "valence_penalty_schema": VALENCE_PENALTY_SCHEMA,
                          "graph_encoder_training": GRAPH_ENCODER_TRAINING,
                          "seed_pairing": SEED_PAIRING,
                          "group_input_schema": GROUP_INPUT_SCHEMA,
                          "initialization_training": initialization_training_metadata(cfg),
                          "state_pool_geometry": {"schema": GEOMETRY_VALIDATION_SCHEMA,
                              "thresholds": dataclasses.asdict(GeometryValidationConfig()),
                              "scope": "intramolecular_candidate_admission"},
                          "pool_builder_defaults": dataclasses.asdict(PoolCfg()),
                          "pool_cap_policy": POOL_CAP_POLICY,
                          "pocket_score_schema": POCKET_SCORE_SCHEMA,
                          "pocket_receptor_scope": POCKET_RECEPTOR_SCOPE,
                          "pocket_radius_source": POCKET_RADIUS_SOURCE,
                          "pocket_pharm_schema": POCKET_PHARM_SCHEMA,
                          "pocket_score_weights": dataclasses.asdict(PktWt()),
                          "cli_arguments": _jsonable({key: value for key, value in vars(args).items() if not key.startswith("_")}),
                          "status": "new_unvalidated_reconstruction_defaults"}, indent=2))
        return 0
    if args.epochs < 1 and not args.dry_run:
        raise ValueError("--epochs must be positive for training")
    return _run_training(args, cfg, train_cfg, _dev(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
