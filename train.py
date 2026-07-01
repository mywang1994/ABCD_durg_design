from __future__ import annotations

import argparse
import dataclasses
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from config import ABCMConfig, HIDDEN, N_LAYERS, POCKET_HIDDEN, POCKET_LAYERS, PRIOR_HIDDEN, TrainConfig
from dataset.sampling import MapGroup, MapPool, MapCfg, MapSampler, TrajChain, TrajPool, build_map_pool, build_traj_pool
from dataset.protein_data import discover_crossdock_pocket10, pocket_coords, task_vec
from dataset.state_pool import build_crossdock_bundle, build_state_pool_test, filter_pool_by_edit_delta
from dataset.pool_core import PoolCfg, load_reference_mol_from_sdf
from mol_state import MolState
from model import PocketEncoder, TaskEncoder, build_model, edge_loss
from utils.losses import (
    adjacency_from_molstate,
    loss_basin_group,
    loss_basin_step,
    loss_fixpoint,
    loss_mapping_group,
    loss_pkt_fit,
    loss_trajectory_chain,
    tensor_alignment_to_molstate,
)
from utils.pkt_fit import mk_pkt_ctx, pkt_fit

_FLOAT_CFG = frozenset({
    "w_geom", "w_graph", "b_global", "b_seed", "g_init", "w_edge", "w_edge_op",
    "w_fp", "w_pocket", "pocket_min", "tau", "w_basin", "fp_star",
})
_CFG_KEYS = (
    "w_geom", "w_graph", "b_global", "b_seed", "g_init", "w_edge", "w_edge_op",
    "w_fp", "fp_steps", "fp_starts", "w_pocket", "pocket_min", "chain_len",
    "carry_bptt", "tau", "w_basin", "grp_min", "grp_max", "edge_T",
)
_TRAIN_CFG_KEYS = ("batch_size", "node_feat_dim", "cond_dim", "pocket_atoms", "test_pts")


@dataclass
class TrainData:
    X_star: MolState
    pool: list[MolState]
    traj: list | MapPool | TrajPool
    cid: str
    note: str
    coords: np.ndarray | None = None
    pdb: str | None = None
    pkt_ctx: object | None = None
    pocket_tgt: float | None = None


def _items(ds) -> list:
    if isinstance(ds, MapPool):
        return ds.groups
    if isinstance(ds, TrajPool):
        return ds.samples
    return ds


def _xstar(ds, items: list) -> MolState | None:
    if isinstance(ds, MapPool):
        return ds.X_star
    if isinstance(ds, TrajPool):
        return ds.x_star
    return items[0].X_star if items else None


def _uniq_params(*mods: torch.nn.Module) -> list[torch.nn.Parameter]:
    seen: set[int] = set()
    out: list[torch.nn.Parameter] = []
    for m in mods:
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
    return out


def _cond(fn, hP, hc, dev):
    return fn() if fn else (hP.to(dev), hc.to(dev))


def train(
    model,
    opt,
    dataset,
    *,
    cfg: ABCMConfig,
    dev: torch.device,
    cond_fn=None,
    hP=None,
    hc=None,
    seed=None,
    prior=None,
    pool=None,
    bs: int = 32,
    ep: int = 0,
    rng_seed: int = 0,
    pkt_ctx=None,
    pocket_tgt: float | None = None,
    crd_loss: str = "mse",
    kabsch: bool = False,
    fp_bptt: int | None = None,
    x_star: MolState | None = None,
) -> dict[str, float]:
    model.train()
    if seed:
        seed.train()
    if prior:
        prior.train()
    joint = seed is not None and prior is not None
    if joint and not pool:
        raise ValueError("empty pool")
    if cond_fn is None and (hP is None or hc is None):
        raise TypeError("need cond_fn or hP/hc")

    totals = defaultdict(float)
    n_fp = 0
    items = list(_items(dataset))
    x_star_ref = _xstar(dataset, items)
    by_group = isinstance(dataset, MapPool)
    sampler = MapSampler(MapCfg(tau=cfg.tau, d_edit=cfg.delta_edit, g_min=cfg.grp_min, g_max=cfg.grp_max))
    step_scale = getattr(model, "step_scale", 0.1)
    batch_n = max(1, bs)
    loss_kw = dict(
        w_g=cfg.w_geom, w_gr=cfg.w_graph, delta_edit=cfg.delta_edit, device=dev,
        coord_loss=crd_loss, use_kabsch=kabsch, pkt_ctx=pkt_ctx,
        w_pocket=cfg.w_pocket, pocket_tgt=pocket_tgt, w_edge_op=cfg.w_edge_op,
    )
    random.Random(rng_seed + ep * 1_000_003 + 17).shuffle(items)

    def _seed_loss(hp, hc_i, X):
        z = prior.rsample(hp, hc_i)
        out0 = seed.forward_trainable(z, None, hp, hc_i)
        l_init = tensor_alignment_to_molstate(out0["R"], out0["node_feat"], X, w_g=cfg.w_init_g, w_f=cfg.w_init_f, device=dev, coord_loss=crd_loss, use_kabsch=kabsch)
        l_seed = tensor_alignment_to_molstate(out0["R"], out0["node_feat"], random.choice(pool), w_g=cfg.w_seed_g, w_f=cfg.w_seed_f, device=dev, coord_loss=crd_loss, use_kabsch=kabsch)
        adj_tgt = adjacency_from_molstate(X, seed.num_nodes, device=dev, dtype=out0["node_emb"].dtype)
        l_edge = edge_loss(seed.edge_denoiser, adj_tgt, out0["node_emb"], hp, hc_i, T=seed.edge_T)
        loss = cfg.b_seed * l_seed + cfg.g_init * l_init + cfg.w_edge * l_edge
        if pkt_ctx and cfg.w_pocket > 0:
            l_pocket = loss_pkt_fit(out0["R"], pkt_ctx, tgt=pocket_tgt)
            loss = loss + cfg.w_pocket * 0.5 * l_pocket
            totals["pocket_s"] += float(l_pocket.detach().cpu())
        totals["seed"] += float(l_seed.detach().cpu())
        totals["init"] += float(l_init.detach().cpu())
        totals["edge_s"] += float(l_edge.detach().cpu())
        return loss

    def _group(g: MapGroup) -> torch.Tensor:
        hp, hc_i = _cond(cond_fn, hP, hc, dev)
        res = loss_mapping_group(model, g, hP=hp, hc=hc_i, w_basin=0.0, **loss_kw)
        loss = res.tot
        if cfg.w_basin > 0 and pool:
            pair = sampler.sample_pair(pool, g.X_star, random.Random(random.randint(0, 2**31 - 1)))
            if pair:
                l_basin = loss_basin_group(model, pair[0], pair[1], hP=hp, hc=hc_i, w_g=cfg.w_geom, w_gr=cfg.w_graph, device=dev, coord_loss=crd_loss, use_kabsch=kabsch)
                loss = loss + cfg.w_basin * l_basin
                totals["basin"] += float(l_basin.detach().cpu())
        if joint:
            loss = loss + _seed_loss(hp, hc_i, g.X_star)
        totals["tgt"] += float(res.tgt.detach().cpu())
        totals["glob"] += float(res.tgt.detach().cpu())
        totals["edit"] += float(res.edit.detach().cpu())
        if cfg.w_pocket > 0:
            totals["pocket"] += float(res.pocket.detach().cpu())
        if cfg.w_edge_op > 0:
            totals["edge"] += float(res.edge.detach().cpu())
        totals["gsz"] += float(res.gsz)
        totals["L"] += float(loss.detach().cpu())
        return loss

    def _chain(ch: TrajChain) -> torch.Tensor:
        hp, hc_i = _cond(cond_fn, hP, hc, dev)
        if cfg.basin_leap:
            res = loss_basin_step(model, ch, hP=hp, hc=hc_i, b_global=cfg.b_global, step_scale=step_scale, **loss_kw)
        else:
            res = loss_trajectory_chain(
                model, ch, hP=hp, hc=hc_i, b_global=cfg.b_global, step_scale=step_scale,
                chain_global_ramp=cfg.chain_ramp, carry_rollout=cfg.chain_carry, carry_bptt_steps=cfg.carry_bptt, **loss_kw,
            )
        loss = res.tot
        if joint:
            loss = loss + _seed_loss(hp, hc_i, ch.X_star)
        totals["loc"] += float(res.loc.detach().cpu())
        totals["glob"] += float(res.glob.detach().cpu())
        totals["edit"] += float(res.edit.detach().cpu())
        if cfg.w_pocket > 0:
            totals["pocket"] += float(res.pocket.detach().cpu())
        if cfg.w_edge_op > 0:
            totals["edge"] += float(res.edge.detach().cpu())
        totals["steps"] += float(res.n)
        totals["L"] += float(loss.detach().cpu())
        return loss

    for i in range(0, len(items), batch_n):
        batch = items[i : i + batch_n]
        opt.zero_grad(set_to_none=True)
        batch_loss = sum(_group(x) if by_group and cfg.group_train else _chain(x) for x in batch) / len(batch)
        if cfg.w_fp > 0 and cfg.fp_steps > 0 and pool and len(pool) >= 2:
            hp, hc_i = _cond(cond_fn, hP, hc, dev)
            starts = random.Random(rng_seed + ep * 997 + i * 13).sample(pool, min(cfg.fp_starts, len(pool)))
            bptt = fp_bptt if fp_bptt is not None else cfg.carry_bptt
            l_fp = loss_fixpoint(
                model, starts, hP=hp, hc=hc_i, device=dev, steps=cfg.fp_steps, step_scale=step_scale,
                w_g=cfg.w_geom, w_gr=cfg.w_graph, carry_bptt_steps=bptt,
                X_star=x_star or x_star_ref, star_align=cfg.fp_star,
            )
            batch_loss = batch_loss + cfg.w_fp * l_fp
            totals["fp"] += float(l_fp.detach().cpu())
            n_fp += 1
        batch_loss.backward()
        opt.step()

    n = max(1, len(items))
    m = {k: v / n for k, v in totals.items()}
    if by_group and cfg.group_train and "gsz" in m:
        m["gsz"] = totals["gsz"] / n
    if n_fp and "fp" in m:
        m["fp"] = totals["fp"] / n_fp
    return m


def _traj_pool(states, x_star, cfg: ABCMConfig, n: int, seed: int):
    if cfg.group_train and cfg.basin_leap:
        mg = MapCfg(tau=cfg.tau, d_edit=cfg.delta_edit, g_min=cfg.grp_min, g_max=cfg.grp_max)
        return build_map_pool(pool=states, x_star=x_star, n=n, seed=seed, cfg=mg)
    return build_traj_pool(states, x_star, cfg.delta_edit, n, seed, clen=cfg.chain_len, basin_leap=cfg.basin_leap)


def _load_data(args, cfg: ABCMConfig, train_cfg: TrainConfig, dev) -> TrainData:
    pocket_tgt = args.ps_tgt
    if args.mode == "test":
        X = build_state_pool_test.make_target_state(num_nodes=12)
        pool = filter_pool_by_edit_delta(build_state_pool_test.build_pool(X, args.pool_size), X, cfg.delta_edit)
        return TrainData(X, pool, _traj_pool(pool, X, cfg, args.num_pairs, args.seed), "test", "test")

    complexes = discover_crossdock_pocket10(args.pocket10_root.resolve(), args.max_clusters, args.max_pairs_per_cluster)
    if not complexes:
        sys.exit(f"no data: {args.pocket10_root}")
    if not 0 <= args.complex_index < len(complexes):
        sys.exit(f"bad complex-index")

    complex = complexes[args.complex_index]
    pool_cfg = PoolCfg(seed=args.seed, pocket_min=cfg.pocket_min)
    if args.mmp_db:
        pool_cfg = dataclasses.replace(pool_cfg, mmp_db=args.mmp_db.resolve())
    bundle = build_crossdock_bundle(complex, pool_cfg)
    X = bundle.X_star
    pool = filter_pool_by_edit_delta(bundle.all_molstates(), X, cfg.delta_edit)
    if not pool:
        sys.exit(f"empty pool: {complex.complex_id}")

    coords = pocket_coords(complex.receptor_pdb, max_atoms=train_cfg.pocket_atoms, seed=args.seed)
    pocket_pack = {"receptor_pdb": complex.receptor_pdb, "receptor_coords": coords, "reference_mol": load_reference_mol_from_sdf(complex.reference_ligand_sdf)}
    if coords.shape[0] < 4:
        coords = np.asarray(np.random.default_rng(args.seed).standard_normal((max(64, train_cfg.test_pts), 3)), dtype=np.float32)
        pocket_pack["receptor_coords"] = coords

    pkt_ctx = None
    if cfg.w_pocket > 0:
        pkt_ctx = mk_pkt_ctx(pocket_pack, X, device=dev)
        if pkt_ctx and pocket_tgt is None:
            pocket_tgt = float(pkt_fit(X, pocket_pack).tot)

    note = f"crossdock:{complex.complex_id}"
    if cfg.pocket_min is not None:
        note += f":min={cfg.pocket_min}"
    return TrainData(X, pool, _traj_pool(pool, X, cfg, args.num_pairs, args.seed), complex.complex_id, note, coords, str(complex.receptor_pdb.resolve()), pkt_ctx, pocket_tgt)


def _save_ckpt(path, *, model, seed, prior, pe, te, opt, cfg, train_cfg, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "seed": seed.state_dict(), "prior": prior.state_dict(),
        "pocket_encoder": pe.state_dict(), "task_encoder": te.state_dict(),
        "optimizer": opt.state_dict(),
        "abcm_config": dataclasses.asdict(cfg),
        "train_config": dataclasses.asdict(train_cfg),
        "meta": meta,
    }, path)


def _parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("test", "crossdock"), default="crossdock")
    p.add_argument("--pocket10-root", type=Path, default=_ROOT / "dataset" / "crossdocked_pocket10")
    p.add_argument("--mmp-db", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=_ROOT / "output")
    p.add_argument("--checkpoint", default="abcm_checkpoint.pt")
    p.add_argument("--device", default="gpu")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1.4e-3)
    p.add_argument("--num-pairs", type=int, default=512)
    for k in _TRAIN_CFG_KEYS:
        p.add_argument(f"--{k.replace('_', '-')}", type=int, default=None)
    p.add_argument("--progressive", action="store_true")
    p.add_argument("--chain-len", type=int, default=None)
    p.add_argument("--no-chain-ramp", action="store_true")
    p.add_argument("--no-chain-carry", action="store_true")
    p.add_argument("--carry-bptt", type=int, default=None)
    p.add_argument("--pool-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-clusters", type=int, default=1)
    p.add_argument("--max-pairs-per-cluster", type=int, default=2)
    p.add_argument("--complex-index", type=int, default=0)
    p.add_argument("--coord-loss", default="mse", choices=("mse", "kabsch"))
    p.add_argument("--kabsch", action="store_true")
    for k in _CFG_KEYS:
        p.add_argument(f"--{k.replace('_', '-')}", type=float if k in _FLOAT_CFG else int, default=None)
    p.add_argument("--ps-tgt", type=float, default=None)
    p.add_argument("--fp-bptt", type=int, default=None)
    p.add_argument("--no-group", action="store_true")
    p.add_argument("--pocket-r", type=float, default=6.0)
    return p.parse_args(argv)


def _cfg(args) -> ABCMConfig:
    if args.progressive:
        return replace(ABCMConfig(), basin_leap=False, chain_len=3, chain_ramp=True, chain_carry=True, w_fp=0.1, fp_steps=3, b_global=0.2)
    o = {k: getattr(args, k) for k in _CFG_KEYS if getattr(args, k) is not None}
    if args.chain_len and args.chain_len > 1:
        o["basin_leap"] = False
    if args.no_chain_ramp:
        o["chain_ramp"] = False
    if args.no_chain_carry:
        o["chain_carry"] = False
    if args.no_group:
        o["group_train"] = False
    return replace(ABCMConfig(), **o) if o else ABCMConfig()


def _train_cfg(args) -> TrainConfig:
    o = {k: getattr(args, k) for k in _TRAIN_CFG_KEYS if getattr(args, k) is not None}
    if args.fp_bptt is not None:
        o["fp_bptt"] = args.fp_bptt
    return replace(TrainConfig(), **o) if o else TrainConfig()


def _dev(s: str) -> torch.device:
    s = s.strip().lower()
    if s in ("gpu", "cuda"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def main(argv=None):
    args = _parse(argv)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = _cfg(args)
    train_cfg = _train_cfg(args)
    dev = _dev(args.device)
    data = _load_data(args, cfg, train_cfg, dev)

    model, seed, prior = build_model(
        train_cfg.node_feat_dim, train_cfg.cond_dim, cfg.z_dim, data.X_star.num_nodes(),
        edge_T=cfg.edge_T, prior_hidden=PRIOR_HIDDEN,
    )
    pe = PocketEncoder(train_cfg.cond_dim, radius=args.pocket_r, hidden_nf=POCKET_HIDDEN, n_layers=POCKET_LAYERS)
    te = TaskEncoder(train_cfg.cond_dim, train_cfg.cond_dim)
    for m in (model, seed, prior, pe, te):
        m.to(dev)

    task = torch.from_numpy(task_vec(data.X_star, train_cfg.cond_dim)).to(dev, dtype=torch.float32)
    if args.mode == "crossdock":
        crd = torch.from_numpy(data.coords).to(dev, dtype=torch.float32)
        cond = lambda: (pe(crd), te(task))
    else:
        cond = lambda: (pe(torch.randn(train_cfg.test_pts, 3, device=dev)), te(task))

    opt = torch.optim.AdamW(_uniq_params(model, seed, prior, pe, te), lr=args.lr)
    kabsch = args.kabsch or args.coord_loss == "kabsch"

    for ep in range(args.epochs):
        m = train(
            model, opt, data.traj, cfg=cfg, dev=dev, cond_fn=cond, seed=seed, prior=prior,
            pool=data.pool, bs=train_cfg.batch_size, ep=ep, rng_seed=args.seed,
            pkt_ctx=data.pkt_ctx, pocket_tgt=data.pocket_tgt, crd_loss=args.coord_loss,
            kabsch=kabsch, fp_bptt=train_cfg.fp_bptt, x_star=data.X_star,
        )
        print(f"ep={ep} loss={m.get('L', 0.0):.4f}")

    meta = {
        **dataclasses.asdict(cfg),
        **dataclasses.asdict(train_cfg),
        "mode": args.mode, "complex_id": data.cid, "note": data.note,
        "num_nodes": data.X_star.num_nodes(),
        "hidden": HIDDEN, "n_layers": N_LAYERS, "pocket_hidden": POCKET_HIDDEN, "pocket_layers": POCKET_LAYERS,
        "prior_hidden": PRIOR_HIDDEN, "epochs": args.epochs, "lr": args.lr,
        "num_pairs": args.num_pairs, "pocket10_root": str(args.pocket10_root.resolve()) if args.mode == "crossdock" else None,
        "receptor_pdb": data.pdb, "mmp_db": str(args.mmp_db.resolve()) if args.mmp_db else str(_ROOT / "dataset" / "mmp.db"),
        "seed_global": args.seed, "coord_loss": args.coord_loss, "kabsch": kabsch,
        "pocket_r": args.pocket_r, "ps_tgt": data.pocket_tgt,
    }
    _save_ckpt((args.output_dir / args.checkpoint).resolve(), model=model, seed=seed, prior=prior, pe=pe, te=te, opt=opt, cfg=cfg, train_cfg=train_cfg, meta=meta)


if __name__ == "__main__":
    main()
