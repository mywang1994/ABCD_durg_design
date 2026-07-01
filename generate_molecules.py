from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from config import ABCMConfig, PRIOR_HIDDEN
from dataset.protein_data import CrossDockComplex, discover_crossdock_pocket10, pocket_coords, task_vec
from dataset.state_pool import build_crossdock_bundle, build_state_pool_test
from dataset.pool_core import PoolCfg, load_reference_mol_from_sdf, mol_to_molstate
from mol_state import MolState
from model import PocketEncoder, TaskEncoder, build_model
from model.anchor import top_anchors
from model.consistency_operator import Route, infer_grp_routes, infer_routes, rollout
from utils.distances import molstate_to_mol, rmsd_aligned_same_index, tanimoto_molstate
from utils.pkt_fit import PktWt, pkt_fit


def _clamp_basin(n):
    return max(0, min(20, int(n)))


def _cfg_from_ckpt(d: dict) -> ABCMConfig:
    if not d:
        return ABCMConfig()
    ok = {f.name for f in dataclasses.fields(ABCMConfig)}
    return ABCMConfig(**{k: v for k, v in d.items() if k in ok})


def load_ckpt(path, dev):
    try:
        ck = torch.load(path, map_location=dev, weights_only=False)
    except TypeError:
        ck = torch.load(path, map_location=dev)
    meta = ck.get("meta", {})
    cfg = _cfg_from_ckpt(ck.get("abcm_config", {}))

    n_feat = int(meta.get("node_feat_dim", 16))
    c_dim = int(meta.get("cond_dim", 32))
    z_dim = int(meta.get("z_dim", 32))
    n_nodes = int(meta.get("num_nodes", 12))
    edge_T = int(meta.get("edge_T", meta.get("edge_diffusion_steps", 20)))

    core_kw: dict = {}
    h = meta.get("hidden", meta.get("graph_geom_hidden_dim"))
    nl = meta.get("n_layers", meta.get("graph_geom_num_layers"))
    if h is not None:
        core_kw["hidden"] = int(h)
    if nl is not None:
        core_kw["num_layers"] = int(nl)

    model, seed, prior = build_model(
        node_feat_dim=n_feat,
        cond_dim=c_dim,
        z_dim=z_dim,
        num_nodes=n_nodes,
        shared_core=True,
        core_kwargs=core_kw or None,
        edge_T=edge_T,
        prior_hidden=int(meta.get("prior_hidden", meta.get("prior_mlp_hidden_dim", PRIOR_HIDDEN))),
    )
    model.load_state_dict(ck["model"])
    model.to(dev)
    if "seed" in ck:
        seed.load_state_dict(ck["seed"])
    if "prior" in ck:
        prior.load_state_dict(ck["prior"])
    seed.to(dev)
    prior.to(dev)

    pe = PocketEncoder(
        out_dim=c_dim,
        radius=float(meta.get("pocket_r", meta.get("pocket_encoder_radius", 5.0))),
        hidden_nf=int(meta.get("pocket_hidden", meta.get("pocket_egnn_hidden_nf", 256))),
        n_layers=int(meta.get("pocket_layers", meta.get("pocket_egnn_num_layers", 6))),
    )
    te = TaskEncoder(in_dim=c_dim, out_dim=c_dim)
    if "pocket_encoder" in ck:
        pe.load_state_dict(ck["pocket_encoder"])
    if "task_encoder" in ck:
        te.load_state_dict(ck["task_encoder"])
    pe.to(dev)
    te.to(dev)

    pkt_max = int(meta.get("pocket_max_atoms", 4096))
    return model, seed, prior, pe, te, cfg, meta, pkt_max


def _write_sdf(path, X):
    from rdkit import Chem

    path.parent.mkdir(parents=True, exist_ok=True)
    mol = molstate_to_mol(X)
    if mol is None:
        return False
    w = Chem.SDWriter(str(path))
    w.write(mol)
    w.close()
    return True


def mk_cond(pocket_pdb, *, n_feat, c_dim, pe, te, dev, pkt_max, seed, ref_sdf=None):
    xyz = pocket_coords(pocket_pdb, max_atoms=pkt_max, seed=seed)
    if xyz.shape[0] < 4:
        xyz = np.asarray(np.random.default_rng(seed).standard_normal((64, 3)), dtype=np.float32)
    if ref_sdf:
        ref = mol_to_molstate(load_reference_mol_from_sdf(ref_sdf), node_feat_dim=n_feat)
        tvec = task_vec(ref, c_dim)
    else:
        tvec = np.zeros(c_dim, dtype=np.float32)
    pe.eval()
    te.eval()
    with torch.no_grad():
        crd = torch.from_numpy(xyz).to(device=dev, dtype=torch.float32)
        t = torch.from_numpy(tvec).to(device=dev, dtype=torch.float32)
        return pe(crd), te(t)


def mk_pocket(pdb: Path, ref_sdf: Path | None) -> dict:
    pkt = {"receptor_pdb": pdb.resolve()}
    if ref_sdf is not None:
        pkt["reference_mol"] = load_reference_mol_from_sdf(ref_sdf)
    return pkt


def pkt_dict(X: MolState, pocket: dict) -> dict:
    s = pkt_fit(X, pocket)
    return {"spharm": float(s.spharm), "sshape": float(s.sshape), "sfit": float(s.sfit), "sclash": float(s.sclash), "svina": float(s.svina), "pkt_fit": float(s.tot)}


def pick_anchors(pdb, ref_sdf, *, n_anchors, max_sites, n_feat, dev, seed):
    if n_anchors <= 0:
        return None
    crd = pocket_coords(pdb, max_atoms=max_sites, seed=seed)
    if crd.shape[0] == 0:
        return None
    pkt = mk_pocket(pdb, ref_sdf)
    kw = {"wt": PktWt(), "clash_r": 3.4, "probe_z": 7, "feat_dim": n_feat}
    return top_anchors(crd, pkt, k=n_anchors, score_kw=kw).to(device=dev)


def _smi(X: MolState) -> str | None:
    mol = molstate_to_mol(X)
    if mol is None:
        return None
    try:
        from rdkit import Chem
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _vs_ref(X, X_ref, pkt=None, *, w_pkt=0.5):
    """Higher = better. ref sim + optional pkt fit."""
    t = tanimoto_molstate(X, X_ref)
    d = None
    try:
        d = float(X.distance_to(X_ref))
    except Exception:
        pass
    if t is not None:
        ref_sc = float(t)
    elif d is not None and np.isfinite(d):
        ref_sc = 1.0 / (1.0 + d)
    else:
        ref_sc = float("-inf")

    ps = pkt_dict(X, pkt)["pkt_fit"] if pkt is not None else None
    if ps is not None and np.isfinite(ref_sc):
        w = float(np.clip(w_pkt, 0.0, 1.0))
        return (1.0 - w) * ref_sc + w * ps, t, d, ps
    if ps is not None and not np.isfinite(ref_sc):
        return ps, t, d, ps
    return ref_sc, t, d, ps


def basin_from_routes(routes, X_star, bmax, pkt=None, w_pkt=0.5):
    bmax = _clamp_basin(bmax)
    if bmax <= 0 or not routes:
        return [], [], [], [], []

    seen: set[str] = set()
    rows: list[tuple[float, MolState, float | None, float | None, dict]] = []
    for rt in routes:
        for step, X in enumerate(rt.trajectory[:-1]):
            smi = _smi(X)
            if smi is not None:
                if smi in seen:
                    continue
                seen.add(smi)
            key, t, d, ps = _vs_ref(X, X_star, pkt, w_pkt=w_pkt)
            if not np.isfinite(key):
                continue
            extra = {"route_init_index": rt.init_index, "trajectory_step": step}
            if pkt is not None:
                extra.update(pkt_dict(X, pkt))
            rows.append((key, X, t, d, extra))

    if not rows:
        return [], [], [], [], []
    rows.sort(key=lambda r: r[0], reverse=True)
    top = rows[:bmax]
    return (
        [r[1] for r in top],
        [float(r[0]) for r in top],
        [r[2] for r in top],
        [r[3] for r in top],
        [r[4] for r in top],
    )


def basin_from_traj(traj, bmax, ref=None):
    bmax = _clamp_basin(bmax)
    if bmax <= 0 or len(traj) < 2:
        return [], [], [], [], []

    anchor = ref if ref is not None else traj[-1]
    rows = []
    for step, X in enumerate(traj[:-1]):
        key, t, d, _ = _vs_ref(X, anchor)
        rmsd = None
        last = traj[-1]
        if X.num_nodes() == last.num_nodes() and X.num_nodes() > 0:
            rv = rmsd_aligned_same_index(X, last)
            if np.isfinite(rv):
                rmsd = float(rv)
        if not np.isfinite(key) and t is None and rmsd is None:
            continue
        if not np.isfinite(key):
            key = float(t) if t is not None else (1.0 / (1.0 + rmsd) if rmsd is not None else float("-inf"))
        rows.append((key, X, t, rmsd, step))

    if not rows:
        return [], [], [], [], []
    rows.sort(key=lambda r: r[0], reverse=True)
    top = rows[:bmax]
    return (
        [r[1] for r in top],
        [float(r[0]) for r in top],
        [r[2] for r in top],
        [r[3] for r in top],
        [{"trajectory_step": r[4]} for r in top],
    )


def explore_pool(model, pool, X_star, hP, hc, cfg, n_routes, bmax, seed, pkt=None, w_pkt=0.5, *, use_grp=None, soft_grp=True):
    grp = bool(getattr(cfg, "group_train", False)) if use_grp is None else bool(use_grp)
    kw = dict(model=model, pool=pool, hP=hP, hc=hc, cfg=cfg, num_starts=n_routes, X_star=X_star, seed=seed, early_stop=True)
    routes = infer_grp_routes(**kw, use_trainable_infer=soft_grp) if grp else infer_routes(**kw)

    def _rank(rt: Route) -> float:
        if pkt is not None:
            _, _, _, ps = _vs_ref(rt.final, X_star, pkt, w_pkt=1.0)
            if ps is not None and np.isfinite(ps):
                return float(ps)
        return -float(rt.final.distance_to(X_star))

    routes.sort(key=_rank, reverse=True)
    basin, sc, tani, dist, extra = basin_from_routes(routes, X_star, bmax, pkt=pkt, w_pkt=w_pkt)
    return routes, basin, sc, tani, dist, extra


def mk_infer_pool(pdb, ref_sdf, *, seed=0, mmp_db=None, pocket_min=None):
    cx = CrossDockComplex(complex_id="infer", receptor_pdb=pdb.resolve(), reference_ligand_sdf=ref_sdf.resolve())
    pc = PoolCfg(seed=seed, pocket_min=pocket_min)
    if mmp_db is not None:
        pc = dataclasses.replace(pc, mmp_db=mmp_db.resolve())
    built = build_crossdock_bundle(cx, pc)
    pool = built.all_molstates()
    if not pool:
        ref_mol = load_reference_mol_from_sdf(ref_sdf)
        x_star = mol_to_molstate(ref_mol)
        return build_state_pool_test.build_pool(x_star, pool_size=64), x_star
    return pool, built.X_star


def gen_seed(model, seed, prior, hP, hc, cfg, bmax, run_seed, anchors=None):
    model.eval()
    seed.eval()
    prior.eval()
    torch.manual_seed(run_seed)
    with torch.no_grad():
        z = prior.sample(hP, hc, num_samples=1)[0]
        X0 = seed.forward(z, anchors, hP, hc)
    final, traj = rollout(model, X0, hP, hc, cfg, early_stop=False)
    basin, sc, tani, rmsd, _ = basin_from_traj(traj, bmax=bmax)
    return final, basin, sc, tani, rmsd, traj


def _dump_basin(out_dir: Path, mols, prefix="rank"):
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for j, X in enumerate(mols):
        p = out_dir / f"{prefix}_{j:02d}.sdf"
        if _write_sdf(p, X):
            names.append(p.name)
    return names


def _dump_routes(out_root, routes, basin, sc, tani, dist, extra, X_star, pkt=None):
    rdir = out_root / "routes"
    rdir.mkdir(parents=True, exist_ok=True)
    rjson = []
    for i, rt in enumerate(routes):
        tag = f"route_{i:03d}"
        p_init = rdir / f"{tag}_init.sdf"
        p_fin = rdir / f"{tag}_final.sdf"
        _write_sdf(p_init, rt.X_init)
        _write_sdf(p_fin, rt.final)
        try:
            d_fin = float(rt.final.distance_to(X_star))
        except Exception:
            d_fin = None
        row = {
            "id": tag,
            "pool_init_index": rt.init_index,
            "init_sdf": str(p_init.relative_to(out_root).as_posix()),
            "final_sdf": str(p_fin.relative_to(out_root).as_posix()),
            "trajectory_len": len(rt.trajectory),
            "d_to_reference": d_fin,
            "group_size": int(getattr(rt, "group_size", 0)),
        }
        if getattr(rt, "group_states", ()):
            gdir = rdir / f"{tag}_group"
            gdir.mkdir(parents=True, exist_ok=True)
            g_sdfs = []
            for j, gX in enumerate(rt.group_states):
                gp = gdir / f"member_{j:02d}.sdf"
                if _write_sdf(gp, gX):
                    g_sdfs.append(str(gp.relative_to(out_root).as_posix()))
            row["group_sdfs"] = g_sdfs
        if pkt is not None:
            row.update(pkt_dict(rt.final, pkt))
            row["init_pkt_fit"] = pkt_dict(rt.X_init, pkt)["pkt_fit"]
        rjson.append(row)

    bdir = out_root / "basin"
    bfiles = _dump_basin(bdir, basin)
    best = routes[0] if routes else None
    p_best = out_root / "best_final.sdf"
    if best is not None:
        _write_sdf(p_best, best.final)

    out = {
        "best_final_sdf": str(p_best.name) if best is not None else None,
        "num_routes": len(routes),
        "global_basin_dir": str(bdir.relative_to(out_root).as_posix()),
        "global_basin_count": len(basin),
        "global_basin_sdfs": bfiles,
        "global_basin_scores": sc,
        "global_basin_tanimoto_to_ref": tani,
        "global_basin_d_to_reference": dist,
        "global_basin_meta": extra,
        "routes": rjson,
    }
    if best is not None and pkt is not None:
        out["best_pkt_fit"] = pkt_dict(best.final, pkt)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate from checkpoint + pocket PDB (seed / pool / both).")
    p.add_argument("--pocket-pdb", type=Path, required=True)
    p.add_argument("--num-generate", type=int, required=True, help="Seed samples or pool routes.")
    p.add_argument("--init-mode", choices=("seed", "pool", "both"), default="pool")
    p.add_argument("--output-dir", type=Path, default=_ROOT / "output")
    p.add_argument("--basin-max", type=int, default=20)
    p.add_argument("--checkpoint", type=Path, default=_ROOT / "output" / "abcm_checkpoint.pt")
    p.add_argument("--reference-sdf", type=Path, default=None)
    p.add_argument("--mmp-db", type=Path, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-anchors", type=int, default=4)
    p.add_argument("--anchor-sites", type=int, default=128)
    p.add_argument("--pocket-min", type=float, default=None)
    p.add_argument("--basin-pocket-weight", type=float, default=0.5)
    p.add_argument("--group-inference", action="store_true")
    p.add_argument("--no-group-inference", action="store_true")
    p.add_argument("--group-infer-discrete", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dev = torch.device(args.device)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    pdb = args.pocket_pdb.resolve()
    ref_sdf = args.reference_sdf.resolve() if args.reference_sdf else None

    if not args.checkpoint.is_file():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)
    if not pdb.is_file():
        print(f"Pocket PDB not found: {pdb}", file=sys.stderr)
        sys.exit(1)
    if args.init_mode in ("pool", "both") and ref_sdf is None:
        print("--reference-sdf is required for --init-mode pool or both.", file=sys.stderr)
        sys.exit(1)
    if ref_sdf is not None and not ref_sdf.is_file():
        print(f"Reference SDF not found: {ref_sdf}", file=sys.stderr)
        sys.exit(1)

    model, seed, prior, pe, te, cfg, meta, pkt_max = load_ckpt(args.checkpoint, dev)
    n_feat = int(meta.get("node_feat_dim", 16))
    c_dim = int(meta.get("cond_dim", 32))

    hP, hc = mk_cond(pdb, n_feat=n_feat, c_dim=c_dim, pe=pe, te=te, dev=dev, pkt_max=pkt_max, seed=args.seed, ref_sdf=ref_sdf)
    pkt = mk_pocket(pdb, ref_sdf)
    anchors = pick_anchors(pdb, ref_sdf, n_anchors=int(args.num_anchors), max_sites=int(args.anchor_sites), n_feat=n_feat, dev=dev, seed=args.seed)
    if anchors is not None:
        print(f"[anchor] picked {anchors.shape[0]} sites")

    pool: list[MolState] = []
    X_star: MolState | None = None
    if args.init_mode in ("pool", "both"):
        pool, X_star = mk_infer_pool(pdb, ref_sdf, seed=args.seed, mmp_db=args.mmp_db, pocket_min=args.pocket_min)
        print(f"[pool] size={len(pool)}")

    bmax = _clamp_basin(args.basin_max)
    use_grp = bool(getattr(cfg, "group_train", False))
    if args.group_inference:
        use_grp = True
    if args.no_group_inference:
        use_grp = False
    soft_grp = not bool(args.group_infer_discrete)

    man: dict = {
        "pocket_pdb": str(pdb),
        "checkpoint": str(args.checkpoint.resolve()),
        "init_mode": args.init_mode,
        "num_generate": args.num_generate,
        "basin_max": bmax,
        "reference_sdf": str(ref_sdf) if ref_sdf else None,
        "pool_size": len(pool),
        "num_anchors": int(args.num_anchors),
        "pocket_min": args.pocket_min,
        "basin_pocket_weight": float(args.basin_pocket_weight),
        "group_inference": use_grp,
        "group_infer_trainable": soft_grp,
    }

    if args.init_mode in ("pool", "both") and X_star is not None:
        n_rt = int(args.num_generate) if args.init_mode == "pool" else max(1, int(args.num_generate) // 2)
        routes, basin, sc, tani, dist, extra = explore_pool(
            model, pool, X_star, hP, hc, cfg,
            n_routes=n_rt, bmax=bmax, seed=args.seed,
            pkt=pkt, w_pkt=float(args.basin_pocket_weight),
            use_grp=use_grp, soft_grp=soft_grp,
        )
        tag = "group" if use_grp else "single"
        print(f"[pool/{tag}] routes={len(routes)} basin={len(basin)}")
        man["pool_exploration"] = _dump_routes(out, routes, basin, sc, tani, dist, extra, X_star, pkt=pkt)

    seed_rows: list[dict] = []
    n_seed = int(args.num_generate) if args.init_mode == "seed" else (
        0 if args.init_mode == "pool" else max(1, int(args.num_generate) - max(1, int(args.num_generate) // 2))
    )

    for i in range(n_seed):
        run_seed = args.seed + 1000 * i
        final, basin, sc, tani, rmsd, traj = gen_seed(
            model, seed, prior, hP, hc, cfg, bmax, run_seed, anchors=anchors,
        )
        tag = f"seed_{i:04d}"
        p_fin = out / f"{tag}_final.sdf"
        ok = _write_sdf(p_fin, final)
        row = {
            "id": tag,
            "init_mode": "seed",
            "final_sdf": str(p_fin.name) if ok else None,
            "trajectory_len": len(traj),
            "basin_count": len(basin),
            "basin_sort_keys": sc,
            "basin_tanimoto": tani,
            "basin_rmsd_to_final": rmsd,
        }
        row.update(pkt_dict(final, pkt))
        if X_star is not None:
            try:
                row["d_to_reference"] = float(final.distance_to(X_star))
            except Exception:
                row["d_to_reference"] = None
        row["basin_sdfs"] = _dump_basin(out / f"{tag}_basin", basin) if basin else []
        seed_rows.append(row)

    if seed_rows:
        man["seed_runs"] = seed_rows

    man_path = out / "generate_manifest.json"
    with man_path.open("w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, ensure_ascii=False)
    print("Done")


if __name__ == "__main__":
    main()
