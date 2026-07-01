from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, SDWriter
from vina import Vina

from .distances import molstate_to_mol
from mol_state import MolState

# pocket fit score
# pharmacophore channels + soft Gaussian with element VdW radii.
# optional vina dock 

POCKET_TRUNC_A = 10.0

_PHARM_CH = {
    "Hydrophobe": 0,
    "LumpedHydrophobe": 0,
    "Aromatic": 1,
    "Acceptor": 2,
    "Donor": 3,
}

_VDW = {
    "H": 1.2, "C": 1.7, "N": 1.55, "O": 1.52, "F": 1.47, "P": 1.8, "S": 1.8,
    "CL": 2.27, "BR": 1.85, "I": 1.98, "SE": 1.9, "SI": 2.1,
}


@dataclass(frozen=True)
class PktWt:
    wp: float = 0.35
    ws: float = 0.40
    wf: float = 0.35
    wc: float = 0.35
    wv: float = 0.0   # vina dock (0 = off unless dock backend set)
    clash_k: float = 50.0


@dataclass(frozen=True)
class VinaDockCfg:
    """Optional AutoDock Vina scoring/docking (pip vina)."""

    redock: bool = False          # False: score pose; True: run dock
    box_pad: float = 4.0
    box_size: tuple[float, float, float] | None = None
    exhaustiveness: int = 8
    n_poses: int = 1
    obabel: str = "obabel"
    aff_min: float = -12.0        # kcal/mol -> [0, 1]
    aff_max: float = 0.0


@dataclass(frozen=True)
class PktFit:
    spharm: float
    sshape: float
    sfit: float
    sclash: float
    tot: float
    svina: float = 0.0


def _sym(z: int) -> str:
    try:
        return Chem.Atom(int(z)).GetSymbol().upper()
    except Exception:
        return "C"


def _radii_from_graph(G, n: int) -> np.ndarray:
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        z = 6
        if G is not None and G.has_node(i):
            z = int(G.nodes[i].get("atomic_num", 6))
        out[i] = _VDW.get(_sym(z), 1.7)
    return out


def _read_pdb_heavy(pdb_path: Path) -> tuple[np.ndarray, np.ndarray]:
    pts: list[list[float]] = []
    rad: list[float] = []
    with open(pdb_path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
                continue
            if len(ln) < 54:
                continue
            el = (ln[76:78].strip() or ln[12:14].strip()[:1] or "C").upper()
            if el in ("H", "D", ""):
                continue
            try:
                x, y, z = float(ln[30:38]), float(ln[38:46]), float(ln[46:54])
            except ValueError:
                continue
            pts.append([x, y, z])
            rad.append(_VDW.get(el[:2] if el.startswith("CL") else el[:1], 1.7))
    if not pts:
        z = np.zeros((0, 3), dtype=np.float64)
        return z, np.zeros(0, dtype=np.float64)
    return np.asarray(pts, dtype=np.float64), np.asarray(rad, dtype=np.float64)


def _read_pdb_pts(pdb_path: Path) -> np.ndarray:
    crd, _ = _read_pdb_heavy(pdb_path)
    return crd


def _truncate_rec(rec: np.ndarray, rec_r: np.ndarray, center: np.ndarray, radius: float):
    if rec.size == 0:
        return rec, rec_r
    d = np.linalg.norm(rec - center.reshape(1, 3), axis=1)
    m = d <= float(radius)
    return rec[m], rec_r[m] if rec_r.size else rec_r


def _clash_vdw(lig, lig_r, rec, rec_r) -> float:
    if lig.size == 0 or rec.size == 0:
        return 0.0
    dist = np.linalg.norm(lig[:, None, :] - rec[None, :, :], axis=-1)
    rl = lig_r[:, None]
    rr = rec_r[None, :]
    bump = np.maximum(0.0, rl + rr - dist)
    return float((bump * bump).sum())


def _fit_gauss(lig, lig_r, rec, rec_r) -> float:
    if lig.size == 0 or rec.size == 0:
        return 0.0
    dist = np.linalg.norm(lig[:, None, :] - rec[None, :, :], axis=-1)
    rl = lig_r[:, None]
    rr = rec_r[None, :]
    sig = np.maximum(rl + rr, 0.5)
    ov = np.exp(-0.5 * (dist / sig) ** 2)
    return float(np.clip(ov.max(axis=1).mean(), 0.0, 1.0))


def _feat_factory():
    fdef = str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
    return ChemicalFeatures.BuildFeatureFactory(fdef)


def _feat_centers(mol: Chem.Mol, fac: Any) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    mol.UpdatePropertyCache(False)
    if mol.GetNumConformers() < 1:
        return out
    try:
        feats = fac.GetFeaturesForMol(mol)
    except Exception:
        return out
    conf = mol.GetConformer()
    for ft in feats:
        fam = ft.GetFamily()
        ids = ft.GetAtomIds()
        if not ids:
            continue
        ctr = np.zeros(3, dtype=np.float64)
        for i in ids:
            p = conf.GetAtomPosition(int(i))
            ctr += np.array([p.x, p.y, p.z], dtype=np.float64)
        ctr /= max(len(ids), 1)
        out.append((fam, ctr))
    return out


def _pharm_by_channel(mol: Chem.Mol, fac) -> list[list[np.ndarray]]:
    ch: list[list[np.ndarray]] = [[] for _ in range(4)]
    for fam, ctr in _feat_centers(mol, fac):
        cid = _PHARM_CH.get(fam)
        if cid is not None:
            ch[cid].append(ctr)
    return ch


def _pharm_match(ref_m, prb_m, fac, sig: float = 1.5) -> float:
    ref_ch = _pharm_by_channel(ref_m, fac)
    prb_ch = _pharm_by_channel(prb_m, fac)
    if not any(ref_ch) and not any(prb_ch):
        return 0.5
    scores: list[float] = []
    for r_pts, p_pts in zip(ref_ch, prb_ch):
        if not r_pts and not p_pts:
            continue
        if not r_pts or not p_pts:
            scores.append(0.0)
            continue
        R = np.stack(r_pts, axis=0)
        P = np.stack(p_pts, axis=0)
        dist = np.linalg.norm(R[:, None, :] - P[None, :, :], axis=-1)
        ov = np.exp(-dist / max(sig, 1e-6))
        scores.append(float(np.clip(ov.max(axis=1).mean(), 0.0, 1.0)))
    if not scores:
        return 0.0
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def _shape_tani(ref_m, prb_m) -> float:
    try:
        from rdkit.Chem import rdShapeHelpers
    except ImportError:
        return 0.0
    if ref_m.GetNumConformers() < 1 or prb_m.GetNumConformers() < 1:
        return 0.0
    try:
        dd = float(rdShapeHelpers.ShapeTanimotoDist(ref_m, prb_m, 0, 0))
    except Exception:
        return 0.0
    dd = max(0.0, min(1.0, dd))
    return float(1.0 - dd)


def _rec_from_pocket(pocket: dict, center: np.ndarray, trunc: float):
    rec = pocket.get("receptor_coords")
    rec_r = pocket.get("receptor_radii")
    if rec is None:
        pdb = pocket.get("receptor_pdb")
        if pdb is not None:
            rec, rec_r = _read_pdb_heavy(Path(pdb))
        else:
            rec = np.zeros((0, 3), dtype=np.float64)
            rec_r = np.zeros(0, dtype=np.float64)
    else:
        rec = np.asarray(rec, dtype=np.float64)
        if rec_r is not None:
            rec_r = np.asarray(rec_r, dtype=np.float64)
        else:
            rec_r = np.full(rec.shape[0], 1.7, dtype=np.float64)
    return _truncate_rec(rec, rec_r, center, trunc)


# --- vina dock (optional, pip vina) ---

_REC_CACHE_DIR = Path(tempfile.gettempdir()) / "abcm_pkt_rec_pdbqt"


def _resolve_dock_cfg(
    dock: VinaDockCfg | bool | None,
    pocket: dict,
) -> VinaDockCfg | None:
    if dock is None:
        dock = pocket.get("vina_dock")
    if dock is None or dock is False:
        return None
    if dock is True:
        return VinaDockCfg()
    return dock


def _run_obabel(cfg: VinaDockCfg, *args: str) -> None:
    if not shutil.which(cfg.obabel):
        raise RuntimeError(f"obabel not found: {cfg.obabel!r}")
    subprocess.run([cfg.obabel, *args], check=True, capture_output=True, text=True)


def _write_mol_sdf(mol: Chem.Mol, path: Path) -> None:
    w = SDWriter(str(path))
    w.write(mol)
    w.close()


def _box_from_pocket(pocket: dict, cfg: VinaDockCfg) -> tuple[list[float], list[float]]:
    ref_m = pocket.get("reference_mol")
    if ref_m is None:
        raise ValueError("vina dock needs reference_mol in pocket for search box")
    rm = Chem.Mol(ref_m)
    if rm.GetNumConformers() < 1:
        raise ValueError("reference_mol has no conformer")
    pos = np.asarray(rm.GetConformer().GetPositions(), dtype=np.float64)
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    center = (0.5 * (lo + hi)).tolist()
    if cfg.box_size is not None:
        size = [float(x) for x in cfg.box_size]
    else:
        size = np.maximum(hi - lo + float(cfg.box_pad), 12.0).tolist()
    return center, size


def _aff_to_score(aff: float, cfg: VinaDockCfg) -> float:
    s = (float(cfg.aff_max) - float(aff)) / max(float(cfg.aff_max) - float(cfg.aff_min), 1e-6)
    return float(np.clip(s, 0.0, 1.0))


def _receptor_pdbqt(receptor_pdb: Path, cfg: VinaDockCfg) -> Path:
    _REC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rec = receptor_pdb.resolve()
    out = _REC_CACHE_DIR / f"{rec.stem}.pdbqt"
    if out.is_file() and out.stat().st_mtime >= rec.stat().st_mtime:
        return out
    _run_obabel(cfg, "-ipdb", str(rec), "-opdbqt", "-O", str(out), "-xr")
    return out


def _ligand_pdbqt(mol: Chem.Mol, cfg: VinaDockCfg, work: Path) -> Path:
    sdf = work / "ligand.sdf"
    pdbqt = work / "ligand.pdbqt"
    _write_mol_sdf(mol, sdf)
    _run_obabel(cfg, "-isdf", str(sdf), "-opdbqt", "-O", str(pdbqt))
    return pdbqt


def _vina_affinity(
    rec_pdbqt: Path,
    lig_pdbqt: Path,
    center: list[float],
    box_size: list[float],
    cfg: VinaDockCfg,
) -> float:
    v = Vina(sf_name="vina", verbosity=0)
    v.set_receptor(str(rec_pdbqt))
    v.set_ligand_from_file(str(lig_pdbqt))
    v.compute_vina_maps(center=center, box_size=box_size)
    if cfg.redock:
        v.dock(exhaustiveness=int(cfg.exhaustiveness), n_poses=int(cfg.n_poses))
        return float(v.energies(n_poses=1)[0][0])
    scored = v.score()
    if isinstance(scored, (list, tuple, np.ndarray)):
        return float(scored[0])
    return float(scored)


def vina_dock_fit(
    X: MolState,
    pocket: dict,
    *,
    cfg: VinaDockCfg | None = None,
) -> float:
    """Vina score/dock via pip vina. Returns unit score in [0, 1] (higher = better)."""
    dock = _resolve_dock_cfg(cfg, pocket)
    if dock is None:
        return 0.0

    receptor = pocket.get("receptor_pdb")
    if receptor is None:
        return 0.0
    receptor_pdb = Path(receptor)
    if not receptor_pdb.is_file():
        return 0.0

    mol = molstate_to_mol(X)
    if mol is None or mol.GetNumConformers() < 1:
        return 0.0

    try:
        center, box_size = _box_from_pocket(pocket, dock)
        with tempfile.TemporaryDirectory(prefix="pkt_vina_") as td:
            work = Path(td)
            rec_pdbqt = _receptor_pdbqt(receptor_pdb, dock)
            lig_pdbqt = _ligand_pdbqt(mol, dock, work)
            aff = _vina_affinity(rec_pdbqt, lig_pdbqt, center, box_size, dock)
        return _aff_to_score(aff, dock)
    except Exception:
        return 0.0


def pkt_fit(
    X: MolState,
    pocket: dict,
    *,
    wt: PktWt | None = None,
    clash_r: float = 3.4,
    pocket_radius: float = POCKET_TRUNC_A,
    dock: VinaDockCfg | str | None = None,
) -> PktFit:
    w = wt or PktWt()
    lig = np.asarray(X.R, dtype=np.float64)
    n = lig.shape[0]
    lig_r = _radii_from_graph(X.G, n)
    center = lig.mean(axis=0) if n > 0 else np.zeros(3, dtype=np.float64)

    ref_m = pocket.get("reference_mol")
    if ref_m is not None:
        ref_m = Chem.Mol(ref_m)
        try:
            Chem.SanitizeMol(ref_m)
        except Exception:
            pass
        if ref_m.GetNumConformers() > 0:
            center = np.asarray(ref_m.GetConformer().GetPositions(), dtype=np.float64).mean(axis=0)

    prb_m = molstate_to_mol(X)
    s_shape = 0.0
    s_pharm = 0.0
    if ref_m is not None and prb_m is not None and ref_m.GetNumConformers() > 0 and prb_m.GetNumConformers() > 0:
        s_shape = _shape_tani(ref_m, prb_m)
        try:
            s_pharm = _pharm_match(ref_m, prb_m, _feat_factory())
        except Exception:
            s_pharm = 0.0

    rec, rec_r = _rec_from_pocket(pocket, center, pocket_radius)
    if rec_r.size == 0 and rec.size > 0:
        rec_r = np.full(rec.shape[0], float(clash_r) * 0.5, dtype=np.float64)

    s_fit = _fit_gauss(lig, lig_r, rec, rec_r)
    clash_raw = _clash_vdw(lig, lig_r, rec, rec_r)
    s_clash = float(min(1.0, clash_raw / max(w.clash_k, 1e-6)))

    dock_cfg = _resolve_dock_cfg(dock, pocket)
    s_vina = 0.0
    if dock_cfg is not None:
        s_vina = vina_dock_fit(X, pocket, cfg=dock_cfg)
    wv = w.wv if dock_cfg is not None else 0.0

    tot = w.wp * s_pharm + w.ws * s_shape + w.wf * s_fit - w.wc * s_clash + wv * s_vina
    return PktFit(spharm=s_pharm, sshape=s_shape, sfit=s_fit, sclash=s_clash, tot=tot, svina=s_vina)


def pkt_fit_stub(X: MolState, pocket: object) -> PktFit:
    if not isinstance(pocket, dict):
        pocket = {}
    return pkt_fit(X, pocket)


@dataclass(frozen=True)
class PktCtx:
    rec: torch.Tensor
    ref_lig: torch.Tensor
    pharm_pts: torch.Tensor
    rec_r: torch.Tensor
    lig_r: torch.Tensor
    pocket_radius: float = POCKET_TRUNC_A


@dataclass(frozen=True)
class PktFitT:
    spharm: torch.Tensor
    sshape: torch.Tensor
    sfit: torch.Tensor
    sclash: torch.Tensor
    tot: torch.Tensor


def _kabsch_rmsd(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    n = min(int(P.shape[0]), int(Q.shape[0]))
    if n == 0:
        return torch.tensor(0.0, device=P.device, dtype=P.dtype)
    P = P[:n]
    Q = Q[:n]
    P0 = P - P.mean(dim=0, keepdim=True)
    Q0 = Q - Q.mean(dim=0, keepdim=True)
    if n < 2:
        return torch.sqrt(torch.mean((P0 - Q0) ** 2) + 1e-8)
    H = P0.T @ Q0
    U, _, Vh = torch.linalg.svd(H)
    R = U @ Vh
    if torch.det(R) < 0:
        Vh = Vh.clone()
        Vh[-1, :] *= -1.0
        R = U @ Vh
    Pal = P0 @ R
    return torch.sqrt(torch.mean((Pal - Q0) ** 2) + 1e-8)


def clash_t(lig, rec, *, lig_r=None, rec_r=None, clash_r: float = 3.4, clash_k: float = 50.0) -> torch.Tensor:
    if lig.numel() == 0 or rec.numel() == 0:
        return torch.tensor(0.0, device=lig.device, dtype=lig.dtype)
    if lig_r is None:
        lig_r = torch.full((lig.shape[0],), float(clash_r) * 0.5, device=lig.device, dtype=lig.dtype)
    if rec_r is None:
        rec_r = torch.full((rec.shape[0],), float(clash_r) * 0.5, device=rec.device, dtype=rec.dtype)
    dist = torch.cdist(lig, rec)
    bump = torch.relu(lig_r[:, None] + rec_r[None, :] - dist)
    raw = (bump * bump).sum()
    return torch.clamp(raw / max(float(clash_k), 1e-6), max=1.0)


def fit_t(lig, rec, *, lig_r=None, rec_r=None) -> torch.Tensor:
    if lig.numel() == 0 or rec.numel() == 0:
        return torch.tensor(0.0, device=lig.device, dtype=lig.dtype)
    if lig_r is None:
        lig_r = torch.full((lig.shape[0],), 1.7, device=lig.device, dtype=lig.dtype)
    if rec_r is None:
        rec_r = torch.full((rec.shape[0],), 1.7, device=rec.device, dtype=rec.dtype)
    dist = torch.cdist(lig, rec)
    sig = torch.clamp(lig_r[:, None] + rec_r[None, :], min=0.5)
    ov = torch.exp(-0.5 * (dist / sig) ** 2)
    return ov.max(dim=1).values.mean()


def shape_t(R_pred, R_ref, *, sig: float = 2.0) -> torch.Tensor:
    rmsd = _kabsch_rmsd(R_pred, R_ref)
    return torch.exp(-rmsd / max(float(sig), 1e-6))


def pharm_t(lig, ctrs, *, sig: float = 1.5) -> torch.Tensor:
    if lig.numel() == 0 or ctrs.numel() == 0:
        return torch.tensor(0.0, device=lig.device, dtype=lig.dtype)
    dist = torch.cdist(lig, ctrs)
    sim = torch.exp(-dist / max(float(sig), 1e-6))
    return sim.max(dim=0).values.mean()


def pkt_fit_train(
    R_pred: torch.Tensor,
    ctx: PktCtx,
    *,
    wt: PktWt | None = None,
    clash_r: float = 3.4,
) -> PktFitT:
    w = wt or PktWt()
    center = ctx.ref_lig.mean(dim=0) if ctx.ref_lig.numel() else R_pred.mean(dim=0)
    rec = ctx.rec
    rec_r = ctx.rec_r
    if rec.numel() > 0 and float(ctx.pocket_radius) > 0:
        d = torch.norm(rec - center.unsqueeze(0), dim=1)
        m = d <= float(ctx.pocket_radius)
        rec = rec[m]
        rec_r = rec_r[m] if rec_r.numel() else rec_r

    s_clash = clash_t(R_pred, rec, lig_r=ctx.lig_r, rec_r=rec_r, clash_r=clash_r, clash_k=w.clash_k)
    s_shape = shape_t(R_pred, ctx.ref_lig)
    s_pharm = pharm_t(R_pred, ctx.pharm_pts)
    s_fit = fit_t(R_pred, rec, lig_r=ctx.lig_r, rec_r=rec_r)
    tot = w.wp * s_pharm + w.ws * s_shape + w.wf * s_fit - w.wc * s_clash
    return PktFitT(spharm=s_pharm, sshape=s_shape, sfit=s_fit, sclash=s_clash, tot=tot)


def mk_pkt_ctx(
    pocket: dict,
    X_star: MolState,
    *,
    device: torch.device,
    pocket_radius: float = POCKET_TRUNC_A,
) -> PktCtx | None:
    nn = X_star.num_nodes()
    ref_xyz = torch.from_numpy(np.asarray(X_star.R[:nn], dtype=np.float32)).to(device=device)
    center_np = np.asarray(X_star.R[:nn], dtype=np.float64).mean(axis=0)

    ref_m = pocket.get("reference_mol")
    if ref_m is not None:
        try:
            rm = Chem.Mol(ref_m)
            if rm.GetNumConformers() > 0:
                center_np = np.asarray(rm.GetConformer().GetPositions(), dtype=np.float64).mean(axis=0)
        except Exception:
            pass

    rec_np, rec_r_np = _rec_from_pocket(pocket, center_np, pocket_radius)
    if rec_np.size == 0:
        return None
    rec_t = torch.from_numpy(rec_np.astype(np.float32)).to(device=device)
    rec_r_t = torch.from_numpy(rec_r_np.astype(np.float32)).to(device=device)

    lig_r_np = _radii_from_graph(X_star.G, nn)
    lig_r_t = torch.from_numpy(lig_r_np.astype(np.float32)).to(device=device)

    ctr_list: list[np.ndarray] = []
    if ref_m is not None:
        try:
            rm = Chem.Mol(ref_m)
            for ch in _pharm_by_channel(rm, _feat_factory()):
                ctr_list.extend(ch)
        except Exception:
            ctr_list = []
    if ctr_list:
        pts_t = torch.from_numpy(np.stack(ctr_list, axis=0).astype(np.float32)).to(device=device)
    else:
        pts_t = torch.zeros((0, 3), device=device, dtype=torch.float32)

    return PktCtx(
        rec=rec_t,
        ref_lig=ref_xyz,
        pharm_pts=pts_t,
        rec_r=rec_r_t,
        lig_r=lig_r_t,
        pocket_radius=float(pocket_radius),
    )
