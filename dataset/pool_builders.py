from __future__ import annotations

import importlib
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures
from rdkit.Chem.Scaffolds import MurckoScaffold

from .pool_core import CrossDockState, LocalMMPConfig, ensure_conformer, mol_to_molstate, mol_to_smiles


# --- shared helpers ---

def _st(mol, kind: str, cid: str, note: str, *, conf_id: int = -1) -> CrossDockState:
    return CrossDockState(
        X=mol_to_molstate(mol, conf_id=conf_id),
        pool_kind=kind,
        complex_id=cid,
        source_note=note,
        smiles=mol_to_smiles(mol),
    )


def _jitter(mol, rng: random.Random, sigma: float, atoms: set[int] | None = None):
    m = Chem.Mol(mol)
    ensure_conformer(m)
    if m.GetNumConformers() == 0:
        return None
    conf_id = m.GetNumConformers() - 1
    conf = m.GetConformer(conf_id)
    idxs = atoms if atoms is not None else range(m.GetNumAtoms())
    for ai in idxs:
        p = conf.GetAtomPosition(ai)
        conf.SetAtomPosition(
            ai,
            Chem.rdGeometry.Point3D(
                p.x + rng.gauss(0, sigma),
                p.y + rng.gauss(0, sigma),
                p.z + rng.gauss(0, sigma),
            ),
        )
    return m, conf_id


# --- geo / scaffold ---

def build_geo(mol, cid, n, rng, sigma):
    if n <= 0:
        return []
    tpl = Chem.Mol(mol)
    ensure_conformer(tpl)
    if tpl.GetNumConformers() == 0:
        return []
    out: list[CrossDockState] = []
    for i in range(n):
        got = _jitter(tpl, rng, sigma)
        if got is None:
            break
        m, conf_id = got
        out.append(_st(m, "geo", cid, f"geo_noise#{i}", conf_id=conf_id))
    return out


def build_scaffold(mol, cid, n, rng):
    if n <= 0:
        return []
    murcko = MurckoScaffold.GetScaffoldForMol(Chem.Mol(mol))
    if murcko.GetNumAtoms() < 3:
        return []
    try:
        if murcko.GetNumConformers() == 0:
            AllChem.EmbedMolecule(murcko, randomSeed=rng.randint(0, 1_000_000))
        if murcko.GetNumConformers() > 0:
            AllChem.MMFFOptimizeMolecule(murcko)
    except Exception:
        return []
    if murcko.GetNumConformers() == 0:
        return []

    out = [_st(murcko, "scaffold", cid, "murcko_once")]
    for j in range(max(0, n - 1)):
        copy = Chem.Mol(murcko)
        try:
            AllChem.EmbedMolecule(copy, randomSeed=rng.randint(0, 1_000_000))
            AllChem.MMFFOptimizeMolecule(copy)
            out.append(_st(copy, "scaffold", cid, f"murcko_reembed#{j}"))
        except Exception:
            continue
    return out[:n]


# --- retro rxn ---

RXN_TPL: dict[str, dict[str, Any]] = {
    "amidation": {
        "name": "Amidation Reaction",
        "smarts": "[C:1](=[O:2])-[OH:3].[N:4]>>[C:1](=[O:2])-[N:4]",
        "fragments": {"acid": {"atoms": [1, 2, 3], "type": "carboxylic_acid"}, "amine": {"atoms": [4], "type": "amine"}},
        "bonds": [(1, 4)],
        "angle_constraints": None,
    },
    "friedel_crafts": {
        "name": "Friedel-Crafts Alkylation",
        "smarts": "[c:1].[C:2]-[Cl:3]>>[c:1]-[C:2]",
        "fragments": {"aromatic": {"atoms": [1], "type": "aromatic"}, "alkyl": {"atoms": [2, 3], "type": "alkyl_halide"}},
        "bonds": [(1, 2)],
        "angle_constraints": None,
    },
    "click_chemistry": {
        "name": "Click Chemistry (Azide-Alkyne Cycloaddition)",
        "smarts": "[N:1]=[N:2]=[N:3].[C:4]#[C:5]>>[N:1]1-[N:2]=[N:3]-[C:4]=[C:5]-1",
        "fragments": {"azide": {"atoms": [1, 2, 3], "type": "azide"}, "alkyne": {"atoms": [4, 5], "type": "alkyne"}},
        "bonds": [(1, 4), (3, 5)],
        "angle_constraints": {"type": "ring_closure", "atoms": [1, 4, 5, 3], "target_angle": 90.0, "weight": 0.5},
    },
}


def _retro_smarts(fwd: str) -> str:
    if ">>" not in fwd:
        return fwd
    left, right = fwd.split(">>", 1)
    return f"{right.strip()}>>{left.strip()}"


def _retro_run(mol, smarts: str, *, cap: int) -> list[Chem.Mol]:
    rxn = AllChem.ReactionFromSmarts(smarts)
    if rxn is None:
        return []
    m = Chem.Mol(mol)
    if m is None:
        return []
    try:
        outs = rxn.RunReactants((m,))
    except (ValueError, RuntimeError):
        return []
    products: list[Chem.Mol] = []
    for i, tpl in enumerate(outs):
        if i >= cap:
            break
        for pm in tpl:
            if pm is None:
                continue
            try:
                Chem.SanitizeMol(pm)
                products.append(Chem.Mol(pm))
            except Exception:
                continue
    return products


def build_rxn(mol, *, cid: str, n: int, rng: random.Random, tpl: dict[str, dict[str, Any]] | None = None):
    if n <= 0:
        return []
    rxn_tpl = RXN_TPL if tpl is None else tpl
    order = [k for k, v in rxn_tpl.items() if v.get("smarts")]
    if not order:
        return []
    rng.shuffle(order)

    seen: set[str] = set()
    try:
        seen.add(mol_to_smiles(mol))
    except Exception:
        pass

    out: list[CrossDockState] = []
    limit = max(64, n * 32)
    for key in order:
        if len(out) >= n:
            break
        smarts = _retro_smarts(str(rxn_tpl[key]["smarts"]))
        products = _retro_run(mol, smarts, cap=limit)
        rng.shuffle(products)
        for pm in products:
            if len(out) >= n:
                break
            try:
                smi = mol_to_smiles(pm)
            except Exception:
                continue
            if smi in seen:
                continue
            seen.add(smi)
            out.append(_st(pm, "rxn", cid, f"rxn_retro:{key}"))
    return out


# --- local MMP / CReM ---

def _subst_mod(mmp: LocalMMPConfig) -> str | None:
    name = (mmp.substitution_module or "").strip()
    if not name:
        name = (os.environ.get("ABCM_FRAGMENT_SUBST_MODULE") or "").strip()
    return name or None


def _prepend_mmp_root(root) -> None:
    roots: list[Path] = []
    if root is not None:
        roots.append(Path(root))
    env = os.environ.get("ABCM_MMP_ENGINE_ROOT")
    if env:
        roots.append(Path(env))
    for p in roots:
        if p.is_dir():
            s = str(p.resolve())
            if s not in sys.path:
                sys.path.insert(0, s)


def _load_mutators(mmp: LocalMMPConfig):
    mod = _subst_mod(mmp)
    if not mod:
        return None, None
    _prepend_mmp_root(mmp.mmp_engine_root)
    try:
        pkg = importlib.import_module(mod)
        return pkg.mutate_mol, pkg.grow_mol
    except (ImportError, AttributeError):
        return None, None


def _parse_yield(item):
    if isinstance(item, list) and len(item) >= 2:
        smi = str(item[0])
        m = item[-1]
        if m is None or not isinstance(m, Chem.Mol):
            m = Chem.MolFromSmiles(smi)
        return smi, m
    if isinstance(item, str):
        return item, Chem.MolFromSmiles(item)
    return None, None


def _bond_side(mol, root, anchor) -> list[int]:
    stack, seen, order = [root], set(), []
    while stack:
        u = stack.pop()
        if u in seen or u == anchor:
            continue
        seen.add(u)
        order.append(u)
        for nb in mol.GetAtomWithIdx(u).GetNeighbors():
            vi = nb.GetIdx()
            if vi != anchor and vi not in seen:
                stack.append(vi)
    return order


def _from_fragdb(mol, cid, n, mmp: LocalMMPConfig) -> list[CrossDockState]:
    db = Path(mmp.fragment_db_path) if mmp.fragment_db_path else None
    if db is None or not db.is_file():
        return []
    mutate, grow = _load_mutators(mmp)
    if mutate is None:
        return []

    parent = Chem.Mol(mol)
    seen: set[str] = set()
    try:
        seen.add(Chem.MolToSmiles(Chem.RemoveHs(parent)))
    except Exception:
        pass
    out: list[CrossDockState] = []
    prot = set(mmp.protected_atom_indices) if mmp.protected_atom_indices else None

    def _add(variant, note: str) -> None:
        if len(out) >= n or variant is None:
            return
        try:
            Chem.SanitizeMol(variant)
            smi = mol_to_smiles(variant)
        except Exception:
            return
        if smi in seen:
            return
        seen.add(smi)
        out.append(_st(variant, "local", cid, note))

    try:
        for item in mutate(
            parent,
            str(db),
            radius=mmp.radius,
            min_size=mmp.mutate_min_size,
            max_size=mmp.mutate_max_size,
            min_inc=mmp.mutate_min_inc,
            max_inc=mmp.mutate_max_inc,
            max_replacements=mmp.mutate_max_replacements,
            protected_ids=prot,
            return_mol=True,
        ):
            _, variant = _parse_yield(item)
            if variant is not None:
                _add(variant, "fragdb_mutate")
            if len(out) >= n:
                break
    except (FileNotFoundError, OSError, ValueError, RuntimeError):
        pass

    if mmp.grow_from_db and len(out) < n and grow is not None:
        try:
            ref_h = Chem.AddHs(Chem.Mol(parent))
            for item in grow(
                ref_h,
                str(db),
                radius=mmp.radius,
                min_atoms=mmp.grow_min_atoms,
                max_atoms=mmp.grow_max_atoms,
                max_replacements=mmp.grow_max_replacements,
                protected_ids=prot,
                return_mol=True,
            ):
                _, variant = _parse_yield(item)
                if variant is not None:
                    _add(Chem.RemoveHs(variant), "fragdb_grow")
                if len(out) >= n:
                    break
        except (OSError, ValueError, RuntimeError):
            pass
    return out


def _fallback_prune(mol, cid, n, rng, max_try, prune_max=4, skip: set[str] | None = None):
    used = skip or set()
    out: list[CrossDockState] = []
    attempts = 0

    sides: list[frozenset[int]] = []
    parent = Chem.Mol(mol)
    for bond in parent.GetBonds():
        if bond.IsInRing():
            continue
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for root, anchor in ((a1, a2), (a2, a1)):
            atoms = _bond_side(parent, root, anchor)
            if not atoms:
                continue
            nh = sum(1 for i in atoms if parent.GetAtomWithIdx(i).GetAtomicNum() > 1)
            if nh > prune_max or nh < 1:
                continue
            sides.append(frozenset(atoms))
    uniq: list[frozenset[int]] = []
    seen_sets: set[frozenset[int]] = set()
    for c in sides:
        if c not in seen_sets:
            seen_sets.add(c)
            uniq.append(c)
    rng.shuffle(uniq)

    for atoms in uniq:
        if len(out) >= n or attempts >= max_try:
            break
        attempts += 1
        rw = Chem.RWMol(Chem.Mol(mol))
        for idx in sorted(atoms, reverse=True):
            rw.RemoveAtom(int(idx))
        try:
            variant = rw.GetMol()
            Chem.SanitizeMol(variant)
            smi = mol_to_smiles(variant)
        except Exception:
            continue
        if smi in used:
            continue
        used.add(smi)
        out.append(_st(variant, "local", cid, "mmp_fallback_prune"))

    while len(out) < n and attempts < max_try:
        attempts += 1
        rw = Chem.RWMol(Chem.Mol(mol))
        if rw.GetNumAtoms() <= 3:
            break
        leaves = [a.GetIdx() for a in rw.GetAtoms() if a.GetDegree() == 1 and a.GetAtomicNum() != 1]
        if not leaves:
            break
        rw.RemoveAtom(rng.choice(leaves))
        try:
            variant = rw.GetMol()
            Chem.SanitizeMol(variant)
            smi = mol_to_smiles(variant)
        except Exception:
            continue
        if smi in used:
            continue
        used.add(smi)
        out.append(_st(variant, "local", cid, f"mmp_fallback_leaf#{attempts}"))
    return out


def build_local(mol, cid, n, rng, max_try, mmp):
    opts = mmp or LocalMMPConfig()
    out: list[CrossDockState] = []
    if opts.fragment_db_path:
        out.extend(_from_fragdb(mol, cid, n, opts))
    if len(out) < n:
        seen = {s for s in (x.smiles for x in out if x.smiles)}
        out.extend(_fallback_prune(mol, cid, n - len(out), rng, max_try, opts.fallback_prune_max_heavy, skip=seen))
    return out[:n]


# --- pharm (PLIP-protected jitter) ---

def _pdb_max_serial(lines) -> int:
    mx = 0
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            try:
                mx = max(mx, int(line[6:11]))
            except ValueError:
                continue
    return mx


def _lig_pdb_lines(lig, start):
    block = Chem.MolToPDBBlock(lig)
    lines, rdk_to_serial = [], {}
    i = 0
    for line in block.splitlines():
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
            continue
        serial = start + i
        lines.append(f"{line[0:6]}{serial:5d}{line[11:]}")
        rdk_to_serial[i] = serial
        i += 1
    serial_to_rdk = {v: k for k, v in rdk_to_serial.items()}
    return lines, serial_to_rdk


def _merge_pdb(rec_pdb, lig):
    raw = Path(rec_pdb).read_text(errors="replace").splitlines()
    body = [ln for ln in raw if ln.strip().upper() != "END"]
    lig_lines, serial_to_rdk = _lig_pdb_lines(lig, _pdb_max_serial(body) + 1)
    return "\n".join(body + lig_lines + ["END"]), serial_to_rdk


def _add_int(ob, bag: set[int]) -> None:
    if ob is None or isinstance(ob, bool):
        return
    if isinstance(ob, int):
        bag.add(ob)
        return
    if isinstance(ob, float) and ob.is_integer():
        bag.add(int(ob))
        return
    if isinstance(ob, (list, tuple, set)):
        for x in ob:
            _add_int(x, bag)
        return
    for attr in ("orig_idx", "center", "atoms", "idx", "atom_idx"):
        if hasattr(ob, attr):
            _add_int(getattr(ob, attr), bag)


def _plip_serials(site) -> set[int]:
    serials: set[int] = set()
    for hb in getattr(site, "hbonds_ldon", None) or []:
        _add_int(getattr(hb, "d_orig_idx", None), serials)
        _add_int(getattr(hb, "a_orig_idx", None), serials)
    for hb in getattr(site, "hbonds_pdon", None) or []:
        _add_int(getattr(hb, "d_orig_idx", None), serials)
        _add_int(getattr(hb, "a_orig_idx", None), serials)
    for hyd in getattr(site, "hydrophobic_contacts", None) or []:
        _add_int(getattr(hyd, "ligatom_orig_idx", None), serials)
        _add_int(getattr(hyd, "ligaa_orig_idx", None), serials)
    for pistack in getattr(site, "pistacking", None) or []:
        lr = getattr(pistack, "ligandring", pistack)
        _add_int(getattr(lr, "atoms", None), serials)
    for name in ("pication_laro", "pication_paro"):
        for pic in getattr(site, name, None) or []:
            ring = getattr(pic, "ring", pic)
            _add_int(getattr(ring, "atoms", None), serials)
    for name in ("saltbridge_lneg", "saltbridge_pneg"):
        for salt in getattr(site, name, None) or []:
            neg, pos = getattr(salt, "negative", None), getattr(salt, "positive", None)
            if neg is not None:
                _add_int(getattr(neg, "center", getattr(neg, "orig_idx", None)), serials)
            if pos is not None:
                _add_int(getattr(pos, "center", getattr(pos, "orig_idx", None)), serials)
    for hal in getattr(site, "halogen_bonds", None) or []:
        _add_int(getattr(hal, "don_orig_idx", None), serials)
        _add_int(getattr(hal, "acc_orig_idx", None), serials)
    for wb in getattr(site, "water_bridges", None) or []:
        _add_int(getattr(wb, "don_orig_idx", None), serials)
        _add_int(getattr(wb, "acc_orig_idx", None), serials)
    for mc in getattr(site, "metal_complexes", None) or []:
        _add_int(getattr(mc, "metal_idx", None), serials)
        _add_int(getattr(mc, "target_idx", None), serials)
    return serials


def _run_plip(pdb_path):
    from plip.structure.preparation import PDBComplex

    cx = PDBComplex()
    cx.load_pdb(str(pdb_path))
    if hasattr(cx, "analyze"):
        cx.analyze()
    else:
        for lig_id in getattr(cx, "ligands", None) or {}:
            if hasattr(cx, "characterize_complex"):
                cx.characterize_complex(lig_id)
    return cx


def _load_lig(path) -> Chem.Mol | None:
    p = Path(path)
    if p.suffix.lower() == ".sdf":
        for m in Chem.SDMolSupplier(str(p), removeHs=False):
            if m is not None:
                return m
        return None
    try:
        return Chem.MolFromMolFile(str(p), removeHs=False)
    except Exception:
        return None


def _pharm_prot(rec_pdb, lig_pdb, feat_src):
    text, serial_to_rdk = _merge_pdb(rec_pdb, lig_pdb)
    tmp: str | None = None
    fd, tmp = tempfile.mkstemp(suffix=".pdb", text=True)
    try:
        with os.fdopen(fd, "w", encoding="ascii", errors="replace") as fh:
            fh.write(text)
        sites = getattr(_run_plip(Path(tmp)), "interaction_sets", None) or {}
        raw: set[int] = set()
        for site in sites.values():
            raw |= _plip_serials(site)
        mapped = {serial_to_rdk[ser] for ser in raw if ser in serial_to_rdk}

        lig = Chem.Mol(feat_src)
        fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        feats = ChemicalFeatures.BuildFeatureFactory(fdef).GetFeaturesForMol(lig)
        protected: set[int] = set()
        n_hits = 0
        for feat in feats:
            aids = set(feat.GetAtomIds())
            if aids & mapped:
                protected |= aids
                n_hits += 1
        stats = f"plip_sites={len(sites)},raw={len(raw)},map={len(mapped)},feat={n_hits},prot={len(protected)}"
        return protected, stats
    finally:
        if tmp is not None:
            try:
                Path(tmp).unlink(missing_ok=True)
            except OSError:
                pass


def build_pharm(mol, cid, n, rng, rec_pdb, lig_sdf=None, sigma=0.08):
    if n <= 0:
        return []

    def _as_geo():
        return [
            CrossDockState(
                X=s.X,
                pool_kind="pharm",
                complex_id=s.complex_id,
                source_note="pharm_fallback_geo:" + s.source_note,
                smiles=s.smiles,
            )
            for s in build_geo(mol, cid, n, rng, sigma)
        ]

    if rec_pdb is None or not Path(rec_pdb).is_file():
        return _as_geo()

    lig_pdb = Chem.Mol(mol)
    ensure_conformer(lig_pdb)
    if lig_pdb.GetNumConformers() == 0:
        return _as_geo()

    feat_src = Chem.Mol(mol)
    if lig_sdf is not None and str(lig_sdf).strip():
        lp = Path(lig_sdf)
        if lp.is_file():
            alt = _load_lig(lp)
            if alt is not None and alt.GetNumAtoms() == mol.GetNumAtoms():
                if Chem.MolToSmiles(Chem.RemoveHs(alt)) == Chem.MolToSmiles(Chem.RemoveHs(mol)):
                    feat_src = alt

    try:
        protected, stats = _pharm_prot(Path(rec_pdb), lig_pdb, feat_src)
    except (ImportError, Exception):
        return _as_geo()

    n_atoms = mol.GetNumAtoms()
    protected &= {i for i in protected if 0 <= i < n_atoms}
    if not protected or protected == set(range(n_atoms)):
        return _as_geo()

    movable = {i for i in range(n_atoms) if i not in protected}
    if not movable:
        return _as_geo()

    out: list[CrossDockState] = []
    for i in range(n):
        got = _jitter(lig_pdb, rng, sigma, atoms=movable)
        if got is None:
            break
        m, conf_id = got
        out.append(_st(m, "pharm", cid, f"pharm_plip|{stats}|#{i}", conf_id=conf_id))
    return out
