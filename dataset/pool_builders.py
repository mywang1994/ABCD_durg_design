from __future__ import annotations

import importlib
import importlib.metadata
import os
import random
import sqlite3
import sys
import tempfile
import threading
from contextlib import closing, contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures, rdFMCS, rdMolAlign

from .pool_core import CrossDockState, LocalMMPConfig, ensure_conformer, mol_to_molstate, mol_to_smiles



def _st(mol, kind: str, cid: str, note: str, *, conf_id: int = -1) -> CrossDockState:
    return CrossDockState(
        X=mol_to_molstate(mol, conf_id=conf_id),
        pool_kind=kind,
        complex_id=cid,
        source_note=note,
        smiles=mol_to_smiles(mol),
    )


def _record_failure(diagnostics, reason):
    if diagnostics is not None:
        diagnostics[reason] = diagnostics.get(reason, 0) + 1


def _embed_aligned_to_reference(mol, reference, rng):
    """Embed candidates and align shared atoms to the reference pose.
    Use retained reaction indices or a bounded typed MCS; report alignment failures.
    Coordinates alone do not establish pocket suitability.
    """
    if mol is None or reference is None or not mol.GetNumAtoms():
        raise ValueError("Conformer alignment requires nonempty molecules")
    ref = Chem.Mol(reference)
    ensure_conformer(ref)
    if not ref.GetNumConformers() or not np.isfinite(ref.GetConformer().GetPositions()).all():
        raise ValueError("Reference has no finite conformer")
    result = Chem.Mol(mol)
    Chem.SanitizeMol(result)
    result.RemoveAllConformers()
    # Copy back original atom positions to preserve explicit H vertices and atom indices.
    with_h = Chem.AddHs(result)
    params = AllChem.ETKDGv3()
    params.randomSeed = rng.randint(0, 2**31 - 1)
    if AllChem.EmbedMolecule(with_h, params) < 0:
        raise ValueError("Candidate conformer embedding failed")
    if AllChem.MMFFHasAllMoleculeParams(with_h):
        AllChem.MMFFOptimizeMolecule(with_h, maxIters=200)
    conf = Chem.Conformer(result.GetNumAtoms())
    for i in range(result.GetNumAtoms()):
        conf.SetAtomPosition(i, with_h.GetConformer().GetAtomPosition(i))
    result.AddConformer(conf, assignId=True)

    atom_map = []
    for atom in mol.GetAtoms():
        if atom.HasProp("react_atom_idx"):
            idx = atom.GetIntProp("react_atom_idx")
            if (0 <= idx < ref.GetNumAtoms()
                    and ref.GetAtomWithIdx(idx).GetAtomicNum() == atom.GetAtomicNum()):
                atom_map.append((atom.GetIdx(), idx))
    if len({j for _, j in atom_map}) != len(atom_map):
        raise ValueError("Reaction atom mapping is not one-to-one")
    if not atom_map:
        # Shared atom-map IDs preserve substitution correspondence.
        ref_maps = {a.GetAtomMapNum(): a.GetIdx() for a in ref.GetAtoms() if a.GetAtomMapNum()}
        atom_map = [(a.GetIdx(), ref_maps[a.GetAtomMapNum()]) for a in mol.GetAtoms()
                    if a.GetAtomMapNum() in ref_maps
                    and a.GetAtomicNum() == ref.GetAtomWithIdx(ref_maps[a.GetAtomMapNum()]).GetAtomicNum()]
    if not atom_map:
        exact = ref.GetSubstructMatch(result)
        if exact:
            atom_map = list(enumerate(exact))
        else:
            common = rdFMCS.FindMCS(
                [result, ref], timeout=1,
                atomCompare=rdFMCS.AtomCompare.CompareElements,
                bondCompare=rdFMCS.BondCompare.CompareOrder,
                matchValences=False, ringMatchesRingOnly=False,
            )
            query = Chem.MolFromSmarts(common.smartsString) if common.smartsString else None
            if query is None or query.GetNumAtoms() == 0:
                raise ValueError("No shared atoms for reference alignment")
            atom_map = list(zip(result.GetSubstructMatch(query), ref.GetSubstructMatch(query)))
    if not atom_map or len({j for _, j in atom_map}) != len(atom_map):
        raise ValueError("No one-to-one atom mapping for reference alignment")
    rdMolAlign.AlignMol(result, ref, atomMap=atom_map)
    if not np.isfinite(result.GetConformer().GetPositions()).all():
        raise ValueError("Aligned candidate coordinates are not finite")
    return result


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


def build_scaffold(mol, cid, n, rng, *, diagnostics=None):
    if n <= 0:
        return []
    from .scaffold_tree import scaffold_tree_fragments

    # Distinct recursive subscaffolds retain each atom's reference pose.
    del rng
    return [_st(fragment.mol, "scaffold", cid, fragment.source_note)
            for fragment in scaffold_tree_fragments(mol, max_states=n, diagnostics=diagnostics)]



RXN_TPL: dict[str, dict[str, Any]] = {
    "amidation": {
        "name": "Amidation Reaction",
        # Retro-amidation cleaves only non-ring C-N bonds.
        "smarts": "[C:1](=[O:2])-[OH:3].[N:4]>>[C:1](=[O:2])-!@[N:4]",
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
        # Neutral 1H and 1-substituted 1,2,3-triazoles.
        "smarts": ("[N+0;X2:1]=[N+;X2:2]=[N-;X1:3].[C;X2:4]#[C;X2:5]"
                   ">>[n+0;X3:1]1[n+0;X2;H0:2][n+0;X2;H0:3][c:4][c:5]1"),
        "fragments": {"azide": {"atoms": [1, 2, 3], "type": "azide"}, "alkyne": {"atoms": [4, 5], "type": "alkyne"}},
        "bonds": [(1, 5), (3, 4)],
        "angle_constraints": None,
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


def build_rxn(mol, *, cid: str, n: int, rng: random.Random, tpl: dict[str, dict[str, Any]] | None = None,
              diagnostics=None):
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
            try:
                aligned = _embed_aligned_to_reference(pm, mol, rng)
                out.append(_st(aligned, "rxn", cid, f"rxn_retro_aligned:{key}"))
            except (ValueError, RuntimeError):
                _record_failure(diagnostics, "rxn_conformer_or_alignment_failed")
    return out



def _subst_mod(mmp: LocalMMPConfig) -> str | None:
    name = (mmp.substitution_module or "").strip()
    if not name:
        name = (os.environ.get("ABCM_FRAGMENT_SUBST_MODULE") or "").strip()
    return name or "crem.crem"


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
        return pkg.mutate_mol, getattr(pkg, "grow_mol", None)
    except (ImportError, AttributeError):
        return None, None


class MMPConfigurationError(ValueError):
    """A requested fragment database/backend cannot be used as configured."""


@lru_cache(maxsize=16)
def _inspect_fragment_schema(path: str, radius: int, stamp: tuple[int, int, int]) -> dict:
    # Cache metadata by file identity/size/mtime; never load the full table or open it read/write.
    del stamp
    table = f"radius{radius}"
    with closing(sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True)) as connection:
        try:
            columns = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
            required = {"env", "core_smi", "core_num_atoms", "core_sma", "dist2", "freq"}
            if not required <= columns:
                raise MMPConfigurationError(
                    f"Expected a CReM context-fragment table {table} with columns {sorted(required)}; "
                    "this is not an interchangeable mmpdb rules database"
                )
            if connection.execute(f'SELECT 1 FROM "{table}" LIMIT 1').fetchone() is None:
                raise MMPConfigurationError(f"CReM table {table} is empty")
            indexed = False
            for row in connection.execute(f'PRAGMA index_list("{table}")'):
                quoted = str(row[1]).replace('"', '""')
                index_columns = list(connection.execute(f'PRAGMA index_info("{quoted}")'))
                indexed |= bool(index_columns and index_columns[0][2] == "env")
            if not indexed:
                raise MMPConfigurationError(f"CReM table {table} needs an index beginning with env")
            return {"schema": "crem_radius_tables_v1", "table": table, "env_index": True}
        finally:
            connection.close()


def validate_mmp_backend(mmp: LocalMMPConfig | None = None) -> dict:
    """Validate the database/backend without mutating data or generating candidates.
    Required mode raises configuration errors; optional mode reports them and allows labelled pruning.
    A ready backend may yield no replacement for a ligand.
    """
    mmp = mmp or LocalMMPConfig()
    name = _subst_mod(mmp)
    info = {"backend": name, "path": None, "radius": mmp.radius, "status": "disabled"}
    try:
        if mmp.fragment_db_path is None:
            if mmp.require_backend:
                raise MMPConfigurationError("MMP backend is required but no database path was configured")
            return info
        db = Path(mmp.fragment_db_path).expanduser().resolve()
        info["path"] = str(db)
        if not db.is_file():
            raise MMPConfigurationError(f"CReM fragment database does not exist: {db}")
        if isinstance(mmp.radius, bool) or not isinstance(mmp.radius, int) or mmp.radius < 1:
            raise MMPConfigurationError("CReM radius must be a positive integer")
        stat = db.stat()
        info.update(_inspect_fragment_schema(str(db), mmp.radius,
                    (stat.st_ino, stat.st_size, stat.st_mtime_ns)))
        mutate, grow = _load_mutators(mmp)
        if mutate is None or (mmp.grow_from_db and grow is None):
            raise MMPConfigurationError(
                f"CReM substitution backend {name!r} is unavailable; install crem==0.2.16 "
                "in the Python environment used to prepare/train this dataset"
            )
        if name == "crem.crem":
            pkg = importlib.import_module(name)
            version = importlib.metadata.version("crem")
            info["version"] = version
            if not version.startswith("0.2.") or not hasattr(pkg, "sqlite3") or not hasattr(pkg, "random"):
                raise MMPConfigurationError(
                    f"This radius-table database adapter requires CReM 0.2.x (supported release: 0.2.16), got {version}; "
                    "do not implicitly convert the supplied database"
                )
        info["status"] = "ready"
        return info
    except (OSError, sqlite3.Error, MMPConfigurationError, importlib.metadata.PackageNotFoundError) as exc:
        info.update(status="unavailable", error=str(exc))
        if mmp.require_backend:
            raise MMPConfigurationError(str(exc)) from exc
        return info


_CREM_LOCK = threading.RLock()


@contextmanager
def _mutator_scope(mutate, db: Path, rng: random.Random):
    """Isolate CReM 0.2.x RNG bindings and enforce read-only SQLite.
    Restore module-local bindings in finally and close connections; process-wide functions stay untouched.
    Calls are single-process per worker, with a lock serializing adapter threads.
    """
    pkg = sys.modules.get(getattr(mutate, "__module__", ""))
    if pkg is None or pkg.__name__ != "crem.crem":
        yield
        return
    connections = []

    class ReadOnlySQLite:
        def connect(self, database, *args, **kwargs):
            if Path(database).resolve() != db.resolve():
                raise MMPConfigurationError("CReM attempted to open an unexpected database")
            kwargs["uri"] = True
            connection = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", *args, **kwargs)
            connection.execute("PRAGMA query_only=ON")
            connections.append(connection)
            return connection

        def __getattr__(self, name):
            return getattr(sqlite3, name)

    with _CREM_LOCK:
        previous_sqlite, previous_random = pkg.sqlite3, pkg.random
        previous_fragment = getattr(pkg, "__fragment_mol", None)
        pkg.sqlite3, pkg.random = ReadOnlySQLite(), random.Random(rng.getrandbits(64))
        if previous_fragment is not None:
            # Sort upstream set output for reproducibility across workers.
            def ordered_fragments(*args, **kwargs):
                return sorted(previous_fragment(*args, **kwargs))
            pkg.__fragment_mol = ordered_fragments
        try:
            yield
        finally:
            for connection in connections:
                connection.close()
            pkg.sqlite3, pkg.random = previous_sqlite, previous_random
            if previous_fragment is not None:
                pkg.__fragment_mol = previous_fragment


def _parse_yield(item):
    if isinstance(item, (list, tuple)) and len(item) >= 2:
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


def _from_fragdb(mol, cid, n, mmp: LocalMMPConfig, rng, *, diagnostics=None) -> list[CrossDockState]:
    if n <= 0:
        return []
    status = validate_mmp_backend(mmp)
    if status["status"] != "ready":
        _record_failure(diagnostics, "mmp_backend_" + status["status"])
        return []
    db = Path(mmp.fragment_db_path) if mmp.fragment_db_path else None
    mutate, grow = _load_mutators(mmp)

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
            _record_failure(diagnostics, "mmp_invalid_product")
            return
        if smi in seen:
            return
        seen.add(smi)
        try:
            aligned = _embed_aligned_to_reference(variant, parent, rng)
            out.append(_st(aligned, "local", cid, note + "_aligned"))
        except (ValueError, RuntimeError):
            _record_failure(diagnostics, "local_conformer_or_alignment_failed")

    def _consume(function, source, note, **options):
        try:
            with _mutator_scope(function, db, rng):
                iterator = iter(function(source, str(db), radius=mmp.radius,
                                protected_ids=prot, return_mol=True, **options))
                try:
                    for item in iterator:
                        _, variant = _parse_yield(item)
                        if variant is not None:
                            _add(Chem.RemoveHs(variant), note)
                        else:
                            _record_failure(diagnostics, "mmp_unrecognized_product")
                        if len(out) >= n:
                            break
                finally:
                    if hasattr(iterator, "close"):
                        iterator.close()
        except (OSError, ValueError, RuntimeError, TypeError, sqlite3.Error) as exc:
            _record_failure(diagnostics, "mmp_query_failed")
            if mmp.require_backend:
                raise MMPConfigurationError(f"CReM {note} failed for {cid}: {exc}") from exc

    _consume(mutate, parent, "fragdb_mutate", min_size=mmp.mutate_min_size,
             max_size=mmp.mutate_max_size, min_inc=mmp.mutate_min_inc,
             max_inc=mmp.mutate_max_inc, max_replacements=mmp.mutate_max_replacements)

    if mmp.grow_from_db and len(out) < n and grow is not None:
        _consume(grow, Chem.AddHs(Chem.Mol(parent)), "fragdb_grow",
                 min_atoms=mmp.grow_min_atoms, max_atoms=mmp.grow_max_atoms,
                 max_replacements=mmp.grow_max_replacements)
    if not out:
        _record_failure(diagnostics, "mmp_no_valid_candidates")
    return out


def _fallback_prune(mol, cid, n, rng, max_try, prune_max=4, skip: set[str] | None = None,
                    protected_ids=()):
    used = skip or set()
    protected = set(protected_ids or ())
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
            if not atoms or protected.intersection(atoms):
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
        leaves = [a.GetIdx() for a in rw.GetAtoms() if a.GetDegree() == 1 and a.GetAtomicNum() != 1
                  and a.GetIdx() not in protected]
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


def build_local(mol, cid, n, rng, max_try, mmp, *, diagnostics=None):
    if n <= 0:
        return []
    opts = mmp or LocalMMPConfig()
    out: list[CrossDockState] = []
    out.extend(_from_fragdb(mol, cid, n, opts, rng, diagnostics=diagnostics))
    if len(out) < n:
        seen = {s for s in (x.smiles for x in out if x.smiles)}
        fallback = _fallback_prune(mol, cid, n - len(out), rng, max_try, opts.fallback_prune_max_heavy,
                                   skip=seen, protected_ids=opts.protected_atom_indices)
        if fallback:
            _record_failure(diagnostics, "mmp_local_pruning_used")
        out.extend(fallback)
    return out[:n]



def _pdb_max_serial(lines) -> int:
    mx = 0
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM", "TER   ")):
            try:
                mx = max(mx, int(line[6:11]))
            except ValueError:
                continue
    return mx


def _lig_pdb_lines(lig, start, *, residue_number=9999):
    block = Chem.MolToPDBBlock(lig)
    lines, rdk_to_serial = [], {}
    i = 0
    for line in block.splitlines():
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
            continue
        if i >= 1000 or start + i > 99999:
            raise ValueError("Ligand cannot be represented with unique fixed-width PDB atom identifiers")
        serial = start + i
        # Residue/atom identity survives PLIP serial renumbering; element and coordinate columns stay intact.
        line = (f"HETATM{serial:5d} {('A' + str(i).zfill(3)):4s} LIG Z{residue_number:4d} "
                + line[27:])
        lines.append(line)
        rdk_to_serial[i] = serial
        i += 1
    serial_to_rdk = {v: k for k, v in rdk_to_serial.items()}
    return lines, serial_to_rdk


def _merge_pdb(rec_pdb, lig):
    raw = Path(rec_pdb).read_text(errors="replace").splitlines()
    body = []
    seen_model = False
    for line in raw:
        record = line[:6].strip().upper()
        if record == "MODEL":
            if seen_model:
                break
            seen_model = True
            continue
        if record == "ENDMDL":
            break
        if record not in {"END", "MASTER"}:
            body.append(line)
    used = {(line[17:20].strip(), line[21:22], line[22:26].strip())
            for line in body if line.startswith(("ATOM  ", "HETATM"))}
    residue_number = next((n for n in range(9999, 0, -1) if ("LIG", "Z", str(n)) not in used), None)
    if residue_number is None:
        raise ValueError("No unique residue identifier is available for the reference ligand")
    lig_lines, serial_to_rdk = _lig_pdb_lines(lig, _pdb_max_serial(body) + 1,
                                            residue_number=residue_number)
    return "\n".join(body + lig_lines + ["END", ""]), serial_to_rdk


def _add_int(ob, bag: set[int]) -> None:
    if ob is None or isinstance(ob, bool):
        return
    if isinstance(ob, (int, np.integer)):
        bag.add(int(ob))
        return
    if isinstance(ob, (list, tuple, set, np.ndarray)):
        for x in ob:
            _add_int(x, bag)


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
        _add_int(getattr(lr, "atoms_orig_idx", None), serials)
    for name in ("pication_laro", "pication_paro"):
        for pic in getattr(site, name, None) or []:
            ring = getattr(pic, "ring", pic)
            _add_int(getattr(ring, "atoms_orig_idx", None), serials)
            _add_int(getattr(getattr(pic, "charge", None), "atoms_orig_idx", None), serials)
    for name in ("saltbridge_lneg", "saltbridge_pneg"):
        for salt in getattr(site, name, None) or []:
            neg, pos = getattr(salt, "negative", None), getattr(salt, "positive", None)
            if neg is not None:
                _add_int(getattr(neg, "atoms_orig_idx", None), serials)
            if pos is not None:
                _add_int(getattr(pos, "atoms_orig_idx", None), serials)
    for hal in getattr(site, "halogen_bonds", None) or []:
        _add_int(getattr(hal, "don_orig_idx", None), serials)
        _add_int(getattr(hal, "acc_orig_idx", None), serials)
    for wb in getattr(site, "water_bridges", None) or []:
        _add_int(getattr(wb, "d_orig_idx", None), serials)
        _add_int(getattr(wb, "a_orig_idx", None), serials)
    for mc in getattr(site, "metal_complexes", None) or []:
        _add_int(getattr(mc, "metal_orig_idx", None), serials)
        _add_int(getattr(mc, "target_orig_idx", None), serials)
    return serials


def _run_plip(pdb_path, *, output_dir=None):
    from plip.structure.preparation import PDBComplex

    cx = PDBComplex()
    if output_dir is not None:
        cx.output_path = str(output_dir)
    cx.load_pdb(str(pdb_path))
    if hasattr(cx, "analyze"):
        cx.analyze()
    else:
        for lig_id in getattr(cx, "ligands", None) or {}:
            if hasattr(cx, "characterize_complex"):
                cx.characterize_complex(lig_id)
    return cx


class PharmacophoreMappingError(ValueError):
    """PLIP could not retain the explicitly marked reference ligand identity."""


def _plip_ligand_atom_map(complex_object, source_text, original_serial_to_rdk):
    """Map PLIP interaction IDs through retained residue/atom identities.
    *_orig_idx uses corrected-complex serials via Mapper.proteinmap, not input PDB serials,
    ligand-local indices, or coordinate components.
    """
    identities = {}
    for line in source_text.splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            serial = int(line[6:11])
            if serial in original_serial_to_rdk:
                key = (line[17:20].strip(), line[21:22], int(line[22:26]), line[12:16].strip())
                if key in identities:
                    raise PharmacophoreMappingError("Reference ligand atom identifiers are not unique")
                identities[key] = original_serial_to_rdk[serial]
    mapper = getattr(complex_object, "Mapper", None)
    if mapper is None or mapper.original_structure is None or not mapper.proteinmap:
        raise PharmacophoreMappingError("PLIP did not expose its corrected-complex atom mapping")
    result = {}
    for internal_id, plip_id in mapper.proteinmap.items():
        atom = mapper.original_structure.GetAtom(int(internal_id))
        residue = atom.GetResidue() if atom is not None else None
        if residue is None:
            continue
        key = (str(residue.GetName()).strip(), str(residue.GetChain()),
               int(residue.GetNum()), str(residue.GetAtomID(atom)).strip())
        if key in identities:
            rdk_id = identities[key]
            if rdk_id in result.values():
                raise PharmacophoreMappingError("PLIP returned duplicated reference ligand atom identities")
            result[int(plip_id)] = rdk_id
    if set(result.values()) != set(original_serial_to_rdk.values()):
        raise PharmacophoreMappingError("PLIP did not retain every reference ligand atom identity")
    return result


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
    text, input_serial_to_rdk = _merge_pdb(rec_pdb, lig_pdb)
    # Keep PLIP auxiliary files in this scope for cleanup.
    with tempfile.TemporaryDirectory(prefix="abcd-plip-") as temp:
        path = Path(temp) / "complex.pdb"
        path.write_text(text, encoding="ascii", errors="replace")
        complex_object = _run_plip(path, output_dir=temp)
        all_sites = getattr(complex_object, "interaction_sets", None) or {}
        serial_to_rdk = _plip_ligand_atom_map(complex_object, text, input_serial_to_rdk)
        reference_sites = {
            f"{line[17:20].strip()}:{line[21:22]}:{int(line[22:26])}"
            for line in text.splitlines()
            if line.startswith(("ATOM  ", "HETATM")) and int(line[6:11]) in input_serial_to_rdk
        }
        sites = {key: site for key, site in all_sites.items() if key in reference_sites}
        if not sites:
            raise PharmacophoreMappingError("PLIP did not characterize the marked reference ligand")
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
        stats = (f"plip_sites={len(sites)},raw={len(raw)},map={len(mapped)},feat={n_hits},prot={len(protected)},"
                 f"ligand_atoms_mapped={len(serial_to_rdk)},total_sites={len(all_sites)}")
        return protected, stats


def build_pharm(mol, cid, n, rng, rec_pdb, lig_sdf=None, sigma=0.08, *, diagnostics=None):
    if n <= 0:
        return []

    def _as_geo():
        return [
            CrossDockState(
                X=s.X,
                pool_kind="geo",
                complex_id=s.complex_id,
                source_note="pharm_fallback_geo:" + s.source_note,
                smiles=s.smiles,
            )
            for s in build_geo(mol, cid, n, rng, sigma)
        ]

    if rec_pdb is None or not Path(rec_pdb).is_file():
        _record_failure(diagnostics, "pharm_pocket_missing")
        return _as_geo()

    lig_pdb = Chem.Mol(mol)
    ensure_conformer(lig_pdb)
    if lig_pdb.GetNumConformers() == 0:
        _record_failure(diagnostics, "pharm_conformer_missing")
        return _as_geo()

    # Use lig_pdb for features: PLIP serials follow its atom order, which external SDFs may not share.
    feat_src = Chem.Mol(lig_pdb)

    try:
        protected, stats = _pharm_prot(Path(rec_pdb), lig_pdb, feat_src)
    except ImportError:
        _record_failure(diagnostics, "pharm_backend_unavailable")
        return _as_geo()
    except PharmacophoreMappingError:
        _record_failure(diagnostics, "pharm_atom_mapping_failed")
        return _as_geo()
    except Exception:
        _record_failure(diagnostics, "pharm_interactions_failed")
        return _as_geo()

    n_atoms = mol.GetNumAtoms()
    protected &= {i for i in protected if 0 <= i < n_atoms}
    if not protected or protected == set(range(n_atoms)):
        _record_failure(diagnostics, "pharm_no_protected_atoms" if not protected else "pharm_no_movable_atoms")
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
