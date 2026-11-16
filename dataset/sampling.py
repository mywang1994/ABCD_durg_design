from __future__ import annotations

import random
from dataclasses import dataclass

import networkx as nx
import numpy as np

from mol_state import MolState
from utils.distances import d_mcm, edit_distance_molstate, molstate_to_mol



def _key(X: MolState) -> tuple:
    # Include decoder attributes: node features alone omit double-bond stereo.
    atoms = tuple((int(a.get("atomic_num", -1)), int(a.get("formal_charge", 0)),
                   int(a.get("num_explicit_hs", 0)), int(a.get("isotope", 0)),
                   int(a.get("chiral_tag", 0)), bool(a.get("no_implicit", False)),
                   bool(a.get("is_aromatic", False)), int(a.get("radical_electrons", 0)),
                   int(a.get("atom_map_num", 0))) for _, a in sorted(X.G.nodes(data=True)))
    edges = tuple(sorted((min(u, v), max(u, v), int(a.get("bond_type", -1)),
                          int(a.get("bond_index", -1)), int(a.get("begin_atom", u)),
                          int(a.get("end_atom", v)), int(a.get("stereo", 0)),
                          tuple(a.get("stereo_atoms", ())), int(a.get("bond_dir", 0)))
                         for u, v, a in X.G.edges(data=True)))
    # Distinct conformers remain separate mapping-group members.
    return (atoms, edges, (X.R.shape, str(X.R.dtype), X.R.tobytes()),
            (X.node_feat.shape, str(X.node_feat.dtype), X.node_feat.tobytes()))


def is_valid(X: MolState, *, require_geometry: bool = False) -> bool:
    try:
        n = X.num_nodes()
        if (n < 1 or X.R.shape != (n, 3) or X.node_feat.ndim != 2
                or X.node_feat.shape[0] != n or X.node_feat.shape[1] < 1
                or not np.isfinite(X.R).all() or not np.isfinite(X.node_feat).all()
                or X.G.number_of_nodes() != n or not nx.is_connected(X.G)):
            return False
        mol = molstate_to_mol(X)
        if mol is None:
            return False
        if require_geometry:
            from utils.geometry_validation import evaluate_geometry
            return evaluate_geometry(mol)["intramolecular_valid"] is True
        return True
    except (ValueError, TypeError, AttributeError, nx.NetworkXException):
        return False


def edits_ok(states: list[MolState], d_edit: int) -> bool:
    m = len(states)
    for i in range(m):
        for j in range(i + 1, m):
            if edit_distance_molstate(states[i], states[j]) > int(d_edit):
                return False
    return True



@dataclass(frozen=True)
class MapGroup:
    states: tuple[MolState, ...]
    X_star: MolState

    @property
    def n(self) -> int:
        return len(self.states)


@dataclass
class MapPool:
    groups: list[MapGroup]
    X_star: MolState
    src_n: int = 0
    n_req: int = 0
    seed: int | None = None

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self):
        return iter(self.groups)


@dataclass(frozen=True)
class MapCfg:
    tau: float = 0.92
    d_edit: int = 2
    g_min: int = 3
    g_max: int = 6
    valid: bool = True
    max_tries: int = 50
    distance_config: dict | None = None

    def __post_init__(self):
        if not 0 <= self.tau <= 1 or self.d_edit < 0:
            raise ValueError("Invalid molecular/edit distance bounds")
        if not 1 <= self.g_min <= self.g_max or self.max_tries < 1:
            raise ValueError("Invalid mapping-group sizes or retry budget")


class MapSampler:
    def __init__(self, cfg: MapCfg | None = None):
        self.cfg = cfg or MapCfg()
        self._distances = {}
        self._edits = {}
        self._geometry_valid = {}

    @staticmethod
    def _cache_key(X):
        return (_key(X), tuple(X.G.nodes), X.G.is_directed(), X.G.is_multigraph())

    def _edit(self, X, Y):
        key = tuple(sorted((_key(X)[:2], _key(Y)[:2])))
        if key not in self._edits:
            self._edits[key] = edit_distance_molstate(X, Y)
        return self._edits[key]

    def eligible(self, pool: list[MolState], x_star: MolState) -> list[MolState]:
        tau = float(self.cfg.tau)
        target_key = self._cache_key(x_star)
        distance_config = tuple(sorted((self.cfg.distance_config or {}).items()))
        out: list[MolState] = []
        for X in pool:
            try:
                state_key = self._cache_key(X)
            except (ValueError, TypeError, AttributeError, OverflowError):
                if self.cfg.valid:
                    continue
                raise
            if self.cfg.valid:
                if state_key not in self._geometry_valid:
                    self._geometry_valid[state_key] = is_valid(X, require_geometry=True)
                if not self._geometry_valid[state_key]:
                    continue
            key = (state_key, target_key, distance_config)
            if key not in self._distances:
                self._distances[key] = d_mcm(X, x_star, self.cfg.distance_config)
            if self._distances[key] > tau:
                continue
            out.append(X)
        return out

    def sample_one(self, pool: list[MolState], x_star: MolState, rng: random.Random) -> MapGroup | None:
        pool_ok = self.eligible(pool, x_star)
        lo = max(1, int(self.cfg.g_min))
        hi = max(lo, int(self.cfg.g_max))
        if len(pool_ok) < lo:
            return None

        target = rng.randint(lo, min(hi, len(pool_ok)))
        rng.shuffle(pool_ok)
        picked: list[MolState] = [pool_ok[0]]
        seen = {_key(pool_ok[0])}

        for cand in pool_ok[1:]:
            if len(picked) >= target:
                break
            fp = _key(cand)
            if fp in seen:
                continue
            if all(self._edit(cand, g) <= int(self.cfg.d_edit) for g in picked):
                picked.append(cand)
                seen.add(fp)

        if len(picked) < lo:
            return None
        return MapGroup(states=tuple(picked), X_star=x_star)

    def sample_pair(
        self,
        pool: list[MolState],
        x_star: MolState,
        rng: random.Random,
        *,
        tries: int = 32,
    ) -> tuple[MapGroup, MapGroup] | None:
        left = self.sample_one(pool, x_star, rng)
        if left is None:
            return None
        left_keys = {_key(s) for s in left.states}
        for _ in range(tries):
            right = self.sample_one(pool, x_star, rng)
            if right is None:
                continue
            if {_key(s) for s in right.states} != left_keys:
                return left, right
        return None


def build_map_pool(
    pool: list[MolState],
    x_star: MolState,
    n: int,
    *,
    seed: int | None = None,
    sampler: MapSampler | None = None,
    cfg: MapCfg | None = None,
) -> MapPool:
    if not pool or n <= 0:
        return MapPool([], x_star, src_n=len(pool), n_req=n, seed=seed)

    sampler = sampler or MapSampler(cfg)
    rng = random.Random(seed) if seed is not None else random
    groups: list[MapGroup] = []
    budget = max(1, n * sampler.cfg.max_tries)
    n_try = 0
    seen: set[tuple[tuple, ...]] = set()

    while len(groups) < n and n_try < budget:
        n_try += 1
        mg = sampler.sample_one(pool, x_star, rng)
        if mg is None:
            continue
        sig = tuple(sorted(_key(s) for s in mg.states))
        if sig in seen:
            continue
        seen.add(sig)
        groups.append(mg)

    return MapPool(groups, x_star, src_n=len(pool), n_req=n, seed=seed)



@dataclass(frozen=True)
class TrajSample:
    Xn: MolState
    X1: MolState
    X_star: MolState

    @classmethod
    def from_chain(cls, chain: TrajChain) -> TrajSample:
        if not chain.steps:
            raise ValueError("empty chain")
        Xn, X1 = chain.steps[0]
        return cls(Xn=Xn, X1=X1, X_star=chain.X_star)


@dataclass(frozen=True)
class TrajChain:
    steps: tuple[tuple[MolState, MolState], ...]
    X_star: MolState

    @property
    def n(self) -> int:
        return len(self.steps)

    def as_single(self) -> TrajSample:
        return TrajSample.from_chain(self)


@dataclass
class TrajPool:
    samples: list[TrajChain]
    x_star: MolState
    src_n: int = 0
    n_req: int = 0
    seed: int | None = None
    clen: int = 1

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)


def pick_teacher(pool, Xn, X_star, d_edit):
    dist_n = d_mcm(Xn, X_star)
    teachers = [Xp for Xp in pool if d_mcm(Xp, X_star) < dist_n and edit_distance_molstate(Xn, Xp) <= d_edit]
    if not teachers:
        return None
    teachers.sort(key=lambda x: d_mcm(x, X_star))
    return teachers[0]


def _one_chain(pool: list[MolState], x_basin: MolState, *, d_edit: int, clen: int, rng: random.Random) -> TrajChain | None:
    if not pool or clen < 1:
        return None
    here = rng.choice(pool)
    steps: list[tuple[MolState, MolState]] = []
    seen = {_key(here)}

    for _ in range(clen):
        teacher = pick_teacher(pool, here, x_basin, d_edit)
        if teacher is None:
            break
        fp = _key(teacher)
        if fp in seen:
            break
        steps.append((here, teacher))
        seen.add(fp)
        here = teacher

    if not steps:
        return None
    return TrajChain(steps=tuple(steps), X_star=x_basin)


def _one_basin(pool: list[MolState], x_basin: MolState, *, d_edit: int, rng: random.Random) -> TrajChain | None:
    if not pool:
        return None
    Xn = rng.choice(pool)
    teacher = pick_teacher(pool, Xn, x_basin, d_edit)
    if teacher is None:
        return None
    return TrajChain(steps=((Xn, teacher),), X_star=x_basin)


def build_basin_pool(pool, x_star, d_edit, n, seed=None, max_tries=50):
    if not pool or n <= 0:
        return TrajPool([], x_star, src_n=len(pool), n_req=n, seed=seed, clen=1)

    rng = random.Random(seed) if seed is not None else random
    chains: list[TrajChain] = []
    n_try = 0
    budget = max(1, n * max_tries)
    while len(chains) < n and n_try < budget:
        n_try += 1
        chain = _one_basin(pool, x_star, d_edit=d_edit, rng=rng)
        if chain:
            chains.append(chain)
    return TrajPool(chains, x_star, src_n=len(pool), n_req=n, seed=seed, clen=1)


def build_traj_pool(
    pool,
    x_star,
    d_edit,
    n,
    seed=None,
    max_tries=50,
    clen: int = 1,
    *,
    basin_leap: bool = True,
):
    if basin_leap or int(clen) <= 1:
        return build_basin_pool(pool, x_star, d_edit, n, seed=seed, max_tries=max_tries)

    if not pool or n <= 0:
        return TrajPool([], x_star, src_n=len(pool), n_req=n, seed=seed, clen=int(clen))

    rng = random.Random(seed) if seed is not None else random
    chains: list[TrajChain] = []
    n_try = 0
    budget = max(1, n * max_tries)
    chain_len = max(2, int(clen))
    while len(chains) < n and n_try < budget:
        n_try += 1
        chain = _one_chain(pool, x_star, d_edit=d_edit, clen=chain_len, rng=rng)
        if chain:
            chains.append(chain)
    return TrajPool(chains, x_star, src_n=len(pool), n_req=n, seed=seed, clen=chain_len)


def build_traj_dataset(pool, X_star, d_edit, n, seed=None, clen: int = 1):
    return build_traj_pool(pool, X_star, d_edit, n, seed=seed, clen=clen).samples
