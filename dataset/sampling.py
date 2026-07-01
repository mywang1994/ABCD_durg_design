from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from mol_state import MolState
from utils.distances import d_mcm, edit_distance_molstate, molstate_to_mol


# --- shared ---

def _key(X: MolState) -> tuple:
    edges = tuple(sorted(tuple(sorted(e)) for e in X.G.edges()))
    return (edges, X.R.tobytes(), X.node_feat.tobytes())


def is_valid(X: MolState) -> bool:
    if X.num_nodes() < 1:
        return False
    if not np.isfinite(X.R).all():
        return False
    return molstate_to_mol(X) is not None


def edits_ok(states: list[MolState], d_edit: int) -> bool:
    m = len(states)
    for i in range(m):
        for j in range(i + 1, m):
            if edit_distance_molstate(states[i], states[j]) > int(d_edit):
                return False
    return True


# --- mapping groups (many-to-one basin training) ---

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
    g_min: int = 2
    g_max: int = 4
    valid: bool = True
    max_tries: int = 50


class MapSampler:
    def __init__(self, cfg: MapCfg | None = None):
        self.cfg = cfg or MapCfg()

    def eligible(self, pool: list[MolState], x_star: MolState) -> list[MolState]:
        tau = float(self.cfg.tau)
        out: list[MolState] = []
        for X in pool:
            if self.cfg.valid and not is_valid(X):
                continue
            if d_mcm(X, x_star) > tau:
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
            if all(edit_distance_molstate(cand, g) <= int(self.cfg.d_edit) for g in picked):
                picked.append(cand)
                seen.add(fp)

        if len(picked) < lo or not edits_ok(picked, self.cfg.d_edit):
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


# --- trajectory chains (progressive / leap-like step training) ---

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
