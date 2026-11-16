"""Cache deterministic attraction-basin pools as gzip JSON, never pickled objects.
Mapping groups, losses, model features, and optimization results remain uncached.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import tempfile
import zlib
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import networkx as nx
import numpy as np

from config import POCKET_SCORE_SCHEMA
from mol_state import MolState
from .pool_core import CrossDockState, PoolCfg, StatePool, _local_mmp
from .pool_builders import validate_mmp_backend
from .sampling import is_valid
from .state_pool import POOL_CAP_POLICY, build_crossdock_bundle


POOL_CACHE_SCHEMA = "abcd_state_pool_json_v1"
_CODE_FILES = (
    "mol_state.py", "dataset/pool_core.py", "dataset/pool_builders.py",
    "dataset/state_pool.py", "dataset/scaffold_tree.py", "dataset/pool_cache.py",
    "dataset/protein_data.py", "dataset/sampling.py", "utils/chemistry.py",
    "utils/pkt_fit.py", "utils/distances.py", "utils/molecular_distance.py",
    "utils/edit_distance.py", "utils/geometry_validation.py",
)


class PoolCacheError(ValueError):
    """A cache entry is corrupt/incompatible or its source changed mid-build."""


def _plain(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _file_stat(path: Path) -> tuple[int, int, int, int, int]:
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


@lru_cache(maxsize=4096)
def _content_digest(path: str, stamp: tuple) -> str:
    del stamp
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    path = path.resolve()
    return _content_digest(str(path), _file_stat(path))


def _dependency(package: str, module: str) -> dict:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        spec = None
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = None
    # Include the Open Babel version even when Conda bindings lack pip dist-info.
    if module == "openbabel" and spec is not None:
        try:
            from openbabel import openbabel
            runtime_version = str(openbabel.OBReleaseVersion())
        except (ImportError, OSError, AttributeError):
            runtime_version = None
        return {"available": spec is not None, "version": version,
                "runtime_version": runtime_version}
    return {"available": spec is not None, "version": version}


def _identity(cx, cfg: PoolCfg, implementation_key: str) -> dict:
    if not isinstance(implementation_key, str) or not implementation_key:
        raise ValueError("Pool cache requires a nonempty implementation_key")
    root = Path(__file__).resolve().parent.parent
    code = {name: _sha256(root / name) for name in _CODE_FILES}
    mmp = _local_mmp(cfg)
    backend = validate_mmp_backend(mmp)
    db_identity = None
    if mmp.fragment_db_path is not None:
        path = Path(mmp.fragment_db_path).expanduser().resolve()
        if path.is_file():
            db_identity = {"path": str(path), "stat": list(_file_stat(path)),
                           "method": "file_identity_size_mtime_ctime_not_content_hash"}
    return {
        "schema": POOL_CACHE_SCHEMA,
        "implementation_key": implementation_key,
        "pocket_score_schema": POCKET_SCORE_SCHEMA,
        "pool_cap_policy": POOL_CAP_POLICY,
        "complex_id": cx.complex_id,
        "receptor_sha256": _sha256(Path(cx.receptor_pdb)),
        "ligand_sha256": _sha256(Path(cx.reference_ligand_sdf)),
        "pool_config": _plain(asdict(cfg)),
        "effective_mmp_config": _plain(asdict(mmp)),
        "mmp_backend": backend,
        "mmp_database_identity": db_identity,
        "dependencies": {
            "rdkit": _dependency("rdkit", "rdkit"),
            "numpy": _dependency("numpy", "numpy"),
            "scaffoldgraph": _dependency("ScaffoldGraph", "scaffoldgraph"),
            "crem": _dependency("crem", "crem"),
            "plip": _dependency("plip", "plip"),
            "openbabel": _dependency("openbabel-wheel", "openbabel"),
        },
        "source_sha256": hashlib.sha256(_canonical(code)).hexdigest(),
    }


def _encode_attribute(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_encode_attribute(x) for x in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [_encode_attribute(x) for x in value]}
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise PoolCacheError("Graph attribute keys must be strings")
        return {"type": "dict", "items": {k: _encode_attribute(v) for k, v in value.items()}}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise PoolCacheError(f"Unsupported graph attribute type: {type(value).__name__}")


def _decode_attribute(value):
    if isinstance(value, dict):
        if set(value) != {"type", "items"}:
            raise PoolCacheError("Malformed graph attribute encoding")
        if value["type"] == "dict" and isinstance(value["items"], dict):
            return {k: _decode_attribute(v) for k, v in value["items"].items()}
        if value["type"] in ("list", "tuple") and isinstance(value["items"], list):
            entries = [_decode_attribute(v) for v in value["items"]]
            return tuple(entries) if value["type"] == "tuple" else entries
        raise PoolCacheError("Unknown graph attribute encoding")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise PoolCacheError("Malformed graph attribute value")


def _array_payload(array) -> dict:
    array = np.asarray(array)
    if array.dtype.kind != "f" or array.dtype.itemsize not in (4, 8) or not np.isfinite(array).all():
        raise PoolCacheError("State arrays must contain finite float32/float64 values")
    return {"dtype": array.dtype.str, "shape": list(array.shape), "data": array.tolist()}


def _decode_array(payload) -> np.ndarray:
    if not isinstance(payload, dict) or set(payload) != {"dtype", "shape", "data"}:
        raise PoolCacheError("Malformed array record")
    dtype = np.dtype(payload["dtype"])
    if dtype.kind != "f" or dtype.itemsize not in (4, 8):
        raise PoolCacheError("Only float32/float64 state arrays are accepted")
    array = np.asarray(payload["data"], dtype=dtype)
    if list(array.shape) != payload["shape"] or not np.isfinite(array).all():
        raise PoolCacheError("Cached state array has mismatched shape or nonfinite values")
    return array


def _state_payload(state: MolState) -> dict:
    if not is_valid(state):
        raise PoolCacheError("Cannot cache an invalid molecular state")
    return {
        "graph_attributes": _encode_attribute(dict(state.G.graph)),
        "nodes": [_encode_attribute(dict(state.G.nodes[i])) for i in range(state.num_nodes())],
        "edges": [[int(u), int(v), _encode_attribute(dict(attrs))]
                  for u, v, attrs in state.G.edges(data=True)],
        "coordinates": _array_payload(state.R),
        "node_features": _array_payload(state.node_feat),
    }


def _decode_state(payload) -> MolState:
    graph = nx.Graph()
    graph.graph.update(_decode_attribute(payload["graph_attributes"]))
    for i, attrs in enumerate(payload["nodes"]):
        graph.add_node(i, **_decode_attribute(attrs))
    count = len(payload["nodes"])
    for u, v, attrs in payload["edges"]:
        if (type(u) is not int or type(v) is not int or not 0 <= u < count or not 0 <= v < count
                or u == v or graph.has_edge(u, v)):
            raise PoolCacheError("Cached graph has invalid or duplicate bond endpoints")
        graph.add_edge(u, v, **_decode_attribute(attrs))
    state = MolState(graph, _decode_array(payload["coordinates"]), _decode_array(payload["node_features"]))
    if not is_valid(state):
        raise PoolCacheError("Cached molecular state fails chemistry/shape validation")
    return state


def _pool_payload(pool: StatePool) -> dict:
    return {
        "complex_id": pool.complex_id,
        "target": _state_payload(pool.X_star),
        "build_rejections": dict(pool.build_rejections),
        "by_kind": {kind: [{"state": _state_payload(item.X), "pool_kind": item.pool_kind,
                            "complex_id": item.complex_id, "source_note": item.source_note,
                            "smiles": item.smiles} for item in states]
                    for kind, states in pool.by_kind.items()},
    }


def _decode_pool(payload, complex_id: str) -> StatePool:
    if payload["complex_id"] != complex_id:
        raise PoolCacheError("Cached pool complex id does not match its source")
    diagnostics = payload["build_rejections"]
    if not isinstance(diagnostics, dict) or not all(
            isinstance(k, str) and type(v) is int and v >= 0 for k, v in diagnostics.items()):
        raise PoolCacheError("Malformed pool diagnostics")
    by_kind = {}
    for kind, records in payload["by_kind"].items():
        if kind not in {"local", "pharm", "scaffold", "rxn", "geo"}:
            raise PoolCacheError(f"Unknown cached pool source: {kind}")
        by_kind[kind] = []
        for item in records:
            if item["pool_kind"] != kind or item["complex_id"] != complex_id:
                raise PoolCacheError("Cached state provenance does not match its bucket/source")
            if not isinstance(item["source_note"], str) or not (item["smiles"] is None or isinstance(item["smiles"], str)):
                raise PoolCacheError("Malformed source note or SMILES")
            by_kind[kind].append(CrossDockState(_decode_state(item["state"]), kind,
                                              complex_id, item["source_note"], item["smiles"]))
    return StatePool(complex_id, _decode_state(payload["target"]), by_kind, dict(diagnostics))


def _write_atomic(path: Path, payload: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                compressed.write(_canonical(payload))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _reject_json_constant(value):
    raise PoolCacheError(f"Nonfinite JSON number: {value}")


def load_or_build_pool(cx, cfg: PoolCfg, cache_dir, *, implementation_key: str):
    """Return (StatePool, report) with disabled/hit/built status.
    Hash PDB/SDF contents; identify the MMP database by path/device/inode/size/timestamps, not a full checksum.
    Invalid entries raise PoolCacheError and require explicit removal before rebuilding.
    """
    if cache_dir is None:
        return build_crossdock_bundle(cx, cfg), {"status": "disabled", "key": None, "path": None}
    identity = _identity(cx, cfg, implementation_key)
    key = hashlib.sha256(_canonical(identity)).hexdigest()
    root = Path(cache_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{key}.json.gz"
    report = {"status": "hit", "key": key, "path": str(path), "schema": POOL_CACHE_SCHEMA}
    if path.is_file():
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                payload = json.load(fh, parse_constant=_reject_json_constant)
            if payload["schema"] != POOL_CACHE_SCHEMA or payload["key"] != key or payload["identity"] != identity:
                raise PoolCacheError("Cache schema or source/configuration identity mismatch")
            return _decode_pool(payload["pool"], cx.complex_id), report
        except (OSError, EOFError, zlib.error, ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
            raise PoolCacheError(f"Invalid state pool cache {path}: {exc}; remove this entry to rebuild") from exc
    pool = build_crossdock_bundle(cx, cfg)
    if _identity(cx, cfg, implementation_key) != identity:
        raise PoolCacheError(f"Inputs/backend/code changed while building the pool for {cx.complex_id}; cache not written")
    payload = {"schema": POOL_CACHE_SCHEMA, "key": key, "identity": identity, "pool": _pool_payload(pool)}
    _write_atomic(path, payload)
    report["status"] = "built"
    return pool, report
