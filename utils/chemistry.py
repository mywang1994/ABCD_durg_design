from __future__ import annotations

import networkx as nx
import numpy as np
from rdkit import Chem

ATOMIC_NUMBERS = (1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53)
FORMAL_CHARGES = (-2, -1, 0, 1, 2)
EXPLICIT_HS = (0, 1, 2, 3, 4)
BOND_TYPES = ("single", "double", "triple", "aromatic")
CHEMISTRY_SCHEMA_VERSION = 2

_RDKIT_BONDS = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)


def encode_validated_state(state, *, node_feat_dim, max_nodes=None, require_connected=False, context="Input"):
    from utils.distances import molstate_to_mol

    n = state.num_nodes()
    if max_nodes is not None and not 1 <= n <= max_nodes:
        raise ValueError(f"Atom count {n} is outside [1, {max_nodes}]")
    if require_connected and not nx.is_connected(state.G):
        raise ValueError(f"{context} molecule must be a single connected component")
    if molstate_to_mol(state) is None:
        raise ValueError(f"{context} molecule fails chemistry/coordinate validation")
    atom_targets(state)
    bond_targets(state)
    return state.copy_with(node_feat=graph_node_features(state.G, node_feat_dim))


def _validate_nodes(graph):
    n = graph.number_of_nodes()
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("Molecular graphs must be undirected simple graphs")
    if set(graph.nodes) != set(range(n)):
        raise ValueError("Molecular node indices must be consecutive integers starting at zero")
    return n


def graph_from_mol(mol):
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(
            atom.GetIdx(), atomic_num=int(atom.GetAtomicNum()),
            formal_charge=int(atom.GetFormalCharge()),
            num_explicit_hs=int(atom.GetNumExplicitHs()),
            no_implicit=bool(atom.GetNoImplicit()),
            is_aromatic=bool(atom.GetIsAromatic()),
            chiral_tag=int(atom.GetChiralTag()), isotope=int(atom.GetIsotope()),
            radical_electrons=int(atom.GetNumRadicalElectrons()),
            atom_map_num=int(atom.GetAtomMapNum()),
        )
    for bond in mol.GetBonds():
        try:
            label = _RDKIT_BONDS.index(bond.GetBondType()) + 1
        except ValueError as exc:
            raise ValueError(f"Unsupported bond type: {bond.GetBondType()}") from exc
        graph.add_edge(
            bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=label,
            bond_index=int(bond.GetIdx()), begin_atom=int(bond.GetBeginAtomIdx()),
            end_atom=int(bond.GetEndAtomIdx()), stereo=int(bond.GetStereo()),
            stereo_atoms=tuple(int(i) for i in bond.GetStereoAtoms()),
            bond_dir=int(bond.GetBondDir()),
        )
    return graph


def graph_to_mol(graph, *, sanitize=True):
    """Preserve input stereo; decoded graphs carry no predicted stereo or isotopes."""
    n = _validate_nodes(graph)
    rw = Chem.RWMol()
    for i in range(n):
        attrs = graph.nodes[i]
        if "atomic_num" not in attrs:
            raise ValueError(f"Atom {i} is missing atomic_num")
        z = int(attrs["atomic_num"])
        if not 1 <= z <= 118:
            raise ValueError(f"Invalid atomic number {z} at atom {i}")
        atom = Chem.Atom(z)
        atom.SetFormalCharge(int(attrs.get("formal_charge", 0)))
        atom.SetNumExplicitHs(int(attrs.get("num_explicit_hs", 0)))
        atom.SetNoImplicit(bool(attrs.get("no_implicit", False)))
        atom.SetIsAromatic(bool(attrs.get("is_aromatic", False)))
        atom.SetChiralTag(Chem.ChiralType(int(attrs.get("chiral_tag", 0))))
        atom.SetIsotope(int(attrs.get("isotope", 0)))
        atom.SetNumRadicalElectrons(int(attrs.get("radical_electrons", 0)))
        atom.SetAtomMapNum(int(attrs.get("atom_map_num", 0)))
        rw.AddAtom(atom)

    edges = list(graph.edges(data=True))
    # Preserve bond order because chiral tags depend on neighbor ordering.
    if edges and all("bond_index" in attrs for _, _, attrs in edges):
        edges.sort(key=lambda item: int(item[2]["bond_index"]))
    oriented = []
    for u, v, attrs in edges:
        if u == v:
            raise ValueError("A molecule cannot contain self bonds")
        if "bond_type" not in attrs:
            raise ValueError(f"Bond ({u}, {v}) is missing bond_type")
        label = int(attrs["bond_type"])
        if not 1 <= label <= len(_RDKIT_BONDS):
            raise ValueError(f"Invalid bond label {label} at ({u}, {v})")
        begin = int(attrs.get("begin_atom", u))
        end = int(attrs.get("end_atom", v))
        if {begin, end} != {u, v}:
            raise ValueError("Bond orientation metadata does not match its endpoints")
        rw.AddBond(begin, end, _RDKIT_BONDS[label - 1])
        b = rw.GetBondBetweenAtoms(begin, end)
        b.SetBondDir(Chem.BondDir(int(attrs.get("bond_dir", 0))))
        oriented.append((begin, end, attrs))
    # SetStereoAtoms requires the reference atoms' connecting bonds to exist.
    for begin, end, attrs in oriented:
        b = rw.GetBondBetweenAtoms(begin, end)
        stereo_atoms = attrs.get("stereo_atoms", ())
        if stereo_atoms:
            if len(stereo_atoms) != 2:
                raise ValueError("Double-bond stereo requires two reference atoms")
            b.SetStereoAtoms(int(stereo_atoms[0]), int(stereo_atoms[1]))
        b.SetStereo(Chem.BondStereo(int(attrs.get("stereo", 0))))
    mol = rw.GetMol()
    if sanitize:
        # Skip CLEANUP so neutral pentavalent nitrogen is rejected, not charged.
        flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_CLEANUP
        Chem.SanitizeMol(mol, sanitizeOps=flags)
        Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
    return mol


def graph_node_features(graph, feat_dim=16):
    """Invalid intermediates use graph-only features and remain invalid at export."""
    n = _validate_nodes(graph)
    if int(feat_dim) <= 0:
        raise ValueError("feat_dim must be positive")
    for i in range(n):
        if "atomic_num" not in graph.nodes[i]:
            raise ValueError(f"Atom {i} is missing atomic_num")
    try:
        mol = graph_to_mol(graph)
    except (ValueError, RuntimeError, KeyError, OverflowError):
        mol = None
    ring_atoms = {i for cycle in nx.cycle_basis(graph) for i in cycle}
    rows = []
    for i in range(n):
        attrs = graph.nodes[i]
        atom = mol.GetAtomWithIdx(i) if mol is not None else None
        rows.append([
            float(attrs["atomic_num"]) / 100.0,
            float(graph.degree[i]) / 6.0,
            float(atom.GetIsAromatic() if atom is not None else attrs.get("is_aromatic", False)),
            float(atom.GetTotalNumHs() if atom is not None else attrs.get("num_explicit_hs", 0)) / 4.0,
            float(attrs.get("formal_charge", 0)) / 4.0,
            float(atom.IsInRing() if atom is not None else i in ring_atoms),
            float(atom.GetHybridization()) if atom is not None else 0.0,
            float(attrs.get("chiral_tag", 0)),
        ])
    raw = np.asarray(rows, dtype=np.float32).reshape(n, 8)
    out = np.zeros((n, int(feat_dim)), dtype=np.float32)
    width = min(8, int(feat_dim))
    out[:, :width] = raw[:, :width]
    return out


def atom_targets(state, device=None):
    import torch

    if _validate_nodes(state.G) != state.num_nodes():
        raise ValueError("Graph and coordinate atom counts disagree")
    fields = (("atom", "atomic_num", ATOMIC_NUMBERS),
              ("charge", "formal_charge", FORMAL_CHARGES),
              ("hydrogen", "num_explicit_hs", EXPLICIT_HS))
    targets = {}
    for key, attr, vocabulary in fields:
        labels = []
        for i in range(state.num_nodes()):
            if attr not in state.G.nodes[i]:
                raise ValueError(f"Atom {i} is missing training attribute {attr}")
            value = int(state.G.nodes[i][attr])
            if value not in vocabulary:
                raise ValueError(f"Unsupported {attr}={value} at atom {i}; vocabulary={vocabulary}")
            labels.append(vocabulary.index(value))
        targets[key] = torch.tensor(labels, dtype=torch.long, device=device)
    return targets


def bond_targets(state, device=None):
    import torch

    n = _validate_nodes(state.G)
    if n != state.num_nodes():
        raise ValueError("Graph and coordinate atom counts disagree")
    labels = torch.zeros((n, n), dtype=torch.long, device=device)
    for u, v, attrs in state.G.edges(data=True):
        if u == v or "bond_type" not in attrs:
            raise ValueError(f"Invalid or untyped bond ({u}, {v})")
        label = int(attrs["bond_type"])
        if not 1 <= label <= len(BOND_TYPES):
            raise ValueError(f"Unsupported bond_type={label}")
        labels[u, v] = labels[v, u] = label
    return labels


def decode_graph(atom_logits, charge_logits, hydrogen_logits, bond_logits, adj, *, threshold=0.5):
    import torch

    n = int(atom_logits.shape[0])
    expected = ((atom_logits, (n, len(ATOMIC_NUMBERS))),
                (charge_logits, (n, len(FORMAL_CHARGES))),
                (hydrogen_logits, (n, len(EXPLICIT_HS))),
                (bond_logits, (n, n, len(BOND_TYPES))), (adj, (n, n)))
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("edge threshold must be between zero and one")
    for tensor, shape in expected:
        if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid prediction tensor; expected finite values with shape {shape}")
    with torch.no_grad():
        elements = atom_logits.argmax(dim=-1).cpu().tolist()
        charges = charge_logits.argmax(dim=-1).cpu().tolist()
        hydrogens = hydrogen_logits.argmax(dim=-1).cpu().tolist()
        types = ((bond_logits + bond_logits.transpose(0, 1)) * 0.5).argmax(dim=-1).cpu().numpy() + 1
        adjacency = ((adj + adj.T) * 0.5).detach().cpu().numpy()
    graph = nx.Graph()
    for i in range(n):
        graph.add_node(
            i, atomic_num=ATOMIC_NUMBERS[elements[i]],
            formal_charge=FORMAL_CHARGES[charges[i]],
            num_explicit_hs=EXPLICIT_HS[hydrogens[i]],
            no_implicit=False, is_aromatic=False, chiral_tag=0, isotope=0,
            radical_electrons=0, atom_map_num=0,
        )
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] >= float(threshold):
                label = int(types[i, j])
                graph.add_edge(i, j, bond_type=label)
                if label == 4:
                    graph.nodes[i]["is_aromatic"] = True
                    graph.nodes[j]["is_aromatic"] = True
    return graph


def decode_joint_graph(out):
    """No-bond competes with each bond type in MAP decoding; summed bond
    probability would produce different edges."""
    import torch
    import torch.nn.functional as F

    logits = out["edge_logits"]
    n = int(out["atom_logits"].shape[0])
    if logits.shape != (n, n, 5):
        raise ValueError("Joint edge logits must have shape (N,N,5)")
    diagonal = torch.eye(n, device=logits.device, dtype=torch.bool)
    if bool(torch.isnan(logits).any()) or bool(torch.isposinf(logits).any()):
        raise ValueError("Joint edge logits contain NaN or positive infinity")
    if not bool(torch.isfinite(logits[~diagonal]).all()):
        raise ValueError("Off-diagonal joint edge logits must be finite")
    with torch.no_grad():
        safe = logits.masked_fill(diagonal[..., None], 0)
        labels = ((safe + safe.transpose(0, 1)) * 0.5).argmax(-1)
        labels.masked_fill_(diagonal, 0)
        adjacency = (labels > 0).to(logits.dtype)
        types = F.one_hot(labels.clamp(min=1) - 1, num_classes=4).to(logits.dtype)
    return decode_graph(out["atom_logits"], out["charge_logits"], out["hydrogen_logits"], types, adjacency)
