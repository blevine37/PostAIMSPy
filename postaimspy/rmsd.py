"""
postaimspy.rmsd — RMSD calculation with Kabsch / quaternion rotation.

Pure-numpy implementation (based on the public-domain rmsd library by
J. Charnley, with modifications by A. Mehmood).
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


# ══════════════════════════════════════════════════════════════════════════════
# Core math
# ══════════════════════════════════════════════════════════════════════════════
def centroid(X: np.ndarray) -> np.ndarray:
    return X.mean(axis=0)


def rmsd(V: np.ndarray, W: np.ndarray) -> float:
    """Plain RMSD (no rotation)."""
    diff = np.asarray(V) - np.asarray(W)
    return np.sqrt((diff * diff).sum() / len(V))


# ── Kabsch ────────────────────────────────────────────────────────────────
def kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Optimal rotation matrix U that rotates P onto Q (both centred)."""
    C = P.T @ Q
    V, _S, W = np.linalg.svd(C)
    if np.linalg.det(V) * np.linalg.det(W) < 0.0:
        V[:, -1] *= -1
    return V @ W


def kabsch_rotate(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Rotate P onto Q and return rotated P."""
    U = kabsch(P, Q)
    return P @ U


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray, translate: bool = False) -> float:
    """RMSD after optimal Kabsch rotation."""
    P, Q = np.array(P, copy=True, dtype=float), np.array(Q, copy=True, dtype=float)
    if translate:
        P -= centroid(P)
        Q -= centroid(Q)
    P_rot = kabsch_rotate(P, Q)
    return rmsd(P_rot, Q)


# ── Quaternion (Horn 1987) ────────────────────────────────────────────────
def quaternion_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """
    RMSD via quaternion-based superposition (Horn 1987).
    P and Q must be centred (Nx3 arrays).
    """
    N = len(P)
    M = P.T @ Q  # 3x3 cross-covariance
    Sxx, Sxy, Sxz = M[0, 0], M[0, 1], M[0, 2]
    Syx, Syy, Syz = M[1, 0], M[1, 1], M[1, 2]
    Szx, Szy, Szz = M[2, 0], M[2, 1], M[2, 2]

    K = np.array([
        [Sxx + Syy + Szz,  Syz - Szy,        Szx - Sxz,        Sxy - Syx       ],
        [Syz - Szy,        Sxx - Syy - Szz,   Sxy + Syx,        Szx + Sxz       ],
        [Szx - Sxz,        Sxy + Syx,        -Sxx + Syy - Szz,  Syz + Szy       ],
        [Sxy - Syx,        Szx + Sxz,         Syz + Szy,       -Sxx - Syy + Szz ],
    ])

    eigvals, _ = np.linalg.eigh(K)
    max_eval = eigvals[-1]

    p2 = (P * P).sum()
    q2 = (Q * Q).sum()
    msd = (p2 + q2 - 2.0 * max_eval) / N
    if msd < 0.0:
        msd = 0.0
    return np.sqrt(msd)


# ══════════════════════════════════════════════════════════════════════════════
# Reordering (Hungarian)
# ══════════════════════════════════════════════════════════════════════════════
def reorder_hungarian(
    p_atoms: np.ndarray, q_atoms: np.ndarray,
    p_coord: np.ndarray, q_coord: np.ndarray,
) -> np.ndarray:
    """Reorder atoms of Q to best match P using the Hungarian algorithm."""
    unique_atoms = np.unique(p_atoms)
    new_order = np.zeros(len(q_atoms), dtype=int)
    for atom_type in unique_atoms:
        p_idx = np.where(p_atoms == atom_type)[0]
        q_idx = np.where(q_atoms == atom_type)[0]
        A = p_coord[p_idx]
        B = q_coord[q_idx]
        dist = cdist(A, B, "euclidean")
        row_ind, col_ind = linear_sum_assignment(dist)
        new_order[p_idx[row_ind]] = q_idx[col_ind]
    return new_order


# ══════════════════════════════════════════════════════════════════════════════
# High-level interface
# ══════════════════════════════════════════════════════════════════════════════
def compute_rmsd(
    atoms_ref: list[str],
    coords_ref: np.ndarray,
    atoms_target: list[str],
    coords_target: np.ndarray,
    method: str = "kabsch",
    reorder: bool = False,
    ignore_hydrogen: bool = False,
) -> float:
    """Compute RMSD between reference and target."""
    from postaimspy.utils import symbol_to_z

    p_atoms = np.array([symbol_to_z(a) for a in atoms_ref])
    q_atoms = np.array([symbol_to_z(a) for a in atoms_target])
    p_all = np.array(coords_ref, dtype=float)
    q_all = np.array(coords_target, dtype=float)

    if ignore_hydrogen:
        p_mask = p_atoms != 1
        q_mask = q_atoms != 1
        p_atoms, p_all = p_atoms[p_mask], p_all[p_mask]
        q_atoms, q_all = q_atoms[q_mask], q_all[q_mask]

    if reorder:
        order = reorder_hungarian(p_atoms, q_atoms, p_all, q_all)
        q_all = q_all[order]
        q_atoms = q_atoms[order]

    P = copy.deepcopy(p_all) - centroid(p_all)
    Q = copy.deepcopy(q_all) - centroid(q_all)

    if method == "kabsch":
        return kabsch_rmsd(P, Q)
    elif method == "quaternion":
        return quaternion_rmsd(P, Q)
    else:
        return rmsd(P, Q)


def minimize_rmsd_and_rotate(
    atoms_ref: list[str],
    coords_ref: np.ndarray,
    atoms_target: list[str],
    coords_target: np.ndarray,
    method: str = "kabsch",
) -> tuple[np.ndarray, float]:
    """
    Rotate *coords_target* onto *coords_ref* using Kabsch,
    returning (rotated_coords, rmsd_value).
    """
    P = np.array(coords_ref, dtype=float)
    Q = np.array(coords_target, dtype=float)

    p_cent = centroid(P)
    q_cent = centroid(Q)
    Pc = P - p_cent
    Qc = Q - q_cent

    U = kabsch(Qc, Pc)
    Q_rot = (Q - q_cent) @ U + p_cent
    rmsd_val = rmsd(Pc, kabsch_rotate(Qc, Pc))
    return Q_rot, rmsd_val
