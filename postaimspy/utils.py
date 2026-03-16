"""
postaimspy.utils — Shared utilities: XYZ/PDB I/O, element data, logging.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"


def get_logger(name: str = "postaimspy", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


log = get_logger()

# ---------------------------------------------------------------------------
# Element look-ups
# ---------------------------------------------------------------------------
ELEMENT_SYMBOLS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti",
    23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu",
    30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr",
    37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc",
    44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba",
}

SYMBOL_TO_Z = {v: k for k, v in ELEMENT_SYMBOLS.items()}

ELEMENT_MASSES = {
    1: 1.008, 2: 4.003, 3: 6.941, 4: 9.012, 5: 10.81, 6: 12.011,
    7: 14.007, 8: 15.999, 9: 18.998, 10: 20.180, 11: 22.990, 12: 24.305,
    13: 26.982, 14: 28.086, 15: 30.974, 16: 32.06, 17: 35.453, 18: 39.948,
    19: 39.098, 20: 40.08, 26: 55.845, 29: 63.546, 30: 65.38,
    35: 79.904, 53: 126.904,
}


def symbol_to_z(sym: str) -> int:
    return SYMBOL_TO_Z.get(sym.capitalize(), 0)


def z_to_symbol(z: int) -> str:
    return ELEMENT_SYMBOLS.get(z, "X")


# ---------------------------------------------------------------------------
# XYZ file I/O
# ---------------------------------------------------------------------------
def read_xyz(path: str | Path) -> tuple[list[str], np.ndarray]:
    """
    Read a single-frame XYZ file.

    Returns
    -------
    atoms : list[str]   — element symbols
    coords : ndarray     — shape (N, 3) in Angstrom
    """
    path = Path(path)
    with open(path) as fh:
        lines = fh.readlines()
    natom = int(lines[0].strip())
    atoms = []
    coords = []
    for line in lines[2: 2 + natom]:
        parts = line.split()
        atoms.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    return atoms, np.array(coords)


def read_xyz_multiframe(path: str | Path) -> list[tuple[list[str], np.ndarray, str]]:
    """
    Read a multi-frame (concatenated) XYZ file.

    Returns list of (atoms, coords, comment_line) per frame.
    """
    path = Path(path)
    frames = []
    with open(path) as fh:
        lines = fh.readlines()

    # Skip truly empty files
    if not lines or all(line.strip() == "" for line in lines):
        return frames

    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        try:
            natom = int(line)
        except ValueError:
            idx += 1
            continue
        if idx + 1 + natom >= len(lines):
            break  # truncated frame
        comment = lines[idx + 1].strip()
        atoms = []
        coords = []
        for j in range(idx + 2, idx + 2 + natom):
            parts = lines[j].split()
            atoms.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
        frames.append((atoms, np.array(coords), comment))
        idx += 2 + natom
    return frames


def write_xyz(path: str | Path, atoms: list[str], coords: np.ndarray,
              comment: str = "") -> None:
    """Write a single-frame XYZ file (Angstrom)."""
    path = Path(path)
    with open(path, "w") as fh:
        fh.write(f"{len(atoms)}\n")
        fh.write(f"{comment}\n")
        for sym, (x, y, z) in zip(atoms, coords):
            fh.write(f"{sym:>2s} {x:17.10f} {y:17.10f} {z:17.10f}\n")


def append_xyz(path: str | Path, atoms: list[str], coords: np.ndarray,
               comment: str = "") -> None:
    """Append a single frame to an XYZ trajectory file."""
    path = Path(path)
    with open(path, "a") as fh:
        fh.write(f"{len(atoms)}\n")
        fh.write(f"{comment}\n")
        for sym, (x, y, z) in zip(atoms, coords):
            fh.write(f"{sym:>2s} {x:20.15f} {y:20.15f} {z:20.15f}\n")


# ---------------------------------------------------------------------------
# PDB generation from reference XYZ
# ---------------------------------------------------------------------------
def xyz_to_pdb(xyz_path: str | Path, pdb_path: str | Path) -> Path:
    """
    Generate a properly formatted PDB topology file from a reference XYZ.

    Uses strict PDB column formatting to avoid mdtraj parsing warnings.
    The PDB format requires exact column positions:
      - HETATM records: columns 1-6, 7-11, 13-16, 18-20, 22, 23-26, 31-38, 39-46, 47-54, 55-60, 61-66, 77-78
    """
    atoms, coords = read_xyz(xyz_path)
    pdb_path = Path(pdb_path)
    with open(pdb_path, "w") as fh:
        for i, (sym, (x, y, z)) in enumerate(zip(atoms, coords), start=1):
            # Strict PDB format (80 columns):
            # HETATM serial  name altLoc resName chain resSeq iCode x y z occ bfac element charge
            sym_clean = sym.strip().capitalize()
            if len(sym_clean) <= 2:
                atom_name = f" {sym_clean:<3s}"  # right-justify for 1-2 char elements
            else:
                atom_name = f"{sym_clean:<4s}"
            fh.write(
                f"HETATM{i:5d} {atom_name:4s} UNL A{1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}"
                f"          {sym_clean:>2s}  \n"
            )
        fh.write("END\n")
    log.info("Generated PDB topology: %s", pdb_path)
    return pdb_path


# ---------------------------------------------------------------------------
# Transition-dipole alignment
# ---------------------------------------------------------------------------
def rotation_matrix_align_to_z(vec: np.ndarray) -> np.ndarray:
    """
    Return a 3x3 rotation matrix that maps *vec* onto the +Z axis.
    """
    vec = np.asarray(vec, dtype=float)
    vec = vec / np.linalg.norm(vec)
    z = np.array([0.0, 0.0, 1.0])
    if np.allclose(vec, z):
        return np.eye(3)
    if np.allclose(vec, -z):
        return np.diag([1.0, -1.0, -1.0])

    v = np.cross(vec, z)
    s = np.linalg.norm(v)
    c = np.dot(vec, z)
    vx = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    return R


def align_coords_to_td(coords: np.ndarray, td_vector: list | np.ndarray) -> np.ndarray:
    """
    Center the molecule at the origin, then rotate so the transition
    dipole vector aligns with the Z axis.
    """
    coords = np.array(coords, dtype=float)
    centroid = coords.mean(axis=0)
    coords -= centroid
    R = rotation_matrix_align_to_z(np.asarray(td_vector))
    return (R @ coords.T).T


def suppress_mdtraj_warnings():
    """Suppress mdtraj PDB parsing warnings about charge fields."""
    warnings.filterwarnings("ignore", message=".*Could not parse charge.*")
    warnings.filterwarnings("ignore", module="mdtraj.formats.pdb")
