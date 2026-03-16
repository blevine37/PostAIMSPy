#!/usr/bin/env python3
"""
postaimspy — command-line interface.

Usage
-----
  postaimspy run input.yaml
  postaimspy run input.yaml --steps separate cluster
  postaimspy init
  postaimspy pdb molecule.xyz
  postaimspy rmsd ref.xyz target.xyz --method kabsch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="postaimspy",
        description=(
            "PostAIMSPy — Post-processing for Ab-Initio Multiple Spawning.\n"
            "Unified toolkit for trajectory separation, clustering, "
            "alignment, and RMSD minimization."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version="postaimspy 1.0.0")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── run ───────────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Run the processing pipeline")
    p_run.add_argument("input", type=str, help="Path to YAML input file")
    p_run.add_argument(
        "--steps", nargs="+",
        choices=["separate", "cluster", "align"],
        default=None,
        help="Run only these steps (default: all enabled steps)",
    )

    # ── init ──────────────────────────────────────────────────────────────
    p_init = sub.add_parser("init", help="Generate a template input.yaml")
    p_init.add_argument("-o", "--output", default="input.yaml", help="Output filename")

    # ── pdb ───────────────────────────────────────────────────────────────
    p_pdb = sub.add_parser("pdb", help="Generate PDB topology from XYZ")
    p_pdb.add_argument("xyz", help="Input XYZ file")
    p_pdb.add_argument("-o", "--output", default=None, help="Output PDB path")

    # ── rmsd ──────────────────────────────────────────────────────────────
    p_rmsd = sub.add_parser("rmsd", help="Compute RMSD between two structures")
    p_rmsd.add_argument("ref", help="Reference XYZ file")
    p_rmsd.add_argument("target", help="Target XYZ file")
    p_rmsd.add_argument("--method", default="kabsch", choices=["kabsch", "quaternion", "none"])
    p_rmsd.add_argument("--reorder", action="store_true")
    p_rmsd.add_argument("--ignore-hydrogen", action="store_true")
    p_rmsd.add_argument("--print", dest="print_rotated", action="store_true",
                        help="Print rotated structure to stdout")

    return parser


# ══════════════════════════════════════════════════════════════════════════════
# Command handlers
# ══════════════════════════════════════════════════════════════════════════════
def cmd_run(args):
    from postaimspy.config import AIMSConfig
    from postaimspy.pipeline import run_pipeline

    cfg = AIMSConfig.from_yaml(args.input)
    run_pipeline(cfg, steps=args.steps)


def cmd_init(args):
    template = """\
# ══════════════════════════════════════════════════════════════
# PostAIMSPy v1.0 — Input File
# Post-processing for Ab-Initio Multiple Spawning in Python
# ══════════════════════════════════════════════════════════════
#
# Edit the values below for your system.
# Run with:  postaimspy run input.yaml
# Run a single step:  postaimspy run input.yaml --steps separate
#

project_name: my_molecule
work_dir: .                         # root directory containing simulation folders (1/, 2/, ...)
reference_xyz: molecule.xyz         # starting geometry used for initial conditions (Angstrom)
transition_dipole: [0.0, 0.0, 1.0]  # [dx, dy, dz] transition dipole vector
source_code: pyspawn                # pyspawn | fms90

# ── Trajectory separation ────────────────────────────────────
separate:
  enabled: true
  tmax: 25000          # max simulation time (a.u.)
  tstep: 250           # time-bin width (a.u.)
  num_sims: 50         # number of initial conditions / simulations
  pop_cutoff: 0.0      # discard TBFs with population <= this value
  states: [0, 1]       # electronic states to extract (0=S0, 1=S1)
  sim_dir_prefix: ""   # "" for 1/,2/,3/; "ic" for ic1/,ic2/; "IC" for IC1/,IC2/

# ── Weighted k-means clustering ──────────────────────────────
cluster:
  enabled: true
  num_clusters: 80     # number of representative structures per bin
  n_init: 200          # KMeans random restarts (higher = more robust)
  max_iter: 1000
  tol: 1.0e-10
  random_seed: 0       # random seed for reproducibility
  cluster_states: null  # null = cluster all states; or [0] for S0 only, [1] for S1 only

# ── Alignment & RMSD minimization ───────────────────────────
align:
  enabled: true
  rotation_method: kabsch    # kabsch | quaternion | none
  reorder: false             # reorder atoms via Hungarian algorithm
  reorder_method: hungarian
  ignore_hydrogen: false     # exclude H atoms from RMSD calculation

# ══════════════════════════════════════════════════════════════
# FMS90 EXAMPLE — uncomment and adjust if using FMS90 data:
# ══════════════════════════════════════════════════════════════
# source_code: fms90
# separate:
#   enabled: true
#   tmax: 88000
#   tstep: 1000
#   num_sims: 50
#   pop_cutoff: 0.0
#   states: [0, 1]
#   sim_dir_prefix: ""    # "" for 1/,2/; "ic" for ic1/,ic2/; "IC" for IC1/,IC2/
#   states: [0, 1]
"""
    out = Path(args.output)
    out.write_text(template)
    print(f"Template input file written to {out}")
    print("Edit the values and run:  postaimspy run input.yaml")


def cmd_pdb(args):
    from postaimspy.utils import xyz_to_pdb

    xyz = Path(args.xyz)
    pdb = Path(args.output) if args.output else xyz.with_suffix(".pdb")
    xyz_to_pdb(xyz, pdb)
    print(f"PDB written to {pdb}")


def cmd_rmsd(args):
    from postaimspy.utils import read_xyz
    from postaimspy.rmsd import compute_rmsd, minimize_rmsd_and_rotate

    ref_atoms, ref_coords = read_xyz(args.ref)
    tgt_atoms, tgt_coords = read_xyz(args.target)

    if args.print_rotated:
        rotated, val = minimize_rmsd_and_rotate(
            ref_atoms, ref_coords, tgt_atoms, tgt_coords, method=args.method,
        )
        print(f"# Minimized RMSD = {val:.6f} Ang")
        for sym, (x, y, z) in zip(tgt_atoms, rotated):
            print(f"{sym:>2s} {x:17.10f} {y:17.10f} {z:17.10f}")
    else:
        val = compute_rmsd(
            ref_atoms, ref_coords, tgt_atoms, tgt_coords,
            method=args.method,
            reorder=args.reorder,
            ignore_hydrogen=args.ignore_hydrogen,
        )
        print(f"{val:.10f}")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "run": cmd_run,
        "init": cmd_init,
        "pdb": cmd_pdb,
        "rmsd": cmd_rmsd,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
