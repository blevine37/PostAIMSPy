"""
postaimspy.align — Transition-dipole alignment and RMSD minimization.

1. Align the reference geometry so its transition dipole points along +Z.
2. RMSD-minimize every centroid geometry against that aligned reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from postaimspy.config import AIMSConfig
from postaimspy.rmsd import minimize_rmsd_and_rotate, compute_rmsd
from postaimspy.utils import (
    log,
    read_xyz,
    write_xyz,
    align_coords_to_td,
)


def _prepare_aligned_reference(cfg: AIMSConfig) -> tuple[list[str], np.ndarray, Path]:
    atoms, coords = read_xyz(cfg.ref_xyz_path)
    td = np.asarray(cfg.transition_dipole, dtype=float)
    aligned = align_coords_to_td(coords, td)

    out_path = cfg.aligned_dir / f"{cfg.ref_xyz_path.stem}_TD_aligned.xyz"
    write_xyz(out_path, atoms, aligned, comment="Reference aligned along TD Z-axis")
    log.info("Aligned reference written to %s", out_path)
    return atoms, aligned, out_path


def run_alignment(cfg: AIMSConfig) -> dict:
    """
    Full alignment pipeline:
      1. Align reference geometry along the transition dipole.
      2. For every centroid XYZ in the clustered directory,
         Kabsch-rotate it onto the aligned reference (RMSD minimization).
    """
    align_cfg = cfg.align
    method = align_cfg.get("rotation_method", "kabsch")
    cludir = cfg.clustered_dir
    aligndir = cfg.aligned_dir

    # Step 1 — prepare aligned reference
    ref_atoms, ref_coords, ref_path = _prepare_aligned_reference(cfg)

    # Step 2 — find all centroid files
    centroid_files = sorted(cludir.rglob("*_centroid_*.xyz"))
    if not centroid_files:
        log.warning("No centroid XYZ files found under %s — nothing to align.", cludir)
        return {}

    log.info("Found %d centroid files to align.", len(centroid_files))

    results = {}
    report_lines = []
    report_lines.append(
        f" {'File Name':>45s} {'Raw RMSD':>12s} {'Min. RMSD':>12s}\n"
    )
    report_lines.append(" " + "-" * 72 + "\n")

    for cf in centroid_files:
        tgt_atoms, tgt_coords = read_xyz(cf)

        # Raw RMSD (no rotation)
        raw = compute_rmsd(ref_atoms, ref_coords, tgt_atoms, tgt_coords, method="none")

        # Kabsch-minimized RMSD
        rotated, min_rmsd = minimize_rmsd_and_rotate(
            ref_atoms, ref_coords, tgt_atoms, tgt_coords, method=method,
        )

        # Determine output sub-directory (mirror clustered structure)
        rel = cf.relative_to(cludir)
        out_path = aligndir / rel.parent / f"aligned_{rel.name}"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Read original comment for provenance
        with open(cf) as fh:
            fh.readline()
            orig_comment = fh.readline().strip()

        write_xyz(
            out_path, tgt_atoms, rotated,
            comment=f"{orig_comment} | minRMSD={min_rmsd:.6f} wrt TD-aligned ref",
        )

        results[cf.name] = {"raw_rmsd": raw, "min_rmsd": min_rmsd}
        report_lines.append(
            f" {cf.stem:>45s} {raw:12.6f} {min_rmsd:12.6f}\n"
        )

    report_lines.append(" " + "-" * 72 + "\n")

    # Write report
    report_path = cfg.output_dir / "rmsd_minimization_results.out"
    with open(report_path, "w") as fh:
        fh.write(
            f"RMSD minimization w.r.t. transition-dipole-aligned reference.\n"
            f"Method: {method}\n\n"
        )
        fh.writelines(report_lines)

    log.info("RMSD minimization report -> %s", report_path)
    log.info("Aligned %d centroids -> %s", len(results), aligndir)
    return results
