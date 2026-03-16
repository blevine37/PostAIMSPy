"""
postaimspy.cluster — Population-weighted k-means clustering using pairwise RMSD.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from postaimspy.config import AIMSConfig
from postaimspy.rmsd import centroid, kabsch_rmsd
from postaimspy.utils import (
    log,
    read_xyz_multiframe,
    write_xyz,
    xyz_to_pdb,
    suppress_mdtraj_warnings,
)


# ══════════════════════════════════════════════════════════════════════════════
# RMSD matrix builders
# ══════════════════════════════════════════════════════════════════════════════
def _rmsd_matrix_numpy(frames: list[np.ndarray]) -> np.ndarray:
    """Build a full pairwise RMSD matrix with pure numpy (Kabsch)."""
    n = len(frames)
    mat = np.zeros((n, n))
    centred = [f - centroid(f) for f in frames]
    for i in range(n):
        for j in range(i + 1, n):
            val = kabsch_rmsd(centred[i], centred[j])
            mat[i, j] = val
            mat[j, i] = val
    return mat


def _rmsd_matrix_mdtraj(xyz_path: str, pdb_path: str) -> np.ndarray:
    """Build pairwise RMSD matrix using mdtraj (faster for large sets)."""
    suppress_mdtraj_warnings()
    import mdtraj as md

    traj = md.load_xyz(xyz_path, top=pdb_path)
    n = traj.n_frames
    mat = np.empty((n, n))
    for i in range(n):
        mat[i] = md.rmsd(traj, traj, i)
    return mat  # in nm


def build_rmsd_matrix(
    xyz_path: str | Path,
    pdb_path: str | Path | None = None,
    use_mdtraj: bool = True,
) -> np.ndarray:
    """
    Build pairwise RMSD matrix.  Tries mdtraj first; falls back to numpy.
    Returns matrix in **Angstrom**.
    """
    xyz_path = str(xyz_path)
    if use_mdtraj:
        try:
            if pdb_path is None:
                raise FileNotFoundError
            suppress_mdtraj_warnings()
            mat = _rmsd_matrix_mdtraj(xyz_path, str(pdb_path))
            return mat * 10.0  # nm -> Angstrom
        except Exception as exc:
            log.info("mdtraj unavailable or failed (%s); using numpy fallback.", exc)

    frames_data = read_xyz_multiframe(xyz_path)
    frames = [f[1] for f in frames_data]
    return _rmsd_matrix_numpy(frames)


# ══════════════════════════════════════════════════════════════════════════════
# Weighted k-means clustering
# ══════════════════════════════════════════════════════════════════════════════
def weighted_kmeans_cluster(
    rmsds: np.ndarray,
    populations: np.ndarray,
    num_clusters: int = 80,
    n_init: int = 200,
    max_iter: int = 1000,
    tol: float = 1e-10,
    random_state: int = 0,
) -> dict:
    """
    Run population-weighted k-means on a pairwise RMSD matrix.

    Returns dict with: labels, centroid_indices, centroid_pops, distances
    """
    kmeans = KMeans(
        n_clusters=num_clusters,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    kmeans.fit(rmsds, sample_weight=populations)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Find the data point closest to each centroid *within* its cluster
    centroid_indices = []
    distances = []
    for k in range(num_clusters):
        members = np.where(labels == k)[0]
        if len(members) == 0:
            centroid_indices.append(-1)
            distances.append(np.inf)
            continue
        member_rows = rmsds[members]
        closest, dist = pairwise_distances_argmin_min(
            [centers[k]], member_rows
        )
        centroid_indices.append(int(members[closest[0]]))
        distances.append(float(dist[0]))

    # Cluster populations = sum of member populations
    centroid_pops = np.zeros(num_clusters)
    for k in range(num_clusters):
        centroid_pops[k] = populations[labels == k].sum()

    return {
        "labels": labels,
        "centroid_indices": centroid_indices,
        "centroid_pops": centroid_pops,
        "distances": distances,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline entry point
# ══════════════════════════════════════════════════════════════════════════════
def run_clustering(cfg: AIMSConfig, xyz_files: list[Path] | None = None) -> dict:
    """
    Cluster every time-bin XYZ file found in the separated directory.
    """
    clu_cfg = cfg.cluster
    num_clusters = clu_cfg["num_clusters"]
    random_seed = clu_cfg.get("random_seed", 0)
    cluster_states = clu_cfg.get("cluster_states", None)
    sepdir = cfg.separated_dir
    cludir = cfg.clustered_dir
    ref_xyz = cfg.ref_xyz_path

    # Auto-generate PDB topology from reference
    pdb_path = cfg.output_dir / (ref_xyz.stem + ".pdb")
    if not pdb_path.exists():
        xyz_to_pdb(ref_xyz, pdb_path)

    if xyz_files is None:
        xyz_files = sorted(sepdir.glob("*.xyz"))

    # Filter by cluster_states if specified
    if cluster_states is not None:
        filtered = []
        for f in xyz_files:
            # File names: {name}_from_{start}_to_{end}_{state}.xyz
            stem = f.stem
            # The last character(s) before .xyz is the state index
            try:
                file_state = int(stem.rsplit("_", 1)[-1])
                if file_state in cluster_states:
                    filtered.append(f)
            except ValueError:
                filtered.append(f)  # include files that don't match pattern
        log.info(
            "Filtering for states %s: %d / %d XYZ files selected.",
            cluster_states, len(filtered), len(xyz_files),
        )
        xyz_files = filtered

    all_results = {}
    skipped = 0
    clustered = 0
    for xyzf in xyz_files:
        pop_file = xyzf.with_name(xyzf.stem + "_pop.txt")
        if not pop_file.exists():
            log.warning("WARNING: No population file found for %s — skipping.", xyzf.name)
            skipped += 1
            continue

        frames = read_xyz_multiframe(xyzf)

        # Skip empty or too-small files
        if len(frames) == 0:
            log.debug("Empty XYZ file %s — skipping.", xyzf.name)
            skipped += 1
            continue

        if len(frames) < 2:
            log.warning(
                "WARNING: Only %d frame in %s — too few to cluster, skipping.",
                len(frames), xyzf.name,
            )
            skipped += 1
            continue

        # Read populations
        pops = _read_pop_file(pop_file, len(frames))
        if len(pops) == 0:
            log.warning("WARNING: Empty population file for %s — skipping.", xyzf.name)
            skipped += 1
            continue

        # Adjust cluster count if more clusters than frames
        actual_k = min(num_clusters, len(frames))
        if actual_k < num_clusters:
            log.info(
                "%s: %d frames < %d requested clusters — using %d clusters.",
                xyzf.name, len(frames), num_clusters, actual_k,
            )

        log.info("Clustering %s  (%d frames, %d clusters)", xyzf.name, len(frames), actual_k)

        # Build RMSD matrix
        rmsds = build_rmsd_matrix(xyzf, pdb_path, use_mdtraj=True)

        if rmsds.size == 0:
            log.warning("WARNING: RMSD matrix is empty for %s — skipping.", xyzf.name)
            skipped += 1
            continue

        log.info("Max pairwise RMSD: %.4f Ang", np.max(rmsds))

        # Cluster
        result = weighted_kmeans_cluster(
            rmsds, pops,
            num_clusters=actual_k,
            n_init=clu_cfg["n_init"],
            max_iter=clu_cfg["max_iter"],
            tol=clu_cfg["tol"],
            random_state=random_seed,
        )

        # ── Write outputs ────────────────────────────────────────────────
        tag = xyzf.stem
        outdir = cludir / tag
        outdir.mkdir(parents=True, exist_ok=True)

        # clusters.dat
        np.savetxt(outdir / "clusters.dat", result["labels"] + 1, fmt="%d")

        # centroid xyz files
        for j, cidx in enumerate(result["centroid_indices"]):
            if cidx < 0:
                continue
            atoms_j, coords_j, _ = frames[cidx]
            comment = (
                f"Cluster {j + 1} centroid "
                f"pop={result['centroid_pops'][j]:.4f} "
                f"for {tag}"
            )
            write_xyz(outdir / f"{tag}_centroid_{j + 1}.xyz", atoms_j, coords_j, comment)

        # summary
        _write_cluster_summary(outdir / "summary.out", result, tag)
        all_results[tag] = result
        clustered += 1
        log.info("  -> centroids written to %s", outdir)

    log.info(
        "Clustering complete: %d files clustered, %d skipped.",
        clustered, skipped,
    )
    return all_results


def _read_pop_file(path: Path, expected: int) -> np.ndarray:
    """Parse a *_pop.txt file into a numpy array of populations."""
    pops = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    pops.append(float(parts[1]))
                except ValueError:
                    continue
    pops = np.array(pops)
    if len(pops) != expected:
        log.warning(
            "WARNING: Population entries (%d) != frames (%d) in %s",
            len(pops), expected, path,
        )
    return pops


def _write_cluster_summary(path: Path, result: dict, tag: str) -> None:
    labels = result["labels"]
    centind = result["centroid_indices"]
    cenpop = result["centroid_pops"]
    dists = result["distances"]
    counts = np.bincount(labels)

    with open(path, "w") as f:
        f.write(f"Weighted k-means clustering summary for {tag}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Cluster member counts:\n")
        f.write(f"{'Cluster':>8s} {'Count':>8s} {'Weight':>12s}\n")
        f.write("-" * 30 + "\n")
        for i in range(len(counts)):
            f.write(f"{i + 1:8d} {counts[i]:8d} {cenpop[i]:12.4f}\n")

        f.write(f"\n{'Centroid':>8s} {'Frame':>8s} {'Distance':>12s}\n")
        f.write("-" * 30 + "\n")
        for i in range(len(centind)):
            f.write(f"{i + 1:8d} {centind[i] + 1:8d} {dists[i]:12.4f}\n")

        f.write(f"\nTotal frames clustered: {len(labels)}\n")
