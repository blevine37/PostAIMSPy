"""
postaimspy.pipeline — Orchestrates the full post-processing workflow.

Steps (in order, each independently toggleable):
  1. separate  — extract trajectories from HDF5 into time-binned XYZ files
  2. cluster   — weighted k-means clustering using pairwise RMSD
  3. align     — transition-dipole alignment + RMSD minimization
"""

from __future__ import annotations

import time
from pathlib import Path

from postaimspy.config import AIMSConfig
from postaimspy.utils import log


def run_pipeline(cfg: AIMSConfig, steps: list[str] | None = None) -> dict:
    """Execute the requested pipeline steps."""
    all_steps = ["separate", "cluster", "align"]

    if steps is None:
        steps = [s for s in all_steps if getattr(cfg, s, {}).get("enabled", True)]

    # Log what we are about to do
    log.info("Steps to execute: %s", steps)
    for s in all_steps:
        enabled = getattr(cfg, s, {}).get("enabled", True)
        in_list = s in steps
        if enabled and not in_list:
            log.info("  Step '%s' is enabled but not in requested steps.", s)
        elif not enabled:
            log.info("  Step '%s' is disabled in config.", s)

    cfg.validate(steps)

    results = {}
    t0 = time.perf_counter()
    n = len(steps)

    _banner(cfg, steps)

    # ── Step 1: Trajectory separation ────────────────────────────────────
    if "separate" in steps:
        log.info("=" * 60)
        log.info("STEP %d / %d : Trajectory separation", steps.index("separate") + 1, n)
        log.info("=" * 60)
        from postaimspy.separate import run_separation
        results["separate"] = run_separation(cfg)
        log.info("Separation complete.\n")

    # ── Step 2: Weighted k-means clustering ──────────────────────────────
    if "cluster" in steps:
        log.info("=" * 60)
        log.info("STEP %d / %d : Weighted k-means clustering", steps.index("cluster") + 1, n)
        log.info("=" * 60)
        from postaimspy.cluster import run_clustering
        results["cluster"] = run_clustering(cfg)
        log.info("Clustering complete.\n")

    # ── Step 3: TD alignment + RMSD minimization ────────────────────────
    if "align" in steps:
        log.info("=" * 60)
        log.info("STEP %d / %d : Alignment & RMSD minimization", steps.index("align") + 1, n)
        log.info("=" * 60)
        from postaimspy.align import run_alignment
        results["align"] = run_alignment(cfg)
        log.info("Alignment complete.\n")

    elapsed = time.perf_counter() - t0
    log.info("Pipeline finished in %.1f s.", elapsed)
    return results


def _banner(cfg: AIMSConfig, steps: list[str]) -> None:
    log.info("")
    log.info("╔═══════════════════════════════════════════════════╗")
    log.info("║                PostAIMSPy — v1.0.0                ║")
    log.info("║  Post-processing for Ab-Initio Multiple Spawning  ║")
    log.info("║ A. Mehmood & Y. You · Levine Research Group · SBU ║")
    log.info("╚═══════════════════════════════════════════════════╝")
    log.info("")
    for line in cfg.summary().split("\n"):
        log.info("  %s", line)
    log.info("  Steps        : %s", " -> ".join(steps))
    log.info("  Output dir   : %s", cfg.output_dir)
    log.info("")
