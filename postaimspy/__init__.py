"""
PostAIMSPy — Post-processing for Ab-Initio Multiple Spawning in Python
=======================================================================

A unified toolkit for post-processing AIMS trajectory data from
pySpawn and FMS90 codes. Includes trajectory separation, weighted
k-means clustering, transition-dipole alignment, and RMSD minimization.

Authors : Arshad Mehmood & Ying You, Levine Research Group, SBU
"""

# ── Prevent OpenBLAS thread explosion on HPC nodes ──────────────────────
# Must be set BEFORE numpy/scipy/sklearn are imported anywhere.
import os as _os
if "OPENBLAS_NUM_THREADS" not in _os.environ:
    _os.environ["OPENBLAS_NUM_THREADS"] = "4"
if "MKL_NUM_THREADS" not in _os.environ:
    _os.environ["MKL_NUM_THREADS"] = "4"
if "OMP_NUM_THREADS" not in _os.environ:
    _os.environ["OMP_NUM_THREADS"] = "4"
# ────────────────────────────────────────────────────────────────────────

__version__ = "1.0.0"
__author__ = "Arshad Mehmood & Ying You"

from postaimspy.config import AIMSConfig
