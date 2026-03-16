# PostAIMSPy — Post-processing for Ab-Initio Multiple Spawning in Python

A unified, modular toolkit for post-processing trajectory data from **pySpawn** and **FMS90** ab-initio multiple spawning (AIMS) simulations.

**Authors:** Arshad Mehmood & Ying You, Levine Research Group, Stony Brook University

---

## Features

| Step | What it does |
|------|-------------|
| **Separate** | Reads HDF5 (pySpawn) or TrajDump (FMS90) data, computes TBF populations, bins geometries by electronic state and time window |
| **Cluster** | Population-weighted k-means clustering on pairwise RMSD distance matrices. Supports state-selective clustering. |
| **Align** | Aligns the reference geometry along the transition dipole (Z-axis), then Kabsch-rotates every centroid to minimize RMSD |

Each step can be run independently or as part of the full pipeline.

---

## Installation

```bash
cd postaimspy/
bash install.sh
```

Or manually:
```bash
pip install .
pip install mdtraj matplotlib   # optional but recommended
```

**Requirements:** Python >= 3.9, NumPy, SciPy, scikit-learn, h5py, PyYAML.

---

## Thread Control (HPC)

PostAIMSPy automatically sets `OPENBLAS_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, and
`OMP_NUM_THREADS=4` to prevent memory crashes on HPC nodes with many cores.

To override, set the environment variable **before** running:

```bash
export OPENBLAS_NUM_THREADS=8   # your value takes priority
export OMP_NUM_THREADS=8
postaimspy run input.yaml
```

---

## Quick Start

```bash
postaimspy init                           # generate template input.yaml
postaimspy run input.yaml                 # run full pipeline
postaimspy run input.yaml --steps separate        # only separation
postaimspy run input.yaml --steps cluster         # only clustering
postaimspy run input.yaml --steps align           # only alignment
postaimspy run input.yaml --steps separate cluster  # two steps
```

---

## Input File Reference

```yaml
project_name: cracac3
work_dir: /path/to/simulations
reference_xyz: geometry.xyz
transition_dipole: [0.1, 0.2, 0.9]
source_code: pyspawn              # pyspawn | fms90

separate:
  enabled: true
  tmax: 25000
  tstep: 250
  num_sims: 50
  pop_cutoff: 0.0
  states: [0, 1]
  sim_dir_prefix: ""              # "" for 1/,2/,3/; "ic" for ic1/,ic2/; "IC" for IC1/,IC2/

cluster:
  enabled: true
  num_clusters: 80
  n_init: 200
  max_iter: 1000
  tol: 1.0e-10
  random_seed: 42             # for reproducibility
  cluster_states: [1]         # null = all states; [0] = S0 only; [1] = S1 only

align:
  enabled: true
  rotation_method: kabsch
  reorder: false
  ignore_hydrogen: false
```

### Simulation Directory Naming

PostAIMSPy supports different naming conventions for simulation directories
via the `sim_dir_prefix` option. Combined with `num_sims`, it controls which
directories are processed:

| `sim_dir_prefix` | `num_sims` | Directories processed |
|---|---|---|
| `""` (default) | 50 | `1/`, `2/`, ..., `50/` |
| `"ic"` | 50 | `ic1/`, `ic2/`, ..., `ic50/` |
| `"IC"` | 30 | `IC1/`, `IC2/`, ..., `IC30/` |
| `"run_"` | 10 | `run_1/`, `run_2/`, ..., `run_10/` |

The prefix is **case-sensitive**. `num_sims` limits processing to directories
where the numeric part is <= `num_sims`.

---

## Utility Commands

```bash
postaimspy pdb molecule.xyz                        # generate PDB from XYZ
postaimspy rmsd ref.xyz target.xyz                 # Kabsch RMSD
postaimspy rmsd ref.xyz target.xyz --method quaternion --reorder
postaimspy rmsd ref.xyz target.xyz --print         # print rotated structure
```

---

## Output Structure

```
postaimspy_output/
├── separated/
│   ├── mol_from_0_to_250_0.xyz
│   ├── mol_from_0_to_250_0_pop.txt
│   ├── mol_from_0_to_250_1.xyz
│   └── ...
├── clustered/
│   ├── mol_from_0_to_250_1/
│   │   ├── clusters.dat
│   │   ├── summary.out
│   │   ├── mol_..._centroid_1.xyz
│   │   └── ...
│   └── ...
├── aligned/
│   ├── mol_TD_aligned.xyz
│   └── mol_from_0_to_250_1/
│       ├── aligned_mol_..._centroid_1.xyz
│       └── ...
├── separation_results.out
├── rmsd_minimization_results.out
└── mol.pdb
```

---
