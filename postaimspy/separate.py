"""
postaimspy.separate — Trajectory separation from pySpawn / FMS90 simulation data.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from postaimspy.config import AIMSConfig
from postaimspy.utils import log, append_xyz, read_xyz

BOHR_TO_ANG = 0.529177249
BOHR_TO_ANG_PYSPAWN = 1.0 / 1.8897161646321


# ══════════════════════════════════════════════════════════════════════════════
# Shared: simulation directory discovery
# ══════════════════════════════════════════════════════════════════════════════
def _discover_sim_dirs(rootdir: Path, num_sims: int, prefix: str) -> list[tuple[int, Path]]:
    """
    Discover simulation directories under *rootdir*.

    Supports naming conventions:
      prefix=""   →  1/, 2/, 3/, ...
      prefix="ic" →  ic1/, ic2/, ic3/, ...
      prefix="IC" →  IC1/, IC2/, IC3/, ...

    Returns list of (sim_id, path) sorted by sim_id, limited to num_sims.
    """
    found = []

    # Strategy 1: if prefix is empty, try simple numbered dirs
    # Strategy 2: scan for dirs matching {prefix}{number}
    for d in rootdir.iterdir():
        if not d.is_dir():
            continue
        name = d.name

        # Skip our own output directories
        if name in ("separated_with_pop", "postaimspy_output", "aimspy_output"):
            continue

        # Try to extract the numeric sim ID
        if prefix:
            # Name must start with prefix (case-sensitive), remainder must be a number
            if name.startswith(prefix) and name[len(prefix):].isdigit():
                sim_id = int(name[len(prefix):])
                found.append((sim_id, d))
        else:
            # No prefix — name itself must be a number
            if name.isdigit():
                sim_id = int(name)
                found.append((sim_id, d))

    # Sort by sim_id and limit to num_sims
    found.sort(key=lambda x: x[0])
    if num_sims is not None and num_sims > 0:
        found = [f for f in found if f[0] <= num_sims]

    if not found:
        pattern = f"'{prefix}N'" if prefix else "'N' (where N is a number)"
        log.warning(
            "No simulation directories found matching pattern %s under %s",
            pattern, rootdir,
        )

    return found


# ══════════════════════════════════════════════════════════════════════════════
# pySpawn reader
# ══════════════════════════════════════════════════════════════════════════════
def _compute_tbf_populations(h5file) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qtimes = h5file["sim/quantum_time"][()][:, 0]
    trajqt = h5file["sim/num_traj_qm"][()].flatten()
    S = h5file["sim/S"][()]
    c = h5file["sim/qm_amplitudes"][()]
    ntraj = c.shape[1]
    ntimes = len(qtimes)
    pop = np.zeros((ntimes, ntraj))

    for i in range(ntimes):
        nt = int(trajqt[i])
        c_t = c[i, :nt]
        S_t = S[i, : nt * nt].reshape((nt, nt))
        for ist in range(nt):
            pop_ist = 0.0
            for ist2 in range(nt):
                pop_ist += np.real(
                    0.5 * (
                        np.dot(np.conj(c_t[ist]), np.dot(S_t[ist, ist2], c_t[ist2]))
                        + np.dot(np.conj(c_t[ist2]), np.dot(S_t[ist2, ist], c_t[ist]))
                    )
                )
            pop[i, ist] = pop_ist
    return qtimes, trajqt, pop


def separate_pyspawn(cfg: AIMSConfig) -> dict:
    import h5py

    sep = cfg.separate
    tmax = sep["tmax"]
    tstep = sep["tstep"]
    nsims = sep["num_sims"]
    pop_cutoff = sep.get("pop_cutoff", 0.0)
    states = sep.get("states", [0, 1])
    prefix = sep.get("sim_dir_prefix", "")
    name = Path(cfg.reference_xyz).stem
    sepdir = cfg.separated_dir
    rootdir = Path(cfg.work_dir)

    _initialise_bin_files(sepdir, name, states, tmax, tstep)

    sim_dirs = _discover_sim_dirs(rootdir, nsims, prefix)
    log.info("Found %d simulation directories to process.", len(sim_dirs))

    global_idx = {s: 1 for s in states}
    sim_counts = []
    sim_ids = []

    for sim_id, simdir in sim_dirs:
        sim_ids.append(sim_id)
        h5path = simdir / "sim.hdf5"
        if not h5path.exists():
            log.warning("Skipping sim %d — %s not found.", sim_id, h5path)
            sim_counts.append({s: 0 for s in states})
            continue

        log.info("Processing simulation %d: %s", sim_id, simdir)
        h5file = h5py.File(str(h5path), "r")

        atoms = np.char.decode(h5file["traj_00"].attrs["atoms"], encoding="ascii")
        labels = np.char.decode(h5file["sim"].attrs["labels"], encoding="ascii")
        qtimes, trajqt, pop = _compute_tbf_populations(h5file)

        cur_counts = {s: 0 for s in states}

        for traj_idx in range(len(labels)):
            curtraj = f"traj_{labels[traj_idx]}"
            state = int(h5file[curtraj].attrs["istate"])
            if state not in states:
                continue
            time_arr = h5file[curtraj]["time"][:, 0]
            pos = h5file[curtraj]["positions"][()]
            pos *= BOHR_TO_ANG_PYSPAWN

            for t_idx in range(len(time_arr)):
                t = time_arr[t_idx]
                if t not in qtimes:
                    continue
                qidx = int(np.where(qtimes == t)[0][0])
                if pop[qidx, traj_idx] <= pop_cutoff:
                    continue

                t_int = int(t)
                bin_start = (t_int // tstep) * tstep
                if bin_start >= tmax:
                    continue
                bin_end = bin_start + tstep

                xyz_file = sepdir / f"{name}_from_{bin_start}_to_{bin_end}_{state}.xyz"
                pop_file = sepdir / f"{name}_from_{bin_start}_to_{bin_end}_{state}_pop.txt"

                natoms = len(atoms)
                coords = np.array([
                    [pos[t_idx, 3 * a], pos[t_idx, 3 * a + 1], pos[t_idx, 3 * a + 2]]
                    for a in range(natoms)
                ])
                comment = (
                    f"{global_idx[state]}  sim no  {sim_id}  "
                    f"traj  {labels[traj_idx]}  time  {t}"
                )
                append_xyz(xyz_file, list(atoms), coords, comment=comment)

                with open(pop_file, "a") as pf:
                    pf.write(
                        f"pop= {pop[qidx, traj_idx]:8.4f} for {global_idx[state]} "
                        f"sim no {sim_id} traj {labels[traj_idx]} time {t}\n"
                    )

                global_idx[state] += 1
                cur_counts[state] += 1

        h5file.close()
        sim_counts.append(cur_counts)

    _write_separation_report(cfg, sim_counts, states, tmax, tstep, name, sepdir, sim_ids=sim_ids)
    return {"sim_counts": sim_counts}


# ══════════════════════════════════════════════════════════════════════════════
# FMS90 reader
# ══════════════════════════════════════════════════════════════════════════════
def separate_fms90(cfg: AIMSConfig) -> dict:
    sep = cfg.separate
    tmax = sep["tmax"]
    tstep = sep["tstep"]
    nsims = sep["num_sims"]
    pop_cutoff = sep.get("pop_cutoff", 0.0)
    states = sep.get("states", [0, 1])
    prefix = sep.get("sim_dir_prefix", "")
    name = Path(cfg.reference_xyz).stem
    sepdir = cfg.separated_dir
    rootdir = Path(cfg.work_dir)

    ref_atoms, _ = read_xyz(cfg.ref_xyz_path)
    natoms = len(ref_atoms)

    _initialise_bin_files(sepdir, name, states, tmax, tstep)

    fms_state_map = {1.0: 0, 2.0: 1}

    global_idx = {s: 1 for s in states}
    sim_counts = []
    sim_ids = []

    sim_dirs = _discover_sim_dirs(rootdir, nsims, prefix)
    log.info("Found %d simulation directories to process.", len(sim_dirs))

    for sim_id, simdir in sim_dirs:
        sim_ids.append(sim_id)
        cur_counts = {s: 0 for s in states}

        traj_files = sorted(simdir.glob("TrajDump.*"))
        if not traj_files:
            log.warning("No TrajDump files in %s — skipping.", simdir)
            sim_counts.append(cur_counts)
            continue

        for traj_file in traj_files:
            traj_label = traj_file.suffix.lstrip(".")
            log.info("Processing sim %d, trajectory %s", sim_id, traj_file.name)

            with open(traj_file) as fh:
                for line in fh:
                    parts = line.split()
                    if not parts or parts[0].startswith("#"):
                        continue

                    time_val = float(parts[0])
                    fms_state = float(parts[-1])
                    pop_val = float(parts[-2])

                    state = fms_state_map.get(fms_state)
                    if state is None:
                        log.error("Unexpected state %.1f at time %.1f in %s",
                                  fms_state, time_val, traj_file)
                        continue
                    if state not in states:
                        continue
                    if pop_val <= pop_cutoff:
                        continue

                    t_int = int(time_val)
                    bin_start = (t_int // tstep) * tstep
                    if bin_start >= tmax:
                        continue
                    bin_end = bin_start + tstep

                    coords = np.zeros((natoms, 3))
                    for j in range(natoms):
                        base = 3 * j + 1
                        coords[j, 0] = float(parts[base]) * BOHR_TO_ANG
                        coords[j, 1] = float(parts[base + 1]) * BOHR_TO_ANG
                        coords[j, 2] = float(parts[base + 2]) * BOHR_TO_ANG

                    xyz_file = sepdir / f"{name}_from_{bin_start}_to_{bin_end}_{state}.xyz"
                    pop_file = sepdir / f"{name}_from_{bin_start}_to_{bin_end}_{state}_pop.txt"

                    comment = (
                        f"{global_idx[state]}  sim no  {sim_id}  "
                        f"traj  {traj_label}  time  {time_val}"
                    )
                    append_xyz(xyz_file, ref_atoms, coords, comment=comment)

                    with open(pop_file, "a") as pf:
                        pf.write(
                            f"pop= {pop_val:8.4f} for {global_idx[state]} "
                            f"sim no {sim_id} traj {traj_label} time {time_val}\n"
                        )

                    global_idx[state] += 1
                    cur_counts[state] += 1

        sim_counts.append(cur_counts)

    _write_separation_report(cfg, sim_counts, states, tmax, tstep, name, sepdir, sim_ids=sim_ids)
    return {"sim_counts": sim_counts}


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════
def _initialise_bin_files(sepdir, name, states, tmax, tstep):
    for n in range(0, tmax, tstep):
        for s in states:
            (sepdir / f"{name}_from_{n}_to_{n + tstep}_{s}.xyz").touch()
            (sepdir / f"{name}_from_{n}_to_{n + tstep}_{s}_pop.txt").touch()


def _write_separation_report(cfg, sim_counts, states, tmax, tstep, name, sepdir, sim_ids=None):
    report_path = cfg.output_dir / "separation_results.out"
    with open(report_path, "w") as res:
        res.write("Distribution of geometries in each simulation.\n")
        res.write(" " + "-" * 50 + "\n")
        header = f"{'':>5s}{'Sim':>8s}"
        for s in states:
            header += f"{'S' + str(s):>12s}"
        header += f"{'Total':>12s}\n"
        res.write(header)
        res.write(" " + "-" * 50 + "\n")

        totals = {s: 0 for s in states}
        for j, sc in enumerate(sim_counts):
            label = sim_ids[j] if sim_ids else j + 1
            line = f"{label:5d}"
            row_total = 0
            for s in states:
                line += f"{sc[s]:12d}"
                totals[s] += sc[s]
                row_total += sc[s]
            line += f"{row_total:12d}\n"
            res.write(line)

        res.write(" " + "-" * 50 + "\n")
        grand = sum(totals.values())
        tline = f"{'Total':>5s}"
        for s in states:
            tline += f"{totals[s]:12d}"
        tline += f"{grand:12d}\n"
        res.write(tline)

        res.write("\n\nDistribution of geometries in each time range (a.u.).\n")
        res.write(" " + "-" * 56 + "\n")
        range_header = f"{'':>5s}{'Range':>20s}"
        for s in states:
            range_header += f"{'S' + str(s):>12s}"
        range_header += f"{'Total':>12s}\n"
        res.write(range_header)
        res.write(" " + "-" * 56 + "\n")

        range_totals = {s: 0 for s in states}
        for n in range(0, tmax, tstep):
            bin_label = f"{n}-{n + tstep}"
            line = f" {bin_label:>20s}"
            row_total = 0
            for s in states:
                fpath = sepdir / f"{name}_from_{n}_to_{n + tstep}_{s}.xyz"
                count = 0
                if fpath.exists():
                    count = fpath.read_text().count("sim no")
                line += f"{count:12d}"
                range_totals[s] += count
                row_total += count
            line += f"{row_total:12d}\n"
            res.write(line)

        res.write(" " + "-" * 56 + "\n")
        grand_range = sum(range_totals.values())
        tline = f"{'Total':>25s}"
        for s in states:
            tline += f"{range_totals[s]:12d}"
        tline += f"{grand_range:12d}\n"
        res.write(tline)

    log.info("Separation report written to %s", report_path)


# ══════════════════════════════════════════════════════════════════════════════
def run_separation(cfg: AIMSConfig) -> dict:
    if cfg.source_code == "pyspawn":
        return separate_pyspawn(cfg)
    elif cfg.source_code == "fms90":
        return separate_fms90(cfg)
    else:
        raise ValueError(f"Unknown source code: {cfg.source_code}")
