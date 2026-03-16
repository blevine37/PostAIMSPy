"""
postaimspy.config — Configuration management via YAML input files.
"""

from __future__ import annotations

import copy
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------
_DEFAULTS = dict(
    # ── General ──────────────────────────────────────────────────────────
    project_name="aims_project",
    work_dir=".",
    reference_xyz=None,           # starting geometry (.xyz) — REQUIRED
    transition_dipole=None,       # [dx, dy, dz] — needed for alignment
    source_code="pyspawn",        # "pyspawn" or "fms90"

    # ── Trajectory separation ────────────────────────────────────────────
    separate=dict(
        enabled=True,
        tmax=25000,
        tstep=250,
        num_sims=50,
        pop_cutoff=0.0,
        states=[0, 1],
        sim_dir_prefix="",        # "" for 1/,2/,3/; "ic" for ic1/,ic2/; "IC" for IC1/,IC2/
    ),

    # ── Clustering ───────────────────────────────────────────────────────
    cluster=dict(
        enabled=True,
        num_clusters=80,
        n_init=200,
        max_iter=1000,
        tol=1e-10,
        random_seed=0,
        cluster_states=None,      # None = all states; or [0] or [1] etc.
    ),

    # ── Alignment & RMSD minimization ────────────────────────────────────
    align=dict(
        enabled=True,
        rotation_method="kabsch",   # kabsch | quaternion | none
        reorder=False,
        reorder_method="hungarian",
        ignore_hydrogen=False,
    ),
)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


@dataclass
class AIMSConfig:
    """Holder for the full pipeline configuration."""

    # General
    project_name: str = "aims_project"
    work_dir: str = "."
    reference_xyz: Optional[str] = None
    transition_dipole: Optional[list] = None
    source_code: str = "pyspawn"

    # Sub-configs stored as plain dicts for flexibility
    separate: dict = field(default_factory=lambda: copy.deepcopy(_DEFAULTS["separate"]))
    cluster: dict  = field(default_factory=lambda: copy.deepcopy(_DEFAULTS["cluster"]))
    align: dict    = field(default_factory=lambda: copy.deepcopy(_DEFAULTS["align"]))

    # ── Constructors ─────────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: str | Path) -> "AIMSConfig":
        """Load configuration from a YAML file, applying defaults."""
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        merged = _deep_merge(_DEFAULTS, raw)
        return cls(**{k: v for k, v in merged.items()})

    @classmethod
    def from_dict(cls, d: dict) -> "AIMSConfig":
        merged = _deep_merge(_DEFAULTS, d)
        return cls(**{k: v for k, v in merged.items()})

    # ── Derived paths ────────────────────────────────────────────────────
    @property
    def ref_xyz_path(self) -> Path:
        return Path(self.work_dir) / self.reference_xyz

    @property
    def output_dir(self) -> Path:
        p = Path(self.work_dir) / "postaimspy_output"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def separated_dir(self) -> Path:
        p = self.output_dir / "separated"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def clustered_dir(self) -> Path:
        p = self.output_dir / "clustered"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def aligned_dir(self) -> Path:
        p = self.output_dir / "aligned"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ── Validation ───────────────────────────────────────────────────────
    def validate(self, steps: list[str] | None = None):
        """Raise ValueError when the config is incomplete for *steps*."""
        steps = steps or ["separate", "cluster", "align"]
        if self.reference_xyz is None:
            raise ValueError("'reference_xyz' is required in the input file.")
        if not self.ref_xyz_path.exists():
            raise FileNotFoundError(
                f"Reference XYZ not found: {self.ref_xyz_path}"
            )
        if "align" in steps and self.align.get("enabled"):
            if self.transition_dipole is None:
                raise ValueError(
                    "'transition_dipole' [dx, dy, dz] is required for alignment."
                )
        allowed_codes = ("pyspawn", "fms90")
        if self.source_code not in allowed_codes:
            raise ValueError(
                f"'source_code' must be one of {allowed_codes}, "
                f"got '{self.source_code}'."
            )

    def summary(self) -> str:
        """Return a human-readable summary of the configuration."""
        lines = [
            f"Project      : {self.project_name}",
            f"Source code  : {self.source_code}",
            f"Reference    : {self.reference_xyz}",
            f"Work dir     : {self.work_dir}",
        ]
        if self.transition_dipole:
            lines.append(f"Trans. dipole: {self.transition_dipole}")
        if self.separate.get("enabled"):
            s = self.separate
            prefix = s.get("sim_dir_prefix", "")
            prefix_str = f", prefix='{prefix}'" if prefix else ""
            lines.append(
                f"Separation   : tmax={s['tmax']}, tstep={s['tstep']}, "
                f"nsims={s['num_sims']}, states={s['states']}{prefix_str}"
            )
        if self.cluster.get("enabled"):
            c = self.cluster
            cs = c.get("cluster_states")
            cs_str = f", states={cs}" if cs else ", states=all"
            lines.append(
                f"Clustering   : {c['num_clusters']} clusters, "
                f"seed={c['random_seed']}{cs_str}"
            )
        if self.align.get("enabled"):
            lines.append(f"Alignment    : method={self.align['rotation_method']}")
        return "\n".join(lines)
