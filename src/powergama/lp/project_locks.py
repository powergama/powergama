"""Optional day-ahead lock tables read from parquet artifacts.

This is project-specific rather than general-purpose: the tables are produced by a
downstream "prepare-rt" step and reach the model through attributes injected onto
the :class:`~powergama.GridData` object (``foreign_gen_lock_parquet`` and friends),
none of which appear in ``GridData.keys_powergama``. A grid built from plain CSV has
none of them, every path below is skipped, and every field stays at its default.

``_be_oriented_sign`` hardcodes a ``"BE"`` node-name prefix, which is the clearest
marker that this layer encodes one study's conventions. Whether it belongs in the
core library at all is an open question, deliberately not answered here -- moving
the loader into one place is what makes the question askable.

These are inputs to the real-time step only: a day-ahead run produces them, and
the RT simulation is pinned to them. They are loaded only when ``is_rt`` is set,
and applied by :func:`powergama.lp.rt.data.apply_da_locks`.
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# The GridData attributes the prepare-rt step sets, each a path to a parquet file.
LOCK_ATTRS = (
    "foreign_gen_lock_parquet",
    "foreign_consumer_lock_parquet",
    "border_ac_flow_lock_parquet",
    "border_dc_flow_lock_parquet",
    "border_ac_flow_bounds_parquet",
    "border_dc_flow_bounds_parquet",
    "da_storage_marginalprice_parquet",
)


def lock_attrs_set(grid) -> list[str]:
    """The lock attributes that are set (non-empty) on *grid*."""
    return [a for a in LOCK_ATTRS if getattr(grid, a, "")]


@dataclass
class DaLocks:
    """Day-ahead lock tables, plus the border branches they identify.

    The lock series are long-format, indexed by ``(timestep, indx)``. ``None``
    means the artifact was absent or unreadable, which is the ordinary case.
    """

    foreign_gen_lock: pd.Series | None = None
    foreign_cons_lock: pd.Series | None = None
    border_ac_flow_lock: pd.Series | None = None
    border_dc_flow_lock: pd.Series | None = None
    border_ac_flow_lb: pd.Series | None = None
    border_ac_flow_ub: pd.Series | None = None
    border_dc_flow_lb: pd.Series | None = None
    border_dc_flow_ub: pd.Series | None = None
    # {(timestep, storage_indx): value}, for storage deviation pricing.
    da_storage_marginalprice: dict[tuple[int, int], float] = field(default_factory=dict)
    # Border branches: those carrying lock data, intersected with the grid.
    idx_border_ac: set[int] = field(default_factory=set)
    idx_border_dc: set[int] = field(default_factory=set)
    border_ac_sign: dict[int, float] = field(default_factory=dict)
    border_dc_sign: dict[int, float] = field(default_factory=dict)


def _extract_loaded_branch_indices(series_obj):
    if series_obj is None:
        return set()
    idx_obj = getattr(series_obj, "index", None)
    if idx_obj is None:
        return set()
    if hasattr(idx_obj, "names") and "indx" in list(idx_obj.names):
        try:
            return {int(v) for v in idx_obj.get_level_values("indx").unique().tolist()}
        except Exception:
            return set()
    return set()


def _be_oriented_sign(branch_df, branch_idx: int) -> float:
    try:
        row = branch_df.loc[int(branch_idx)]
    except Exception:
        return 0.0
    node_from = str(row.get("node_from", ""))
    node_to = str(row.get("node_to", ""))
    if node_from.startswith("BE") and (not node_to.startswith("BE")):
        return 1.0
    if node_to.startswith("BE") and (not node_from.startswith("BE")):
        return -1.0
    return 0.0


def load_da_locks(grid) -> DaLocks:
    """Load optional foreign DA lock tables from parquet artifacts.

    Expected long format:
    - generators: timestep, indx, output
    - consumers: timestep, indx, demand
    """
    locks = DaLocks()

    gen_path = getattr(grid, "foreign_gen_lock_parquet", "")
    con_path = getattr(grid, "foreign_consumer_lock_parquet", "")
    ac_border_path = getattr(grid, "border_ac_flow_lock_parquet", "")
    dc_border_path = getattr(grid, "border_dc_flow_lock_parquet", "")
    ac_border_bounds_path = getattr(grid, "border_ac_flow_bounds_parquet", "")
    dc_border_bounds_path = getattr(grid, "border_dc_flow_bounds_parquet", "")

    if gen_path:
        p = Path(str(gen_path))
        if p.exists():
            try:
                gdf = pd.read_parquet(p)
                gdf["timestep"] = pd.to_numeric(gdf["timestep"], errors="coerce").fillna(-1).astype(int)
                gdf["indx"] = pd.to_numeric(gdf["indx"], errors="coerce").fillna(-1).astype(int)
                gdf["output"] = pd.to_numeric(gdf["output"], errors="coerce").fillna(0.0)
                locks.foreign_gen_lock = gdf.set_index(["timestep", "indx"])["output"]
            except Exception as exc:
                warnings.warn(f"Failed reading foreign generator DA lock parquet '{p}': {exc}", UserWarning)

    if con_path:
        p = Path(str(con_path))
        if p.exists():
            try:
                cdf = pd.read_parquet(p)
                cdf["timestep"] = pd.to_numeric(cdf["timestep"], errors="coerce").fillna(-1).astype(int)
                cdf["indx"] = pd.to_numeric(cdf["indx"], errors="coerce").fillna(-1).astype(int)
                cdf["demand"] = pd.to_numeric(cdf["demand"], errors="coerce").fillna(0.0)
                locks.foreign_cons_lock = cdf.set_index(["timestep", "indx"])["demand"]
            except Exception as exc:
                warnings.warn(f"Failed reading foreign consumer DA lock parquet '{p}': {exc}", UserWarning)

    if ac_border_path:
        p = Path(str(ac_border_path))
        if p.exists():
            try:
                adf = pd.read_parquet(p)
                adf["timestep"] = pd.to_numeric(adf["timestep"], errors="coerce").fillna(-1).astype(int)
                adf["indx"] = pd.to_numeric(adf["indx"], errors="coerce").fillna(-1).astype(int)
                adf["flow"] = pd.to_numeric(adf["flow"], errors="coerce").fillna(0.0)
                locks.border_ac_flow_lock = adf.set_index(["timestep", "indx"])["flow"]
            except Exception as exc:
                warnings.warn(f"Failed reading BE-border AC flow lock parquet '{p}': {exc}", UserWarning)

    if dc_border_path:
        p = Path(str(dc_border_path))
        if p.exists():
            try:
                ddf = pd.read_parquet(p)
                ddf["timestep"] = pd.to_numeric(ddf["timestep"], errors="coerce").fillna(-1).astype(int)
                ddf["indx"] = pd.to_numeric(ddf["indx"], errors="coerce").fillna(-1).astype(int)
                ddf["flow"] = pd.to_numeric(ddf["flow"], errors="coerce").fillna(0.0)
                locks.border_dc_flow_lock = ddf.set_index(["timestep", "indx"])["flow"]
            except Exception as exc:
                warnings.warn(f"Failed reading BE-border DC flow lock parquet '{p}': {exc}", UserWarning)

    if ac_border_bounds_path:
        p = Path(str(ac_border_bounds_path))
        if p.exists():
            try:
                adf = pd.read_parquet(p)
                adf["timestep"] = pd.to_numeric(adf["timestep"], errors="coerce").fillna(-1).astype(int)
                adf["indx"] = pd.to_numeric(adf["indx"], errors="coerce").fillna(-1).astype(int)
                adf["flow_lb"] = pd.to_numeric(adf["flow_lb"], errors="coerce")
                adf["flow_ub"] = pd.to_numeric(adf["flow_ub"], errors="coerce")
                adf_idx = adf.set_index(["timestep", "indx"])
                locks.border_ac_flow_lb = adf_idx["flow_lb"]
                locks.border_ac_flow_ub = adf_idx["flow_ub"]
            except Exception as exc:
                warnings.warn(f"Failed reading BE-border AC flow bounds parquet '{p}': {exc}", UserWarning)

    if dc_border_bounds_path:
        p = Path(str(dc_border_bounds_path))
        if p.exists():
            try:
                ddf = pd.read_parquet(p)
                ddf["timestep"] = pd.to_numeric(ddf["timestep"], errors="coerce").fillna(-1).astype(int)
                ddf["indx"] = pd.to_numeric(ddf["indx"], errors="coerce").fillna(-1).astype(int)
                ddf["flow_lb"] = pd.to_numeric(ddf["flow_lb"], errors="coerce")
                ddf["flow_ub"] = pd.to_numeric(ddf["flow_ub"], errors="coerce")
                ddf_idx = ddf.set_index(["timestep", "indx"])
                locks.border_dc_flow_lb = ddf_idx["flow_lb"]
                locks.border_dc_flow_ub = ddf_idx["flow_ub"]
            except Exception as exc:
                warnings.warn(f"Failed reading BE-border DC flow bounds parquet '{p}': {exc}", UserWarning)

    da_storage_marginalprice_path = getattr(grid, "da_storage_marginalprice_parquet", "")
    if da_storage_marginalprice_path:
        p = Path(str(da_storage_marginalprice_path))
        if p.exists():
            try:
                sdf = pd.read_parquet(p)
                sdf["timestep"] = pd.to_numeric(sdf["timestep"], errors="coerce").fillna(-1).astype(int)
                sdf["indx"] = pd.to_numeric(sdf["indx"], errors="coerce").fillna(-1).astype(int)
                sdf["marginalprice"] = pd.to_numeric(sdf["marginalprice"], errors="coerce").fillna(0.0)
                locks.da_storage_marginalprice = {
                    (int(row["timestep"]), int(row["indx"])): float(row["marginalprice"]) for _, row in sdf.iterrows()
                }
            except Exception as exc:
                warnings.warn(f"Failed reading DA storage marginalprice parquet '{p}': {exc}", UserWarning)

    # Border branches: branches that cross between different areas (used for border-flow locking).
    # Identified by the presence of border flow lock data on the grid object (set externally).
    # The sign convention: +1 if branch carries flow away from area_from, -1 otherwise.
    loaded_ac_idx = set()
    loaded_ac_idx |= _extract_loaded_branch_indices(locks.border_ac_flow_lock)
    loaded_ac_idx |= _extract_loaded_branch_indices(locks.border_ac_flow_lb)
    loaded_ac_idx |= _extract_loaded_branch_indices(locks.border_ac_flow_ub)
    locks.idx_border_ac = {int(i) for i in loaded_ac_idx if int(i) in set(grid.branch.index.tolist())}
    locks.border_ac_sign = {int(i): _be_oriented_sign(grid.branch, int(i)) for i in locks.idx_border_ac}

    loaded_dc_idx = set()
    loaded_dc_idx |= _extract_loaded_branch_indices(locks.border_dc_flow_lock)
    loaded_dc_idx |= _extract_loaded_branch_indices(locks.border_dc_flow_lb)
    loaded_dc_idx |= _extract_loaded_branch_indices(locks.border_dc_flow_ub)
    locks.idx_border_dc = {int(i) for i in loaded_dc_idx if int(i) in set(grid.dcbranch.index.tolist())}
    locks.border_dc_sign = {int(i): _be_oriented_sign(grid.dcbranch, int(i)) for i in locks.idx_border_dc}

    return locks
