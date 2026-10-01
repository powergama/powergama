from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pyproj
import shapely
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from powergama import GridData


def add_line_labels(gdf, col, ax, **kwargs):
    fmt = kwargs.pop("number_format", ".0f")
    for idx, row in gdf.iterrows():
        # rp = row.geometry.interpolate((np.random.rand() * 0.4 + 0.3), normalized=True)
        rp = row.geometry.interpolate(0.5, normalized=True)
        if col == "index":
            label = str(idx)
        else:
            label = f"{row[col]:{fmt}}"
        txt = ax.text(
            rp.x,
            rp.y,
            label,
            # bbox=dict(facecolor='white', edgecolor='none', pad=2),
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            **kwargs,
        )
        txt.set_clip_on(True)
        txt.set_clip_box(ax.bbox)


def plot_map2(
    pg_data: GridData,
    pg_res: GridData = None,
    nodetype=None,
    branchtype=None,
    shapes: dict = {},
    plot_options: dict = {},
    ax=None,
    proj="epsg:3035",
    plot_extent=None,
    plot_gentypes: list = None,
    legend_args: dict = None,
):
    """Plot grid using geopandas (for print)"""

    if nodetype not in [
        None,
        "area",
        "type",
        "nodalprice",
        "loadshedding",
        "curtailed_res",
    ]:
        raise ValueError(f"Unknown nodetype: {nodetype}")
    if branchtype not in [
        None,
        "type",
        "capacity",
        "flow",
        "utilisation",
        "sensitivity",
    ]:
        raise ValueError(f"Unknown branchtype: {branchtype}")

    node_options = plot_options.get("node", {})
    branch_options = plot_options.get("branch", {}).copy()
    dcbranch_options = plot_options.get("dcbranch", {}).copy()
    gen_options = plot_options.get("generator", {}).copy()

    if ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
    ax.set_facecolor("white")

    if legend_args is None:
        # default values
        legend_args = {
            "bbox_to_anchor": (1, 0),
            "loc": "upper right",
            "frameon": False,
            "ncols": 4,
        }

    if plot_extent is None:
        plot_extent = {
            "lat": (pg_data.node["lat"].min() - 1, pg_data.node["lat"].max() + 1),
            "lon": (pg_data.node["lon"].min() - 1, pg_data.node["lon"].max() + 1),
        }
    p1 = pyproj.Proj(proj, preserve_units=False)
    xlim, ylim = p1(plot_extent["lon"], plot_extent["lat"])
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set(xlabel=None, ylabel=None)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Plot background shapes
    for shape_dict in shapes:
        shape = shape_dict["shape"]
        options = shape_dict["plot_options"]
        shape.to_crs(proj).plot(ax=ax, **options)

    branch = pg_data.branch.copy()
    dcbranch = pg_data.dcbranch.copy()
    node = pg_data.node.copy()
    generator = pg_data.generator.copy()
    # consumer = pg_data.consumer.copy()

    # NODE:
    gdf_nodes = geopandas.GeoDataFrame(
        node,
        geometry=geopandas.points_from_xy(node["lon"], node["lat"]),
        crs="EPSG:4326",
    )
    # BRANCH:
    branch["index"] = branch.index
    gdf_edges = branch.merge(
        gdf_nodes[["id", "lat", "lon", "geometry"]],
        how="left",
        left_on="node_from",
        right_on="id",
    ).merge(
        gdf_nodes[["id", "lat", "lon", "geometry"]],
        how="left",
        left_on="node_to",
        right_on="id",
    )
    gdf_edges["geometry"] = gdf_edges.apply(
        lambda r: shapely.LineString([(r.geometry_x.x, r.geometry_x.y), (r.geometry_y.x, r.geometry_y.y)]),
        axis=1,
    )
    gdf_edges = geopandas.GeoDataFrame(gdf_edges, geometry="geometry", crs="EPSG:4326")
    gdf_edges = gdf_edges.set_index("index")

    # DC BRANCH:
    dcbranch["index"] = dcbranch.index
    gdf_dcedges = dcbranch.merge(
        gdf_nodes[["id", "lat", "lon", "geometry"]],
        how="left",
        left_on="node_from",
        right_on="id",
    ).merge(
        gdf_nodes[["id", "lat", "lon", "geometry"]],
        how="left",
        left_on="node_to",
        right_on="id",
    )
    if not gdf_dcedges.empty:
        gdf_dcedges["geometry"] = gdf_dcedges.apply(
            lambda r: shapely.LineString([(r.geometry_x.x, r.geometry_x.y), (r.geometry_y.x, r.geometry_y.y)]),
            axis=1,
        )
    else:
        gdf_dcedges["geometry"] = None
    gdf_dcedges = geopandas.GeoDataFrame(gdf_dcedges, geometry="geometry", crs="EPSG:4326")
    gdf_dcedges = gdf_dcedges.set_index("index")

    # GEN: Use generator lat/lon when available, otherwise node lat/lon
    gen = generator.merge(
        node.rename(columns={"lat": "node_lat", "lon": "node_lon"})[["id", "node_lat", "node_lon"]],
        how="left",
        left_on="node",
        right_on="id",
    )
    # use gen["gen_lat"] if the column exists, otherwise use gen["node_lat"], replace missing values with node_lat
    gen["lat"] = gen.get("gen_lat", gen["node_lat"]).fillna(gen["node_lat"])
    gen["lon"] = gen.get("gen_lon", gen["node_lon"]).fillna(gen["node_lon"])
    gdf_generators = geopandas.GeoDataFrame(
        gen, geometry=geopandas.points_from_xy(gen["lon"], gen["lat"]), crs="EPSG:4326"
    )

    if "width_col" in branch_options:
        width_col = branch_options.pop("width_col", 1.0)
        linewidth = (gdf_edges[width_col[0]] / width_col[1]).clip(upper=width_col[2])
        branch_options["linewidth"] = linewidth
    if "width_col" in dcbranch_options:
        width_col = dcbranch_options.pop("width_col", 1.0)
        linewidth_dc = (gdf_dcedges[width_col[0]] / width_col[1]).clip(upper=width_col[2])
        dcbranch_options["linewidth"] = linewidth_dc
    # Which columns to use for branch labels:
    branch_annotation = branch_options.pop("annotation", None)
    dcbranch_annotation = dcbranch_options.pop("annotation", None)

    if not gdf_edges.empty:
        gdf_edges.to_crs(proj).plot(ax=ax, **branch_options)
    if not gdf_dcedges.empty:
        gdf_dcedges.to_crs(proj).plot(ax=ax, **dcbranch_options)
    if not gdf_nodes.empty:
        gdf_nodes.to_crs(proj).plot(ax=ax, **node_options)
    if plot_gentypes:
        if plot_gentypes == "all":
            gdf_gen_plot = gdf_generators
        else:
            m_gen_keep = gdf_generators["type"].isin(plot_gentypes)
            gdf_gen_plot = gdf_generators[m_gen_keep]
        if not gdf_gen_plot.empty:
            gdf_gen_plot.to_crs(proj).plot(ax=ax, **gen_options)
            gdf_gen_edges = gdf_gen_plot.copy()

            m_has_coords = ~gdf_gen_edges[["gen_lon", "gen_lat"]].isna().any(axis=1)
            gdf_gen_edges = gdf_gen_edges[m_has_coords]
            gdf_gen_edges["geometry_x"] = geopandas.points_from_xy(gdf_gen_edges["gen_lon"], gdf_gen_edges["gen_lat"])
            gdf_gen_edges["geometry_y"] = geopandas.points_from_xy(gdf_gen_edges["node_lon"], gdf_gen_edges["node_lat"])
            gdf_gen_edges["geometry"] = gdf_gen_edges.apply(
                lambda r: shapely.LineString([(r.geometry_x.x, r.geometry_x.y), (r.geometry_y.x, r.geometry_y.y)]),
                axis=1,
            )
            gdf_gen_edges = geopandas.GeoDataFrame(gdf_gen_edges, geometry="geometry", crs="EPSG:4326")
            # gen_options.pop("label", None)
            gdf_gen_edges.to_crs(proj).plot(ax=ax, **gen_options)

    # labels on lines
    if branch_annotation is not None:
        col = branch_annotation.pop("column", None)
        add_line_labels(gdf_edges.to_crs(proj), col=col, ax=ax, **branch_annotation)
    if dcbranch_annotation is not None:
        col = dcbranch_annotation.pop("column", None)
        add_line_labels(gdf_dcedges.to_crs(proj), col=col, ax=ax, **dcbranch_annotation)

    # Legend
    def leg_opts(opts, type="line"):
        l_opts = {k: opts[k] for k in ["marker", "color", "edgecolor", "label"] if k in opts}
        l_opts["markeredgecolor"] = l_opts.pop("edgecolor", None)
        if type == "point":
            l_opts["linestyle"] = "None"
        elif type == "line":
            l_opts["linewidth"] = 2
        return l_opts

    legend_handles = []
    legend_handles.append(Line2D([], [], **leg_opts(node_options, "point")))
    if plot_gentypes is not None:
        legend_handles.append(Line2D([], [], **leg_opts(gen_options, "point")))
    legend_handles.append(Line2D([], [], **leg_opts(branch_options, "line")))
    legend_handles.append(Line2D([], [], **leg_opts(dcbranch_options, "line")))

    ax.legend(handles=legend_handles, **legend_args)

    return legend_handles
