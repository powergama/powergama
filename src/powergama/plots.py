# -*- coding: utf-8 -*-
"""
PowerGAMA module containing plotting functions
"""

import math

import branca.colormap
import folium
import folium.plugins
import folium.utilities
import geopandas
import numpy as np
import pandas as pd
import shapely
from folium.plugins.overlapping_marker_spiderfier import OverlappingMarkerSpiderfier
from folium.plugins.treelayercontrol import TreeLayerControl


def plotMap(
    pg_data,
    pg_res=None,
    filename=None,
    nodetype=None,
    branchtype=None,
    filter_node=None,
    filter_branch=None,
    branch_capacity_width=True,
    timeMaxMin=None,
    spread_nodes_r=None,
    layer_control=True,
    fit_bound=True,
    **kwargs,
):
    """
    Plot PowerGAMA data/results on map

    Parameters
    ==========
    pg_data : powergama.GridData
        powergama data object
    pg_res : powergama.Results
        powergama results object
    filename : str
        name of output file (html)
    nodetype : str ('nodalprice','area') or None (default)
        how to colour nodes
    branchtype : str ('utilisation','sensitivity','flow','capacity','type') or None (default)
        how to colour branches
    filter_node : list
        max/min value used for colouring nodes (e.g. nodalprice)
    filter_branch : list
        max/min value used for colouring branches (e.g. utilisation)
    branch_capacity_width : boolean
        scale branch width according to capacity, True (default) or False
    timeMaxMin : [min,max]
        time interval used when showing simulation results
    spread_nodes_r : float (degrees)
        radius (degrees) of circle on which overlapping nodes are
        spread (use eg 0.04)
    kwargs : arguments passed on to folium.Map(...)
    """

    cmSet1 = branca.colormap.linear.Set1_03

    if timeMaxMin is None:
        timeMaxMin = [pg_data.timerange[0], pg_data.timerange[-1] + 1]

    # Add geographic information to branches and generators/consumers
    branch = pg_data.branch.copy()
    dcbranch = pg_data.dcbranch.copy()
    node = pg_data.node.copy()
    generator = pg_data.generator.copy()
    consumer = pg_data.consumer.copy()

    if spread_nodes_r is not None:
        # spread out nodes lying on top of each other
        coords = node[["lat", "lon"]]
        dupl_coords = pd.DataFrame()
        dupl_coords["cumcount"] = coords.groupby(["lat", "lon"]).cumcount()
        dupl_coords["count"] = coords.groupby(["lat", "lon"])["lon"].transform("count")
        for i in node.index:
            n_sum = dupl_coords.loc[i, "count"]
            if n_sum > 1:
                # there are more nodes with the same coordinates
                n = dupl_coords.loc[i, "cumcount"]
                theta = 2 * math.pi / n_sum
                node.loc[i, "lat"] += spread_nodes_r * math.cos(n * theta)
                node.loc[i, "lon"] += spread_nodes_r * math.sin(n * theta)
        # node[['lat','lon']] = coords

    # Careful! Merge may change the order
    branch = branch.merge(node[["id", "lat", "lon"]], how="left", left_on="node_from", right_on="id")
    branch = branch.merge(node[["id", "lat", "lon"]], how="left", left_on="node_to", right_on="id")
    dcbranch = dcbranch.merge(node[["id", "lat", "lon"]], how="left", left_on="node_from", right_on="id")
    dcbranch = dcbranch.merge(node[["id", "lat", "lon"]], how="left", left_on="node_to", right_on="id")
    branch_gen = pd.DataFrame()
    if ("gen_lat" in generator.columns) and ("gen_lon" in generator.columns):
        latlon_from_node = generator.merge(node[["id", "lat", "lon"]], how="left", left_on="node", right_on="id")
        generator["lat"] = generator["gen_lat"].fillna(latlon_from_node["lat"])
        generator["lon"] = generator["gen_lon"].fillna(latlon_from_node["lon"])
        branch_gen = pd.DataFrame(index=generator.index)
        branch_gen["lat_x"] = generator["gen_lat"]
        branch_gen["lat_y"] = latlon_from_node["lat"]
        branch_gen["lon_x"] = generator["gen_lon"]
        branch_gen["lon_y"] = latlon_from_node["lon"]
        # keep only those where generator lat/lon have been specified, and differs from node
        m_remove = (branch_gen.isna().any(axis=1)) | (
            (branch_gen["lat_x"] == branch_gen["lat_y"]) & (branch_gen["lon_x"] == branch_gen["lon_y"])
        )
        branch_gen = branch_gen[~m_remove]
    else:
        generator = generator.merge(node[["id", "lat", "lon"]], how="left", left_on="node", right_on="id")
    consumer = consumer.merge(node[["id", "lat", "lon"]], how="left", left_on="node", right_on="id")
    gentypes_all = pg_data.getAllGeneratorTypes()
    areas = pg_data.getAllAreas()
    node = node.reset_index().rename(columns={"index": "index_orig"})
    node = node.merge(pd.DataFrame(areas).reset_index(), how="left", left_on="area", right_on=0).rename(
        columns={"index": "area_ind"}
    )
    node = node.set_index("index_orig")
    # node.sort_index(inplace=True)
    node["area_ind"] = 0.5 + node["area_ind"] % 10

    node_value_col = None
    node_fields = []
    branch_value_col = None
    branch_fields = []

    if pg_res is not None:
        node_fields.append("nodalprice")
        nodalprices = pg_res.getAverageNodalPrices(timeMaxMin)
        node["nodalprice"] = nodalprices

        branch_sensitivity = pg_res.getAverageBranchSensitivity(timeMaxMin)
        branch_utilisation = pg_res.getAverageUtilisation(timeMaxMin)
        br_ind = pg_data.getIdxBranchesWithFlowConstraints()
        branch["flow12"] = pg_res.getAverageBranchFlows(timeMaxMin)[0]
        branch["flow21"] = pg_res.getAverageBranchFlows(timeMaxMin)[1]
        branch["flow"] = pg_res.getAverageBranchFlows(timeMaxMin)[2]  # avg of abs value
        branch["utilisation"] = branch_utilisation
        branch["sensitivity"] = 0.0
        branch.loc[branch.index.isin(br_ind), "sensitivity"] = branch_sensitivity
        # DC branch utilisation:
        if dcbranch.shape[0] > 0:
            dcbranch_utilisation = pg_res.getAverageUtilisation(timeMaxMin, branchtype="dc")
            dcbranch_sensitivity = pg_res.getAverageBranchSensitivity(timeMaxMin, branchtype="dc")
            dcbr_ind = pg_data.getIdxDcBranchesWithFlowConstraints()
            dcbranch["flow12"] = pg_res.getAverageBranchFlows(timeMaxMin, branchtype="dc")[0]
            dcbranch["flow21"] = pg_res.getAverageBranchFlows(timeMaxMin, branchtype="dc")[1]
            dcbranch["flow"] = pg_res.getAverageBranchFlows(timeMaxMin, branchtype="dc")[2]
            dcbranch["utilisation"] = 0.0
            dcbranch.loc[dcbranch.index.isin(dcbr_ind), "utilisation"] = dcbranch_utilisation
            dcbranch["sensitivity"] = 0.0
            dcbranch.loc[dcbranch.index.isin(dcbr_ind), "sensitivity"] = dcbranch_sensitivity
        branch_fields = ["utilisation", "flow12", "flow21", "flow", "sensitivity"]

    m = folium.Map(location=[node["lat"].median(), node["lon"].median()], **kwargs)

    # Node colouring
    if nodetype is None:
        pass
    elif nodetype == "nodalprice":
        node_value_col = "nodalprice"
        if filter_node is None:
            filter_node = [node[node_value_col].min(), node[node_value_col].max()]
        cm_node = branca.colormap.LinearColormap(["green", "yellow", "red"], vmin=filter_node[0], vmax=filter_node[1])
        cm_node.caption = "Nodal price"
        m.add_child(cm_node)
    elif nodetype == "area":
        node_value_col = "area_ind"
        val_max = node[node_value_col].max()
        cm_node = cmSet1.scale(0, val_max).to_step(10)
    elif nodetype == "type":
        pass

    # Branch colouring
    if branchtype is None:
        pass
    elif branchtype == "utilisation":
        branch_value_col = "utilisation"
        if filter_branch is None:
            filter_branch = [0, 1]
        cm_branch = branca.colormap.LinearColormap(
            ["green", "yellow", "red"], vmin=filter_branch[0], vmax=filter_branch[1]
        )
        cm_branch.caption = "Branch utilisation"
        m.add_child(cm_branch)
    elif branchtype == "sensitivity":
        branch_value_col = "sensitivity"
        if filter_branch is None:
            filter_branch = [branch[branch_value_col].min(), branch[branch_value_col].max()]
        cm_branch = branca.colormap.LinearColormap(
            ["red", "yellow", "green"], vmin=filter_branch[0], vmax=filter_branch[1]
        )
        cm_branch.caption = "Branch capacity sensitivity"
        m.add_child(cm_branch)
    elif branchtype == "flow":
        branch_value_col = "flow"
        if filter_branch is None:
            filter_branch = [branch[branch_value_col].min(), branch[branch_value_col].max()]
        cm_branch = branca.colormap.LinearColormap(
            ["red", "yellow", "green"], vmin=filter_branch[0], vmax=filter_branch[1]
        )
        cm_branch.caption = "Branch flow (abs value)"
        m.add_child(cm_branch)
    elif branchtype == "capacity":
        branch_value_col = "capacity"
        if filter_branch is None:
            filter_branch = [branch[branch_value_col].min(), branch[branch_value_col].max()]
        cm_branch = branca.colormap.LinearColormap(
            ["red", "yellow", "green"], vmin=filter_branch[0], vmax=filter_branch[1]
        )
        cm_branch.caption = "Branch capacity"
        m.add_child(cm_branch)
    elif branchtype == "type":
        pass
    else:
        raise Exception("Invalid branchtype argument")

    # Styling
    cap_scale = 500
    radius_default = 4
    radius_max = 20
    radius_min = 2
    weight_min = 1
    weight_max = 10
    fill_opacity = 0.7

    def style_node(feature):
        style = {}
        if node_value_col:
            style["fillColor"] = cm_node(feature["properties"][node_value_col])
            style["fillOpacity"] = 1
        return style

    def style_branch(feature):
        style = {"color": "black", "weight": 2}
        if branch_capacity_width:
            style["weight"] = np.clip(feature["properties"]["capacity"] / cap_scale, a_min=weight_min, a_max=weight_max)
        if branchtype == "type":
            style["color"] = "red"
        elif branch_value_col:
            style["color"] = cm_branch(feature["properties"][branch_value_col])
        return style

    def style_dcbranch(feature):
        style = {"color": "black", "weight": 2}
        if branch_capacity_width:
            style["weight"] = np.clip(feature["properties"]["capacity"] / cap_scale, a_min=weight_min, a_max=weight_max)
        if branchtype == "type":
            style["color"] = "blue"
        elif branch_value_col:
            style["color"] = cm_branch(feature["properties"][branch_value_col])
        return style

    def style_generator(feature):
        style = {
            "color": "green",
            # "fillColor": "green",
            "radius": np.clip(feature["properties"]["pmax"] / cap_scale, a_min=radius_min, a_max=radius_max),
        }
        return style

    def style_genbranch(feature):
        style = {
            "color": "green",
            "weight": 1,
        }
        return style

    def style_consumers(feature):
        style = {"radius": np.clip(feature["properties"]["demand_avg"] / cap_scale, a_min=radius_min, a_max=radius_max)}
        return style

    # Nodes:
    map_nodes = geopandas.GeoDataFrame(
        node, geometry=geopandas.points_from_xy(node["lon"], node["lat"]), crs="EPSG:4326"
    )
    m_nodes = folium.GeoJson(
        map_nodes,
        style_function=style_node,
        marker=folium.CircleMarker(
            fill=True, fill_opacity=fill_opacity, radius=radius_default, weight=1, color="black"
        ),
        tooltip=folium.GeoJsonTooltip(fields=["id", "area", "zone"] + node_fields),
        name="Nodes",
    ).add_to(m)

    # Branches:
    line_geometry = [
        shapely.LineString([(z[0], z[1]), (z[2], z[3])])
        for z in zip(branch["lon_x"], branch["lat_x"], branch["lon_y"], branch["lat_y"])
    ]
    map_branches = geopandas.GeoDataFrame(branch, geometry=line_geometry, crs="EPSG:4326")
    m_acbranches = folium.GeoJson(
        map_branches,
        style_function=style_branch,
        tooltip=folium.GeoJsonTooltip(fields=["node_from", "node_to", "capacity", "reactance"] + branch_fields),
        name="AC branches",
    ).add_to(m)
    dcline_geometry = [
        shapely.LineString([(z[0], z[1]), (z[2], z[3])])
        for z in zip(dcbranch["lon_x"], dcbranch["lat_x"], dcbranch["lon_y"], dcbranch["lat_y"])
    ]
    map_dcbranches = geopandas.GeoDataFrame(dcbranch, geometry=dcline_geometry, crs="EPSG:4326")
    m_dcbranches = folium.GeoJson(
        map_dcbranches,
        style_function=style_dcbranch,
        tooltip=folium.GeoJsonTooltip(fields=["node_from", "node_to", "capacity"] + branch_fields),
        name="DC branches",
    ).add_to(m)

    # branch flow direction marker:
    br_all = pd.concat([branch, dcbranch], axis=0, ignore_index=True)
    branch_flowmarker = pd.DataFrame(index=br_all.index)
    if pg_res is not None:
        for ind, n in br_all.iterrows():
            if n["flow12"] + n["flow21"] == 0:
                d = 0.5
            else:
                d = min(0.9, max(0.1, n["flow12"] / (n["flow12"] + n["flow21"])))
            (flow_lat, flow_lon) = _pointBetween((n["lat_x"], n["lon_x"]), (n["lat_y"], n["lon_y"]), weight=d)
            branch_flowmarker.loc[ind, "lat"] = flow_lat
            branch_flowmarker.loc[ind, "lon"] = flow_lon

        map_flowmarker = geopandas.GeoDataFrame(
            branch_flowmarker,
            geometry=geopandas.points_from_xy(branch_flowmarker["lon"], branch_flowmarker["lat"]),
            crs="EPSG:4326",
        )
        m_flowmarker = folium.GeoJson(
            map_flowmarker,
            marker=folium.CircleMarker(fill=True, fill_opacity=0.5, radius=radius_default, weight=1, color="gray"),
            name="Flow marker",
        ).add_to(m)

    # Generators:
    map_generators = geopandas.GeoDataFrame(
        generator, geometry=geopandas.points_from_xy(generator["lon"], generator["lat"]), crs="EPSG:4326"
    )
    if branch_gen.shape[0] > 0:
        genline_geometry = [
            shapely.LineString([(z[0], z[1]), (z[2], z[3])])
            for z in zip(branch_gen["lon_x"], branch_gen["lat_x"], branch_gen["lon_y"], branch_gen["lat_y"])
        ]
        map_genbranches = geopandas.GeoDataFrame(branch_gen, geometry=genline_geometry, crs="EPSG:4326")
    m_generators = {}
    for gentype in gentypes_all:
        m_generators[gentype] = folium.FeatureGroup(name=gentype).add_to(m)
        folium.GeoJson(
            map_generators[map_generators["type"] == gentype],
            marker=folium.CircleMarker(fill=True, fillOpacity=0.7, weight=1, fill_color="green"),
            style_function=style_generator,
            tooltip=folium.GeoJsonTooltip(fields=["node", "desc", "type", "pmax", "inflow_ref"]),
            # popup=folium.GeoJsonPopup(fields=["node", "desc", "type","pmax","inflow_ref"]),
            name=gentype,
        ).add_to(m_generators[gentype])
        if branch_gen.shape[0] > 0:
            genbranch_indices = [
                i for i in map_generators[map_generators["type"] == gentype].index if i in map_genbranches.index
            ]
            folium.GeoJson(
                map_genbranches.loc[genbranch_indices],
                style_function=style_genbranch,
                name=f"{gentype}:connection",
            ).add_to(m_generators[gentype])

    # Consumers:
    map_consumers = geopandas.GeoDataFrame(
        consumer, geometry=geopandas.points_from_xy(consumer["lon"], consumer["lat"]), crs="EPSG:4326"
    )
    m_consumers = folium.GeoJson(
        map_consumers,
        marker=folium.CircleMarker(fill=True, color="black", fill_color="yellow", fill_opacity=0.5, weight=1),
        style_function=style_consumers,
        tooltip=folium.GeoJsonTooltip(fields=["node", "demand_avg", "demand_ref"]),
        name="Consumers",
    ).add_to(m)

    if fit_bound:
        sw = node[["lat", "lon"]].min().values.tolist()
        ne = node[["lat", "lon"]].max().values.tolist()
        m.fit_bounds([sw, ne])
    if layer_control:
        overlay_tree = {
            "label": "powergama",
            "select_all_checkbox": True,
            "children": [
                {"label": "Nodes", "layer": m_nodes},
                {
                    "label": "Branches",
                    "select_all_checkbox": True,
                    "children": [
                        {"label": "AC", "layer": m_acbranches},
                        {"label": "DC", "layer": m_dcbranches},
                        {"label": "flow markers", "layer": m_flowmarker},
                    ],
                },
                {"label": "Consumers", "layer": m_consumers},
                {
                    "label": "Generators",
                    "select_all_checkbox": True,
                    "children": [{"label": gentype, "layer": m_generators[gentype]} for gentype in gentypes_all],
                },
            ],
        }
        TreeLayerControl(overlay_tree=overlay_tree).add_to(m)

    # TODO: This does not seem to work with GeoJson data markers
    oms = OverlappingMarkerSpiderfier()  # keep_spiderfied=True, nearby_distance=50, leg_weight=2.0)
    oms.add_to(m)

    if filename:
        print(f"Saving map to file {filename}")
        m.save(filename)

    return m


def _pointBetween(nodeA, nodeB, weight):
    """computes coords on the line between two points

     (lat,lon) = pointBetween(self,lat1,lon1,lat2,lon2,d)

    Parameters
    ----------
        nodeA: dublet (lat,lon)
            latitude/longitude of nodeA (degrees)
        nodeB: dublet (lat,lon)
            latitude/longitude of nodeB (degrees)
        weight: double
            weight=0 is node A, 0.5 is halfway between, and 1 is node B

    Returns
    -------
        (lat,lon): dublet
            coordinates of the point inbetween nodeA and nodeB (degrees)


     ref: http://www.movable-type.co.uk/scripts/latlong.html
    """
    lat1 = nodeA[0]
    lon1 = nodeA[1]
    lat2 = nodeB[0]
    lon2 = nodeB[1]
    if (lat1 == lat2) and (lon1 == lon2):
        lat = lat1
        lon = lon1
    else:
        # transform to radians
        lat1 = lat1 * math.pi / 180
        lat2 = lat2 * math.pi / 180
        lon1 = lon1 * math.pi / 180
        lon2 = lon2 * math.pi / 180

        # initial bearing
        y = math.sin(lon2 - lon1) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
        bearing = math.atan2(y, x)

        # angular distance from A to B
        d_tot = math.acos(math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1))
        d = d_tot * weight

        lat = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(bearing))
        lon = lon1 + math.atan2(
            math.sin(bearing) * math.sin(d) * math.cos(lat1), math.cos(d) - math.sin(lat1) * math.sin(lat)
        )

        # tansform to degrees
        lat = lat * 180 / math.pi
        lon = lon * 180 / math.pi
    return (lat, lon)
