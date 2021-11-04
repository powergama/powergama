# -*- coding: utf-8 -*-
"""
PowerGAMA module containing plotting functions
"""

import folium
import folium.plugins
import branca.colormap
import matplotlib as mpl
import matplotlib.colors
import jinja2
import folium.utilities
import itertools
import pandas as pd
import math


def plotMap(
    pg_data,
    pg_res=None,
    filename=None,
    nodetype=None,
    branchtype=None,
    filter_node=[0, 100],
    filter_branch=None,
    timeMaxMin=None,
    spread_nodes_r=None,
    **kwargs
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
    timeMaxMin : [min,max]
        time interval used when showing simulation results
    spread_nodes_r : float (degrees)
        radius (degrees) of circle on which overlapping nodes are
        spread (use eg 0.04)
    kwargs : arguments passed on to folium.Map(...)
    """

    if branca.__version__ <= "0.2.0":
        cmSet1 = branca.colormap.linear.Set1
    else:
        cmSet1 = branca.colormap.linear.Set1_03

    if timeMaxMin is None:
        timeMaxMin = [pg_data.timerange[0], pg_data.timerange[-1] + 1]

    #    nodetype=None,branchtype=None,dcbranchtype=None,
    #                    show_node_labels=False,branch_style='c',latlon=None,
    #                    timeMaxMin=None,
    #                    dotsize=40, filter_node=None, filter_branch=None,
    #                    draw_par_mer=False,showTitle=True, colors=True

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
    branch = branch.merge(
        node[["id", "lat", "lon"]], how="left", left_on="node_from", right_on="id"
    )
    branch = branch.merge(
        node[["id", "lat", "lon"]], how="left", left_on="node_to", right_on="id"
    )
    dcbranch = dcbranch.merge(
        node[["id", "lat", "lon"]], how="left", left_on="node_from", right_on="id"
    )
    dcbranch = dcbranch.merge(
        node[["id", "lat", "lon"]], how="left", left_on="node_to", right_on="id"
    )
    generator = generator.merge(
        node[["id", "lat", "lon"]], how="left", left_on="node", right_on="id"
    )
    consumer = consumer.merge(
        node[["id", "lat", "lon"]], how="left", left_on="node", right_on="id"
    )
    gentypes = pg_data.getAllGeneratorTypes()
    areas = pg_data.getAllAreas()
    node = node.reset_index().rename(columns={"index": "index_orig"})
    node = node.merge(
        pd.DataFrame(areas).reset_index(), how="left", left_on="area", right_on=0
    ).rename(columns={"index": "area_ind"})
    node = node.set_index("index_orig")
    # node.sort_index(inplace=True)
    node["area_ind"] = 0.5 + node["area_ind"] % 10

    if pg_res is not None:
        nodalprices = pg_res.getAverageNodalPrices(timeMaxMin)
        node["nodalprice"] = nodalprices
        branch_sensitivity = pg_res.getAverageBranchSensitivity(timeMaxMin)
        branch_utilisation = pg_res.getAverageUtilisation(timeMaxMin)
        br_ind = pg_data.getIdxBranchesWithFlowConstraints()
        branch["flow"] = pg_res.getAverageBranchFlows(timeMaxMin)[2]
        branch["utilisation"] = branch_utilisation
        branch["sensitivity"] = 0
        branch.loc[branch.index.isin(br_ind), "sensitivity"] = branch_sensitivity
        # DC branch utilisation:
        if dcbranch.shape[0] > 0:
            dcbranch_utilisation = pg_res.getAverageUtilisation(
                timeMaxMin, branchtype="dc"
            )
            dcbranch_sensitivity = pg_res.getAverageBranchSensitivity(
                timeMaxMin, branchtype="dc"
            )
            dcbr_ind = pg_data.getIdxDcBranchesWithFlowConstraints()
            dcbranch["flow"] = pg_res.getAverageBranchFlows(
                timeMaxMin, branchtype="dc"
            )[2]
            dcbranch["utilisation"] = 0
            dcbranch.loc[
                dcbranch.index.isin(dcbr_ind), "utilisation"
            ] = dcbranch_utilisation
            dcbranch["sensitivity"] = 0
            dcbranch.loc[
                dcbranch.index.isin(dcbr_ind), "sensitivity"
            ] = dcbranch_sensitivity

    m = folium.Map(location=[node["lat"].median(), node["lon"].median()], **kwargs)

    callbackNode = """function (row,colour) {
               if (colour=='') {
                   colour=row[3]
               }
               var marker = L.circleMarker(new L.LatLng(row[0],row[1]),
                                           {"radius":3,
                                            "color":colour} );
                      marker.bindPopup(row[2]);
                      return marker;
            }"""
    callbackBranch = """function (row,colour) {
                if (colour=='') {
                    colour=row[3]
                }
                var polyline = L.polyline([row[0],row[1]],
                                          {"color":colour} );
                polyline.bindPopup(row[2]);
                return polyline;
            }"""

    # print("Nodes...")
    if nodetype == "nodalprice":
        value_col = "nodalprice"
        if filter_node is None:
            filter_node = [node[value_col].min(), node[value_col].max()]
        cm_node = branca.colormap.LinearColormap(
            ["green", "yellow", "red"], vmin=filter_node[0], vmax=filter_node[1]
        )
        cm_node.caption = "Nodal price"
        m.add_child(cm_node)
    elif nodetype == "area":
        value_col = "area_ind"
        val_max = node[value_col].max()
        cm_node = cmSet1.scale(0, val_max).to_step(10)
        # cm_node.caption = 'Area'
        # m.add_child(cm_node)
    elif nodetype == "type":
        type_val, types = node["type"].factorize(sort=True)
        node["type_num"] = type_val
        value_col = "type_num"
        val_max = node[value_col].max()
        cm_node = cmSet1.scale(0, val_max).to_step(10)
    else:
        value_col = None

    locationsN = []
    for i, n in node.iterrows():
        if not (n[["lat", "lon"]].isnull().any()):
            data = [n["lat"], n["lon"], "Node={}, area={}".format(n["id"], n["area"])]
            if pg_res is not None:
                data[2] = "{}; nodalprice={:g}".format(data[2], n["nodalprice"])
            if value_col is not None:
                colHex = cm_node(n[value_col])
                data.append(colHex)
                colour = ""
            else:
                colour = "blue"
            locationsN.append(data)
        else:
            print("Missing lat/lon for node index={}".format(i))
    feature_group_Nodes = folium.FeatureGroup(name="Nodes").add_to(m)
    FeatureCollection(
        data=locationsN, callback=callbackNode, addto=feature_group_Nodes, colour=colour
    ).add_to(feature_group_Nodes)

    # print("AC branches...")
    if branchtype == "utilisation":
        value_col = "utilisation"
        if filter_branch is None:
            filter_branch = [0, 1]
        cm_branch = branca.colormap.LinearColormap(
            ["green", "yellow", "red"], vmin=filter_branch[0], vmax=filter_branch[1]
        )
        cm_branch.caption = "Branch utilisation"
        m.add_child(cm_branch)
    elif branchtype == "sensitivity":
        value_col = "sensitivity"
        if filter_branch is None:
            filter_branch = [branch[value_col].min(), branch[value_col].max()]
        cm_branch = branca.colormap.LinearColormap(
            ["red", "yellow", "green"], vmin=filter_branch[0], vmax=filter_branch[1]
        )
        cm_branch.caption = "Branch capacity sensitivity"
        m.add_child(cm_branch)
    elif branchtype == "flow":
        value_col = "flow"
        if filter_branch is None:
            filter_branch = [branch[value_col].min(), branch[value_col].max()]
        cm_branch = branca.colormap.LinearColormap(
            ["red", "yellow", "green"], vmin=filter_branch[0], vmax=filter_branch[1]
        )
        cm_branch.caption = "Branch flow (abs value)"
        m.add_child(cm_branch)
    elif branchtype == "capacity":
        value_col = "capacity"
        if filter_branch is None:
            filter_branch = [branch[value_col].min(), branch[value_col].max()]
        cm_branch = branca.colormap.LinearColormap(
            ["red", "yellow", "green"], vmin=filter_branch[0], vmax=filter_branch[1]
        )
        cm_branch.caption = "Branch capacity"
        m.add_child(cm_branch)
    elif branchtype == "type":
        type_val, types = branch["type"].factorize(sort=True)
        print(types)
        branch["type_num"] = type_val
        print(branch[["type", "type_num"]])
        value_col = "type_num"
        val_max = branch[value_col].max()
        cm_branch = cmSet1.scale(0, val_max).to_step(10)
    else:
        value_col = None
    locationsB = []
    for i, n in branch.iterrows():
        if (branchtype == "capacity") and (n["capacity"] == 0):
            # skip this branch
            pass
        elif not (n[["lat_x", "lon_x", "lat_y", "lon_y"]].isnull().any()):
            data = [
                [n["lat_x"], n["lon_x"]],
                [n["lat_y"], n["lon_y"]],
                "AC Branch={} ({}-{}), capacity={:g}".format(
                    i, n["node_from"], n["node_to"], n["capacity"]
                ),
            ]
            if branchtype == "type":
                data[2] = "{}; type={}".format(data[2], n["type"])
            if pg_res is not None:
                data[2] = "{}; flow={:g}; utilisation={:g}".format(
                    data[2], n["flow"], n["utilisation"]
                )
            if value_col is not None:
                colHex = cm_branch(n[value_col])
                data.append(colHex)
                colour = ""
            else:
                colour = "black"
            locationsB.append(data)
        else:
            print("Missing lat/lon for node index={}".format(i))
    feature_group_Branches = folium.FeatureGroup(name="AC branches").add_to(m)
    FeatureCollection(
        locationsB, callback=callbackBranch, addto=feature_group_Branches, colour=colour
    ).add_to(feature_group_Branches)

    # print("DC branches...")
    #    if branchtype=="utilisation":
    #        value_col = 'utilisation'
    #        #cm_branch = branca.colormap.LinearColormap(['green', 'yellow', 'red'],
    #        #                                       vmin=0, vmax=1)
    #        #cm_branch.caption = 'AC branch utilisation'
    #        #m.add_child(cm_branch)
    #    else:
    #        value_col=None
    locationsBdc = []
    for i, n in dcbranch.iterrows():
        if not (n[["lat_x", "lon_x", "lat_y", "lon_y"]].isnull().any()):
            data = [
                [n["lat_x"], n["lon_x"]],
                [n["lat_y"], n["lon_y"]],
                "DC Branch={} ({}-{}), capacity={:g}".format(
                    i, n["node_from"], n["node_to"], n["capacity"]
                ),
            ]
            if pg_res is not None:
                data[2] = "{}; flow={:g}; utilisation={:g}".format(
                    data[2], n["flow"], n["utilisation"]
                )
            if value_col is not None:
                colHex = cm_branch(n[value_col])
                data.append(colHex)
                colour = ""
            else:
                colour = "blue"
            locationsBdc.append(data)
        else:
            print("Missing lat/lon for node index={}".format(i))
    feature_group_DcBranches = folium.FeatureGroup(name="DC branches").add_to(m)
    FeatureCollection(
        locationsBdc,
        callback=callbackBranch,
        addto=feature_group_DcBranches,
        colour=colour,
    ).add_to(feature_group_DcBranches)

    # print("Consumers...")
    locationsN = []
    for i, n in consumer.iterrows():
        if not (n[["lat", "lon"]].isnull().any()):
            locationsN.append(
                [
                    n["lat"],
                    n["lon"],
                    "Consumer {} at node={}, avg demand={:g} ({})".format(
                        i, n["node"], n["demand_avg"], n["demand_ref"]
                    ),
                ]
            )
        else:
            print("Missing lat/lon for node index={}".format(i))
    feature_group_Consumer = folium.FeatureGroup(name="Consumers").add_to(m)
    FeatureCollection(
        data=locationsN,
        callback=callbackNode,
        addto=feature_group_Consumer,
        colour="blue",
    ).add_to(feature_group_Consumer)

    # print("Generators...")
    # feature_group_Generator = folium.FeatureGroup(name='Generators').add_to(m)
    ngtypes = max(2, len(gentypes))
    cm_stepG = cmSet1.scale(0, ngtypes - 1).to_step(ngtypes)

    groups = generator.groupby("node")
    feature_group_Generators = folium.FeatureGroup(name="Generators").add_to(m)
    gencluster_icon_create_function = """\
    function(cluster) {
        return L.divIcon({
        html: '<b>' + cluster.getChildCount() + '</b>',
        className: 'marker-cluster marker-cluster-large',
        iconSize: new L.Point(20, 20)
        });
    }"""
    #    for ind,gentype in enumerate(gentypes):
    #        #locationsN=[]
    #        for i,n in generator[generator['type']==gentype].iterrows():
    # Loop by nodes and create marker cluster to avoid generators appearing
    # on top of each other:
    for thenode, genindices in groups.groups.items():
        locationsN = []
        marker_cluster = folium.plugins.MarkerCluster(
            icon_create_function=gencluster_icon_create_function
        )
        marker_cluster.add_to(feature_group_Generators)
        for genind in genindices:
            n = generator.loc[genind]
            gentype = n["type"]
            typeind = gentypes.index(gentype)
            if not (n[["lat", "lon"]].isnull().any()):
                data = [
                    n["lat"],
                    n["lon"],
                    "{}<br>Generator {}: {}, pmax={:g}".format(
                        gentype, genind, n["desc"], n["pmax"]
                    ),
                ]
                col = cm_stepG(typeind)
                data.append(col)
                locationsN.append(data)
            else:
                print("Missing lat/lon for node index={}".format(i))

        # feature_group_GenX = folium.FeatureGroup(name=gentype).add_to(m)
        # FeatureCollection(data=locationsN,callback=callbackNode,
        #                  featuregroup=feature_group_GenX,
        #                  colour="red").add_to(feature_group_GenX)
        FeatureCollection(
            data=locationsN, callback=callbackNode, addto=marker_cluster, colour=""
        ).add_to(marker_cluster)

    legend_generator_html = """
         <div style="position: fixed;
             bottom: 20px; left: 20px;
             border:2px solid grey; z-index:9999; font-size:13px;
             background-color: lightgray">
         &nbsp; <b>Generators</b>"""
    #             bottom: 50px; left: 50px; width: 150px; height: 300px;
    for typeind, gentype in enumerate(gentypes):
        col = cm_stepG(typeind)
        legend_generator_html = (
            '{}<br> &nbsp; <i class="fa fa-circle fa-1x" '
            'style="color:{}">&nbsp;{}</i>'.format(legend_generator_html, col, gentype)
        )
    legend_generator_html = "{}</div>".format(legend_generator_html)
    m.get_root().html.add_child(folium.Element(legend_generator_html))

    folium.LayerControl().add_to(m)

    if filename:
        print("Saving map to file {}".format(filename))
        m.save(filename)

    return m


class FeatureCollection(folium.map.FeatureGroup):
    """
    Add features to a map using in-browser rendering.

    Parameters
    ----------
    data : list
        List of list of shape [[], []]. Data points should be of
        the form [[lat, lng]].
    callback : string, default None
        A string representation of a valid Javascript function
        that will be passed a lat, lon coordinate pair.
    featuregroup : folium.FeatureGroup
        Feature group
    colour : string
        colour
    name : string, default None
        The name of the Layer, as it will appear in LayerControls.
    overlay : bool, default True
        Adds the layer as an optional overlay (True) or the base layer (False).
    control : bool, default True
        Whether the Layer will be included in LayerControls.
    """

    _counts = itertools.count(0)

    def __init__(
        self, data, callback, addto, colour, name=None, overlay=True, control=True
    ):
        super(FeatureCollection, self).__init__(
            name=name, overlay=overlay, control=control
        )
        self._name = "FeatureCollection"
        self._data = data
        self._addto = addto
        self._colour = colour
        self._count = next(self._counts)
        self._callback = "var callback{} = {};".format(self._count, callback)

        self._template = jinja2.Template(
            u"""
            {% macro script(this, kwargs) %}
            {{this._callback}}
            (function(){
                var data = {{this._data}};
                //var map = {{this._parent.get_name()}};
                var addto = {{this._addto.get_name()}};
                var colour = '{{this._colour}}';
                for (var i = 0; i < data.length; i++) {
                    var row = data[i];
                    var feature = callback"""
            + "{}".format(self._count)
            + u"""(row,colour);
                    feature.addTo(addto);
                };
            })();
            {% endmacro %}"""
        )


if __name__ == "__main__":
    plotMap(pg_data, res, filename="mapTest.html")
