# -*- coding: utf-8 -*-
"""
Visualization of results using Google Earth


Attributes

category_colours : list
    list of colour codes (aabbggrr) used for nodes and branches. The second
    last value is for NaN, and the last value is default colour. So with
    e.g. 5 colour categories, the list should have 7 elements.

    Colour codes are strings on the format aabbggrr (8-digit hex) - alpha,
    blue, green, red

Example

powergama.GIS.makekml("output.kml",grid_data=data,res=res,
                      nodetype="nodalprice",branchtype="flow")
"""


import simplekml
import math
import numpy
import matplotlib as mpl


category_colours=["ffff6666","ffffff66","ff66ff66","ff66ffff",
            "f6666fff","ffaaaaaa","ff000000"]
'''Default category colours'''

dcbranch_colour = "ffffffff" #white
generator_colour = "ff0000ff" #red
consumer_colour = "ffff00ff" #purple
flowarrow_colour = "ff0000ff" #red

PRICE_MAX = 200
'''default max cap for colouring'''

linewidth = 1.5
'''Default line width'''

point_icon_href = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"
arrow_icon_href = "http://maps.google.com/mapfiles/kml/shapes/donut.png"
'''url of arrow icon'''


def makekml(kmlfile, grid_data,nodetype=None, branchtype=None,
            res=None,timeMaxMin=None,title='PowerGAMA Results'):
    '''Export KML file for Google Earth plot of data

    Colours can be controlled via the module variable "colours"

    Parameters
    ==========
    kmlfile : string
        name of KLM file to create
    grid_data : powergama.GridData
        grid data object
    nodetype : string
        how to plot nodes - 'nodalprice', 'area', 'powergim_type'
    branchtype : string
        how to plot branches -
        'capacity', 'flow', 'sensitivity', 'utilisation',
        'area', 'powergim_type'
    res : powergama.Results (optional)
        result object (result from powergama simulation)
    timeMaxMin : [min,max]
        time range used when plotting results from simulation
    title : string
        title of KML document
    '''
    kml = simplekml.Kml()
    kml.document.name = title

    # Colours, X categories + NaN category + black(default)
    colorbgr_n = category_colours
    colorbgr_b = category_colours
    numCat_n = len(colorbgr_n)-2
    defaultCat_n = numCat_n+1
    numCat_b = len(colorbgr_b)-2
    defaultCat_b = numCat_b+1
    #balloonstyle messes up Google Earth sidebar, for some reason
    #balloontext = "<h3>$[name]</h3> $[description]"

    if nodetype=='area':
        N_categories = len(grid_data.node.area.unique())
        #rgb_colours = [mpl.cm.jet(i/N_categories) for i in range(N_categories)]
        rgb_colours = mpl.cm.tab20(numpy.arange(N_categories)%20)
        colorbgr_n = [simplekml.Color.rgb(int(c[0]*255),int(c[1]*255),int(c[2]*255)) 
                    for c in rgb_colours]
        numCat_n = N_categories
        defaultCat_n=None
    if branchtype=='area':
        N_categories = len(grid_data.node.area.unique())
        #rgb_colours = [mpl.cm.jet(i/N_categories) for i in range(N_categories)]
        rgb_colours = mpl.cm.tab20(numpy.arange(N_categories)%20)
        colorbgr_b = [simplekml.Color.rgb(int(c[0]*255),int(c[1]*255),int(c[2]*255)) 
                    for c in rgb_colours]
        colorbgr_b.append('ffffffff') #white inter-area lines
        numCat_b = N_categories+1
        defaultCat_b=None

    styleNodes = []
    for col in colorbgr_n:
        styleNode = simplekml.Style()
        styleNode.iconstyle.color = col
        styleNode.iconstyle.icon.href = point_icon_href
        styleNode.labelstyle.scale = 0.0 #hide
        #styleNode.balloonstyle.text = balloontext
        styleNodes.append(styleNode)

    styleGenerator = simplekml.Style()
    styleGenerator.iconstyle.icon.href = point_icon_href
    styleGenerator.iconstyle.color  = generator_colour
    styleGenerator.labelstyle.scale = 0.0 #hide
    #styleGenerator.balloonstyle.text = balloontext

    styleConsumer = simplekml.Style()
    styleConsumer.iconstyle.icon.href = point_icon_href
    styleConsumer.iconstyle.color  = consumer_colour
    styleConsumer.labelstyle.scale = 0.0 #hide
    #styleConsumer.iconstyle.visibility = 0 #doesn't work
    #styleConsumer.balloonstyle.text = balloontext

    styleBranches = []
    for col in colorbgr_b:
        styleBranch = simplekml.Style()
        styleBranch.linestyle.color = col
        styleBranch.linestyle.width = linewidth
        styleBranches.append(styleBranch)

    styleDcBranch = simplekml.Style()
    styleDcBranch.linestyle.color = dcbranch_colour
    styleDcBranch.linestyle.width = linewidth

    styleFlowArrow = simplekml.Style()
    styleFlowArrow.iconstyle.icon.href = arrow_icon_href
    styleFlowArrow.iconstyle.color = flowarrow_colour
    styleFlowArrow.iconstyle.scale = 0.5
    styleFlowArrow.labelstyle.scale = 0.0 #hide


    # NODES ##################################################################
    nodefolder = kml.newfolder(name="Node")
    #nodecount = len(grid_data.node.id)
    if nodetype=='nodalprice':
        ## Show nodal price
        meannodalprices = res.getAverageNodalPrices(timeMaxMin)
        # Obs: some values may be numpy.nan
        maxnodalprice = min(PRICE_MAX,numpy.nanmax(meannodalprices))
        minnodalprice = numpy.nanmin(meannodalprices)
        steprange = (maxnodalprice - minnodalprice) / numCat_n
        categoryMax = [math.ceil(minnodalprice+steprange*(n+1))
            for n in range(numCat_n)]
        nodalpricelevelfolder=[]
        for level in range(numCat_n):
            nodalpricelevelfolder.append(nodefolder.newfolder(
                name="Price <= %s" % (str(categoryMax[level]))))
        nodalpricelevelfolder.append(nodefolder.newfolder(
                name="Price NaN" ))
    elif nodetype=='area':
        nodetypes = grid_data.node.area.unique().tolist()
        nodetypefolder=dict()
        for typ in nodetypes:
            nodetypefolder[typ] = nodefolder.newfolder(
                name="Area = {}".format(typ))
    elif nodetype=='powergim_type':
        nodetypes = grid_data.node.type.unique().tolist()
        nodetypefolder=dict()
        for typ in nodetypes:
            nodetypefolder[typ] = nodefolder.newfolder(
                name="Type = {}".format(typ))


    #for i in range(nodecount):
    for i in grid_data.node.index:
        name = grid_data.node.id[i]
        lon = grid_data.node.lon[i]
        lat = grid_data.node.lat[i]
        area = grid_data.node.area[i]
        node_category=defaultCat_n
        #color = None
        description="ID: {} <br/> AREA: {}".format(name,area)
        if nodetype==None:
            pnt = nodefolder.newpoint(name=name,coords=[(lon,lat)],
                                      description=description)
        elif nodetype=='nodalprice':
            nodalprice = meannodalprices[i]
            # Determine category
            node_category=numCat_n
            for category in range(numCat_n):
                if nodalprice <= categoryMax[category]:
                    node_category=category
                    break

            description = """
            Index .. %s             <br/>
            Busname .. %s           <br/>
            Area .. %s              <br/>
            Lon .. %s, Lat .. %s    <br/>
            Price .. %s             <br/>
            """%(str(i),name,area,lon,lat,nodalprice)
            pnt = nodalpricelevelfolder[node_category].newpoint(
                name=name,
                description=description,
                coords=[(lon,lat)])
        elif nodetype=='area':
            typ = grid_data.node.area[i]
            description = """
            Index .. %s             <br/>
            Busname .. %s           <br/>
            Area .. %s              <br/>
            Lon .. %s, Lat .. %s    <br/>
            """%(str(i),name,area,lon,lat)
            node_category = nodetypes.index(typ)
            pnt = nodetypefolder[typ].newpoint(name=name,
                                            description=description,
                                            coords=[(lon,lat)])
        elif nodetype=='powergim_type':
            typ = grid_data.node.type[i]
            description="ID: {}</br>Type: {}".format(name,typ)
            node_category = nodetypes.index(typ)
            pnt = nodetypefolder[typ].newpoint(name=name,
                                            description=description,
                                            coords=[(lon,lat)])
        pnt.style = styleNodes[node_category]
        #pnt.style.iconstyle.color = color
        #pnt.style.iconstyle.icon.href = circle
        #pnt.style.labelstyle.color = "00000000"


    # GENERATORS #############################################################
    genfolder = kml.newfolder(name="Generator")
    gentypeList = grid_data.getAllGeneratorTypes()
    # Create sub-folders according to generator sub-types
    gentypefolders = {typ: genfolder.newfolder(name=typ)
                        for typ in gentypeList}

    # Get generators
    #for i in range(gencount):
    for i in grid_data.generator.index:
        typ = grid_data.generator.type[i]
        cap = grid_data.generator.pmax[i]
        name = "GENERATOR"
        if 'desc' in grid_data.generator.columns:
            name = simplekml.makeunicode.u("GENERATOR {}"
                        .format(grid_data.generator.desc[i]))
        node = grid_data.generator.node[i]
        nodeIndx = grid_data.node.index[grid_data.node.id==node][0]
        lon = grid_data.node.lon[nodeIndx]
        lat = grid_data.node.lat[nodeIndx]
        description = """
            Index .. %s <br/>
            Busname .. %s           <br/>
            Fuel .. %s         <br/>
            Capacity .. %s         <br/>
            Lon .. %s, Lat .. %s    <br/>
            """ % (str(i),node,typ,str(cap),str(lon),str(lat))
        pnt = gentypefolders[typ].newpoint(
            name=name,description=description, coords=[(lon,lat)])
        pnt.style = styleGenerator

    # CONSUMERS #############################################################
    consfolder = kml.newfolder(name="Consumer")

    for i in grid_data.consumer.index:
        avg = grid_data.consumer.demand_avg[i]
        name = "Consumer"
        node = grid_data.consumer.node[i]
        nodeIndx = grid_data.node.index[grid_data.node.id==node][0]
        lon = grid_data.node.lon[nodeIndx]
        lat = grid_data.node.lat[nodeIndx]
        description = """
            Index .. %s <br/>
            Busname .. %s           <br/>
            Avg demand .. %s         <br/>
            Lon .. %s, Lat .. %s    <br/>
            """ % (str(i),node,str(avg),str(lon),str(lat))
        pnt = consfolder.newpoint(
            name=name,description=description, coords=[(lon,lat)])
        pnt.style = styleConsumer


    # BRANCHES ###############################################################
    branchfolder = kml.newfolder(name="Branch")

    if branchtype in ['capacity','flow','utilisation','sensitivity']:
        if res is not None:
            meanflows = res.getAverageBranchFlows(timeMaxMin)
            absbranchflow = meanflows[2]
            brancharrowfolder = branchfolder.newfolder(
                name="Flow direction",visibility=0)
            utilisation = 100*res.getAverageUtilisation(timeMaxMin)
            #sensitiviy is provided for branches with <inf capacity only
            avgsense = numpy.zeros(grid_data.branch.shape[0])
            sens_values = res.getAverageBranchSensitivity(timeMaxMin)
            avgsense[res.idxConstrainedBranchCapacity] = -sens_values
        if branchtype=='flow':
            categoryValue = numpy.asarray(absbranchflow)
            categoryTitle = "Flow"
        elif branchtype=="capacity":
            categoryValue = grid_data.branch['capacity']
            categoryTitle = "Capacity"
        elif branchtype=="utilisation":
            categoryValue = numpy.asarray(utilisation)
            categoryTitle = "Utilisation"
        elif branchtype=='sensitivity':
            categoryValue = numpy.asarray(avgsense)
            categoryTitle = "Sensitivity"

        #Max/min non-infinite value:
        max_value =  max(categoryValue[numpy.isfinite(categoryValue)])
        min_value =  min(categoryValue[numpy.isfinite(categoryValue)])
        steprange = (max_value - min_value) / float(numCat_b)
        categoryMax = [math.ceil(min_value+steprange*(n+1) )
            for n in range(numCat_b)]
        branchlevelfolder=[]
        for level in range(numCat_b):
            branchlevelfolder.append(branchfolder.newfolder(
                name="{} <= {}".format(categoryTitle,
                                       str(categoryMax[level]))))
        branchlevelfolder.append(branchfolder.newfolder(
                name="{} NaN".format(categoryTitle) ))
    elif branchtype=="area":
        branchtypes = grid_data.node.area.unique().tolist()
        branchtypes.append('INTERAREA')
        branchlevelfolder=dict()
        for typ in branchtypes:
            branchlevelfolder[typ] = branchfolder.newfolder(
                name="Area = {}".format(typ))
    elif branchtype=="powergim_type":
        branchtypes = grid_data.branch.type.unique().tolist()
        branchlevelfolder=dict()
        for typ in branchtypes:
            branchlevelfolder[typ] = branchfolder.newfolder(
                name="Type = {}".format(typ))


    for i in grid_data.branch.index:
        startbus = grid_data.branch.node_from[i]
        endbus = grid_data.branch.node_to[i]
        #startbusIndx = (grid_data.node.id == startbus)[0]
        startbusIndx = grid_data.node.index[grid_data.node.id==startbus][0]
        #endbusIndx = (grid_data.node.id == endbus)[0]
        endbusIndx = grid_data.node.index[grid_data.node.id==endbus][0]
        startbuslon = grid_data.node.lon[startbusIndx]
        startbuslat = grid_data.node.lat[startbusIndx]
        endbuslon = grid_data.node.lon[endbusIndx]
        endbuslat = grid_data.node.lat[endbusIndx]
        capacity = grid_data.branch.capacity[i]
        name = "{}=={}".format(startbus,endbus)
        description = "{}<br/>CAPACITY: {}".format(name,capacity)
        branch_category=defaultCat_b

        if branchtype in ['capacity','flow','utilisation','sensitivity']:
            # Determine category
            branch_category=numCat_b
            for category in range(numCat_b):
                if categoryValue[i] <= categoryMax[category]:
                    branch_category=category
                    break

            reactance = grid_data.branch.reactance[i]
            description = """
                Index .. {} <br/>
                Type .. AC <br/>
                Startbus .. {}          <br/>
                Endbus .. {}            <br/>
                Capacity .. {}          <br/>
                Reactance .. {}         <br/>
                """.format(str(i),startbus,endbus,capacity,reactance)
            if res is not None:
                flowAB = meanflows[0][i]
                flowBA = meanflows[1][i]
                description = """{}
                Mean flow .. {:.6g}         <br/>
                Mean flow A to B .. {:.6g}    <br/>
                Mean flow B to A .. {:.6g}    <br/>
                Mean utilisation .. {:.6g} %  <br/>
                Mean sensitivity .. {:.6g}   <br/>
                """.format(description,absbranchflow[i],flowAB,flowBA,
                           utilisation[i],avgsense[i])
            lin = branchlevelfolder[branch_category].newlinestring(name=name,
                  description = description,
                  coords=[(startbuslon,startbuslat),(endbuslon,endbuslat)])

            # Branch flow direction indicator
            if res is not None:
                if (flowAB+flowBA)==0:
                    d = 0.5
                else:
                    d = min(0.9,max(0.1,flowAB/(flowAB+flowBA)))
                arrowcoord = _pointBetween(nodeA=(startbuslon,startbuslat),
                                           nodeB=(endbuslon,endbuslat),
                                           weight=d)
                arrowpoint = brancharrowfolder.newpoint(name=None,
                                                        description=None,
                                                        coords=[arrowcoord],
                                                        visibility=0)
                arrowpoint.style = styleFlowArrow

        elif branchtype=='area':
            msk1 = grid_data.node['id']==grid_data.branch.loc[i,'node_from']
            msk2 = grid_data.node['id']==grid_data.branch.loc[i,'node_to']
            area1 = grid_data.node.loc[msk1,'area'].iloc[0]
            area2 = grid_data.node.loc[msk2,'area'].iloc[0]
            if area1==area2:
                typ = area1
            else:
                typ = "INTERAREA"
            branch_category = branchtypes.index(typ)
            description= """
            {}=={} </br>
            Capacity :   {}
            """.format(startbus,endbus,capacity)
            lin = branchlevelfolder[typ].newlinestring(name=name,
                  description = description,
                  coords=[(startbuslon,startbuslat),(endbuslon,endbuslat)])
        elif branchtype=='powergim_type':
            typ = grid_data.branch.type[i]
            branch_category = branchtypes.index(typ)
            description= """
            {}=={} </br>
            Branch type: {} </br>
            Capacity :   {}
            """.format(startbus,endbus,typ,capacity)
            lin = branchlevelfolder[typ].newlinestring(name=name,
                  description = description,
                  coords=[(startbuslon,startbuslat),(endbuslon,endbuslat)])
        else:
            lin = branchfolder.newlinestring(name=name,
                  description = description,
                  coords=[(startbuslon,startbuslat),(endbuslon,endbuslat)])

        lin.style = styleBranches[branch_category]


    # DC BRANCHES ############################################################
    if grid_data.dcbranch.shape[0] > 0:
        dcbranchfolder = kml.newfolder(name="DC branch")

    for i in grid_data.dcbranch.index:
        startbus = grid_data.dcbranch.node_from[i]
        endbus = grid_data.dcbranch.node_to[i]
        startbusIndx = grid_data.node.index[grid_data.node.id==startbus][0]
        endbusIndx = grid_data.node.index[grid_data.node.id==endbus][0]
        startbuslon = grid_data.node.lon[startbusIndx]
        startbuslat = grid_data.node.lat[startbusIndx]
        endbuslon = grid_data.node.lon[endbusIndx]
        endbuslat = grid_data.node.lat[endbusIndx]
        capacity = grid_data.dcbranch.capacity[i]
        name = "{}=={}".format(startbus,endbus)

        description = """
        Index .. {} <br/>
        Type .. DC <br/>
        Startbus .. {}          <br/>
        Endbus .. {}            <br/>
        Capacity .. {}          <br/>
        """.format(i,startbus,endbus,capacity)
        lin = dcbranchfolder.newlinestring(name=name,
              description = description,
              coords=[(startbuslon,startbuslat),(endbuslon,endbuslat)])

        lin.style = styleDcBranch

    if kmlfile.split(".")[-1]=="kmz":
        kml.savekmz(kmlfile)
    else:
        kml.save(kmlfile)


def _pointBetween(nodeA,nodeB,weight):
    '''computes coords on the line between two points

     [lat lon] = pointBetween(self,lat1,lon1,lat2,lon2,d)

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
    '''
    lat1 = nodeA[0]
    lon1 = nodeA[1]
    lat2 = nodeB[0]
    lon2 = nodeB[1]
    if ((lat1==lat2) and (lon1==lon2)):
        lat = lat1
        lon = lon1
    else:
        #transform to radians
        lat1 = lat1*math.pi/180
        lat2 = lat2*math.pi/180
        lon1 = lon1*math.pi/180
        lon2 = lon2*math.pi/180

        #initial bearing
        y = math.sin(lon2-lon1) * math.cos(lat2)
        x = (math.cos(lat1)*math.sin(lat2)
             - math.sin(lat1)*math.cos(lat2)*math.cos(lon2-lon1) )
        bearing = math.atan2(y, x)

        #angular distance from A to B
        d_tot = (math.acos(math.sin(lat1)*math.sin(lat2)
                 + math.cos(lat1)*math.cos(lat2)*math.cos(lon2-lon1)) )
        d = d_tot*weight

        lat = math.asin(math.sin(lat1)*math.cos(d)
                        +math.cos(lat1)*math.sin(d)*math.cos(bearing) )
        lon = lon1 + math.atan2(math.sin(bearing)*math.sin(d)*math.cos(lat1),
                           math.cos(d)-math.sin(lat1)*math.sin(lat))

        #tansform to degrees
        lat = lat*180/math.pi
        lon = lon*180/math.pi
    return (lat,lon)


