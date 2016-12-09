# -*- coding: utf-8 -*-
"""
Visualization of results using Google Earth


Attributes
----------
category_colours : list
    list of colour codes (aabbggrr) used for nodes and branches. The second 
    last value is for NaN, and the last value is default colour. So with
    e.g. 5 colour categories, the list should have 7 elements.
    
    Colour codes are strings on the format aabbggrr (8-digit hex) - alpha,
    blue, green, red

Example
-------
powergama.GIS.makekml("output.kml",grid_data=data,res=res,
                      nodetype="nodalprice",branchtype="flow")
"""


import simplekml
import math
import numpy

# Default category colours
category_colours=["ffff6666","ffffff66","ff66ff66","ff66ffff",
            "f6666fff","ffaaaaaa","ff000000"]

dcbranch_colour = "ffffffff" #white
generator_colour = "ff0000ff" #red
consumer_colour = "ffff00ff" #purple

# Default line width
linewidth = 1.5

point_icon_href = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"



            
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
        how to plot nodes - 'nodalprice','powergim_type'
    branchtype : string
        how to plot branches - 'flow','powergim_type'
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
    colorbgr = category_colours
    numCat = len(colorbgr)-2
    defaultCat = numCat+1
    #balloonstyle messes up Google Earth sidebar, for some reason
    #balloontext = "<h3>$[name]</h3> $[description]"
    
    styleNodes = []
    for col in colorbgr:    
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
    #styleConsumer.balloonstyle.text = balloontext
    
    styleBranches = []
    for col in colorbgr:    
        styleBranch = simplekml.Style()
        styleBranch.linestyle.color = col
        styleBranch.linestyle.width = linewidth   
        styleBranches.append(styleBranch)
    
    styleDcBranch = simplekml.Style()
    styleDcBranch.linestyle.color = dcbranch_colour
    styleDcBranch.linestyle.width = linewidth   


    # NODES ##################################################################   
    nodefolder = kml.newfolder(name="Node")
    #nodecount = len(grid_data.node.id)
    if nodetype=='nodalprice':
        ## Show nodal price
        meannodalprices = res.getAverageNodalPrices(timeMaxMin)
        # Obs: some values may be numpy.nan
        maxnodalprice = numpy.nanmax(meannodalprices)
        minnodalprice = numpy.nanmin(meannodalprices)
        steprange = (maxnodalprice - minnodalprice) / numCat
        categoryMax = [math.ceil(minnodalprice+steprange*(n+1)) 
            for n in range(numCat)]        
        nodalpricelevelfolder=[]
        for level in range(numCat):
            nodalpricelevelfolder.append(nodefolder.newfolder(
                name="Price <= %s" % (str(categoryMax[level]))))
        #nodalpricelevelfolder.append(nodalpricefolder.newfolder(
        #    name="Price > %s" % (str(categoryMax[numCat-2]))))
        nodalpricelevelfolder.append(nodefolder.newfolder(
                name="Price NaN" ))
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
        node_category=defaultCat
        #color = None
        description="ID: {} <br/> AREA: {}".format(name,area)
        if nodetype==None:
            pnt = nodefolder.newpoint(name=name,coords=[(lon,lat)],
                                      description=description)
        elif nodetype=='nodalprice':
            nodalprice = meannodalprices[i]
            # Determine category        
            node_category=numCat
            for category in range(numCat):        
                if nodalprice <= categoryMax[category]:
                    node_category=category
                    break
            
            #color = colorbgr[node_category]
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
        elif nodetype=='powergim_type':
            typ = grid_data.node.type[i]
            description="ID: {}</br>Type: {}".format(name,typ)
            node_category = nodetypes.index(typ)
            #color = colorbgr[node_category]
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

    if branchtype=='flow':
        meanflows = res.getAverageBranchFlows(timeMaxMin)
        absbranchflow = meanflows[2]
        maxabsbranchflow =  max(absbranchflow)
        minabsbranchflow =  min(absbranchflow)
        steprange = (maxabsbranchflow - minabsbranchflow) / float(numCat)
        categoryMax = [math.ceil(minabsbranchflow+steprange*(n+1) )
            for n in range(numCat)]        
        branchlevelfolder=[]
        for level in range(numCat):
            branchlevelfolder.append(branchfolder.newfolder(
                name="Flow <= %s" % (str(categoryMax[level]))))
        branchlevelfolder.append(branchfolder.newfolder(
                name="Flow NaN" ))
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
        branch_category=defaultCat
        
        if branchtype=='flow':
            # Determine category        
            branch_category=numCat        
            for category in range(numCat):        
                if absbranchflow[i] <= categoryMax[category]:
                    branch_category=category
                    break
            
            reactance = grid_data.branch.reactance[i]
            description = """
            Index .. %s <br/>
            Type .. AC <br/>
            Startbus .. %s          <br/>
            Endbus .. %s            <br/>
            Capacity .. %s          <br/>
            Reactance .. %s         <br/>
            Mean flow .. %s         <br/>
            """ % (str(i),startbus,endbus,capacity,reactance,absbranchflow[i])
            lin = branchlevelfolder[branch_category].newlinestring(name=name,
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
        
    kml.save(kmlfile)

