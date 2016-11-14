# -*- coding: utf-8 -*-
"""
Visualization of results using Google Earth
"""

import simplekml
import math
import numpy

def makekml(kmlfile, grid_data,nodetype=None, branchtype=None, 
            res=None,timeMaxMin=None,title='PowerGAMA Results'):
    '''Export KML file for Google Earth plot of data
    
    Parameters
    ==========
    kmlfile : string
        name of KLM file to create
    grid_data : powergama.GridData
        grid data object
    nodetype : string
        how to plot nodes - 'areaprice','powergim_type'
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
    circle = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"
    
    # Colours, 5 categories + NaN category
    colorbgr = ["ffff6666","ffffff66","ff66ff66","ff66ffff","f6666fff",
                "ffaaaaaa"] 
    numCat = len(colorbgr)-1
    
    
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
        color = None
        description="ID: {}".format(name)
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
            
            color = colorbgr[node_category]
            description = """
            Busname .. %s           <br/>
            Lon .. %s, Lat .. %s    <br/>
            Price .. %s             <br/>
            """%(name,lon,lat,nodalprice)
            pnt = nodalpricelevelfolder[node_category].newpoint(
                name=name,
                description=description,
                coords=[(lon,lat)])
        elif nodetype=='powergim_type':
            typ = grid_data.node.type[i]
            description="ID: {}</br>Type: {}".format(name,typ)
            node_category = nodetypes.index(typ)
            color = colorbgr[node_category]
            pnt = nodetypefolder[typ].newpoint(name=name,
                                            description=description,
                                            coords=[(lon,lat)])
        pnt.style.iconstyle.color = color
        pnt.style.iconstyle.icon.href = circle
        pnt.style.labelstyle.color = "00000000"
            

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
            Busname .. %s           <br/>
            Fuel .. %s         <br/>
            Capacity .. %s         <br/>
            Lon .. %s, Lat .. %s    <br/>
            """ % (node,typ,str(cap),str(lon),str(lat))
        pnt = gentypefolders[typ].newpoint(
            name=name,description=description, coords=[(lon,lat)])
        pnt.style.iconstyle.icon.href = circle
        pnt.style.iconstyle.color  = "ff0000ff"
        pnt.style.labelstyle.color = "00000000"
        

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
        description = name
        color = None
        
        if branchtype==None:
            lin = branchfolder.newlinestring(name=name,
                  description = description,
                  coords=[(startbuslon,startbuslat),(endbuslon,endbuslat)])
        elif branchtype=='flow':
            # Determine category        
            branch_category=numCat        
            for category in range(numCat):        
                if absbranchflow[i] <= categoryMax[category]:
                    branch_category=category
                    break
            
            color = colorbgr[branch_category]
            reactance = grid_data.branch.reactance[i]
            description = """
            Startbus .. %s          <br/>
            Endbus .. %s            <br/>
            Capacity .. %s          <br/>
            Reactance .. %s         <br/>
            Mean flow .. %s         <br/>
            """ % (startbus,endbus,capacity,reactance,absbranchflow[i])
            lin = branchlevelfolder[branch_category].newlinestring(name=name,
                  description = description,
                  coords=[(startbuslon,startbuslat),(endbuslon,endbuslat)])
        elif branchtype=='powergim_type':
            typ = grid_data.branch.type[i]
            branch_category = branchtypes.index(typ)
            color = colorbgr[branch_category]
            description= """
            {}=={} </br>
            Branch type: {} </br>
            Capacity :   {}
            """.format(startbus,endbus,typ,capacity)
            lin = branchlevelfolder[typ].newlinestring(name=name,
                  description = description,
                  coords=[(startbuslon,startbuslat),(endbuslon,endbuslat)])
            pass
        lin.style.linestyle.color = color
        lin.style.linestyle.width = 1.5    
        
    kml.save(kmlfile)

