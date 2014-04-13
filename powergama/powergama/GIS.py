# -*- coding: utf-8 -*-
"""
Visualization of results using Google Earth
"""

import simplekml

def makekml(res,timeMaxMin):
    kml = simplekml.Kml()
    kml.document.name = "Results"
    circle = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"
    
    #TODO: Compute average values within timerange    
    timestep=0
    #TODO: Update to get results from sqlite...
    
    gentypeList = res.grid.getAllGeneratorTypes()  
    
    ## Show node
    nodefolder = kml.newfolder(name="Node")
    nodecount = len(res.grid.node.name)
    for i in xrange(nodecount):
        name = res.grid.node.name[i]
        lon = res.grid.node.lon[i]
        lat = res.grid.node.lat[i]
        pnt = nodefolder.newpoint(name=name,coords=[(lon,lat)])
        pnt.style.iconstyle.icon.href = circle
        pnt.style.labelstyle.color = "00000000"

    ## Show generator
    # Create folder for generator
    genfolder = kml.newfolder(name="Generator")
    gencount = len(res.grid.generator.node)
    # Create sub-folders according to generator sub-types
    gentypefolders = {typ: genfolder.newfolder(name=typ) for typ in gentypeList}

    # Get generators
    for i in xrange(gencount):
        typ = res.grid.generator.gentype[i]
        name = res.grid.generator.node[i]
        nodeIndx = res.grid.node.name.index(name)
        lon = res.grid.node.lon[nodeIndx]
        lat = res.grid.node.lat[nodeIndx]
        description = """ 
            Busname .. %s           <br/>
            Fuel .. %s         <br/>
            Lon .. %s, Lat .. %s    <br/>
            """ % (name,typ,str(lon),str(lat))
        pnt = gentypefolders[typ].newpoint(
            name=name,description=description, coords=[(lon,lat)])
        pnt.style.iconstyle.icon.href = circle
        pnt.style.iconstyle.color  = "ff0000ff"
        pnt.style.labelstyle.color = "00000000"
        
    ## Show line
    colorbgr = ["ffff6666","ffffff66","ff66ff66","ff66ffff","f6666fff"] 
    # Find range of branch flow
    #absbranchflow = res.getAverageBranchFlow()
    absbranchflow = [abs(foo) for foo in res.branchFlow[timestep]]
    maxabsbranchflow =  max(absbranchflow)
    minabsbranchflow =  min(absbranchflow)
    print "max flow=",maxabsbranchflow," min flow=",minabsbranchflow    
    steprange = (maxabsbranchflow - minabsbranchflow) / 5
    categoryMax = [minabsbranchflow+steprange*(n+1) for n in xrange(5)]        
    # Create folder for line
    branchfolder = kml.newfolder(name="Branch")
    branchcount = len(res.grid.branch.node_from)
    ## Create sub-folder according to branch flow level
    #branchrange = ranges[0]
    branchlevelfolder=[]
    for level in xrange(5):
        branchlevelfolder.append(branchfolder.newfolder(
            name="Flow <= %s" % (str(categoryMax[level]))))
    # Get and arrange branch flow
    branchflowmatrix = res.branchFlow[timestep]
    for i in xrange(branchcount):
        branchflow = branchflowmatrix[i]

        # Determine category        
        branch_category=6        
        for category in xrange(5):        
            if abs(branchflow) <= categoryMax[category]:
                branch_category=category
                break
        
        color = colorbgr[branch_category]
        capacity = res.grid.branch.capacity[i]
        reactance = res.grid.branch.reactance[i]
        startbus = res.grid.branch.node_from[i]
        endbus = res.grid.branch.node_to[i]
        startbusIndx = res.grid.node.name.index(startbus)
        endbusIndx = res.grid.node.name.index(endbus)
        startbuslon = res.grid.node.lon[startbusIndx]
        startbuslat = res.grid.node.lat[startbusIndx]
        endbuslon = res.grid.node.lon[endbusIndx]
        endbuslat = res.grid.node.lat[endbusIndx]
        name = "%s==%s"%(startbus,endbus)
        description = """
        Startbus .. %s          <br/>
        Endbus .. %s            <br/>
        Capacity .. %s          <br/>
        Reactance .. %s         <br/>
        Mean flow .. %s         <br/>
        """ % (startbus,endbus,capacity,reactance,branchflow)
        lin = branchlevelfolder[branch_category].newlinestring(name=name,
              description = description,
              coords=[(startbuslon,startbuslat),(endbuslon,endbuslat)])
        lin.style.linestyle.color = color
        lin.style.linestyle.width = 1.5    
        
    ## Show nodal price
    maxnodalprice = max(res.sensitivityNodePower[timestep])
    minnodalprice = min(res.sensitivityNodePower[timestep])
    print "max price=",maxnodalprice," min price=",minnodalprice    
    steprange = (maxnodalprice - minnodalprice) / 5
    categoryMax = [minnodalprice+steprange*(n+1) for n in xrange(5)]        
    print categoryMax
    # Create folder for nodal price
    nodalpricefolder = kml.newfolder(name="Nodal price")
    nodalpricelevelfolder=[]
    for level in xrange(5):
        nodalpricelevelfolder.append(nodalpricefolder.newfolder(
            name="Price <= %s" % (str(categoryMax[level]))))
    
    # Get and arrange nodal price
    nodalpricematrix = res.sensitivityNodePower[timestep]
    for i in xrange(nodecount):
        nodalprice = nodalpricematrix[i]
        
        # Determine category        
        node_category=6        
        for category in xrange(5):        
            if abs(nodalprice) <= categoryMax[category]:
                node_category=category
                break
        
        color = colorbgr[node_category]
        name = res.grid.node.name[i]
        lon = res.grid.node.lon[i]
        lat = res.grid.node.lat[i]
        description = """
        Busname .. %s           <br/>
        Lon .. %s, Lat .. %s    <br/>
        Price .. %s             <br/>
        """%(name,lon,lat,nodalprice)
        pnt = nodalpricelevelfolder[node_category].newpoint(
            name=name,
            description=description,
            coords=[(lon,lat)])
        pnt.style.labelstyle.color = "00000000"
        pnt.style.iconstyle.icon.href = circle
        pnt.style.iconstyle.color = color


    # Save kml file
    kml.save("result.kml")

