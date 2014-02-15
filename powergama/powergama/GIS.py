# -*- coding: utf-8 -*-
"""
Visualization of results using Google Earth
"""

import simplekml

def makekml(res,timestep):
    kml = simplekml.Kml()
    kml.document.name = "Results"
    circleiconstyleurl = \
    "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"
    ## Show node
    nodefolder = kml.newfolder(name="Node")
    nodecount = len(res.grid.node.name)
    for i in xrange(nodecount):
        name = res.grid.node.name[i]
        lon = res.grid.node.lon[i]
        lat = res.grid.node.lat[i]
        pnt = nodefolder.newpoint(coords=[(lon,lat)],description="name=%s"%name)
        pnt.style.iconstyle.icon.href = circleiconstyleurl
    ## Show generator
    # Create folder for generator
    genfolder = kml.newfolder(name="Generator")
    gencount = len(res.grid.generator.node)
    # Create sub-folders according to generator sub-types
    genbiomassfolder = genfolder.newfolder(name="Biomass")
    gencoalfolder = genfolder.newfolder(name="Coal")
    gencoallgnfolder = genfolder.newfolder(name="Coal LGN") # coal_lgn
    gengasfolder = genfolder.newfolder(name="Gas")
    genhydrofolder = genfolder.newfolder(name="Hydro")
    gennuclearfolder = genfolder.newfolder(name="Nuclear")
    genoilfolder = genfolder.newfolder(name="Oil")
    gensolarcspfolder = genfolder.newfolder(name="Solar CSP") # solar_csp
    genunknownfolder = genfolder.newfolder(name="Unknown")
    genwindfolder = genfolder.newfolder(name="Wind")
    # Get generators
    for i in xrange(gencount):
        if res.grid.generator.gentype[i] == "biomass":
            name = res.grid.generator.node[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == name:
                    lon = res.grid.node.lon[j]
                    lat = res.grid.node.lat[j]
                    break
            pnt = genbiomassfolder.newpoint(coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circleiconstyleurl
            pnt.style.iconstyle.color = "ff0000ff"
        if res.grid.generator.gentype[i] == "coal":
            name = res.grid.generator.node[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == name:
                    lon = res.grid.node.lon[j]
                    lat = res.grid.node.lat[j]
                    break
            pnt = gencoalfolder.newpoint(coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circleiconstyleurl
            pnt.style.iconstyle.color = "ff0000ff"
        if res.grid.generator.gentype[i] == "coal_lgn":
            name = res.grid.generator.node[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == name:
                    lon = res.grid.node.lon[j]
                    lat = res.grid.node.lat[j]
                    break
            pnt = gencoallgnfolder.newpoint(coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circleiconstyleurl
            pnt.style.iconstyle.color = "ff0000ff"
        if res.grid.generator.gentype[i] == "gas":
            name = res.grid.generator.node[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == name:
                    lon = res.grid.node.lon[j]
                    lat = res.grid.node.lat[j]
                    break
            pnt = gengasfolder.newpoint(coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circleiconstyleurl
            pnt.style.iconstyle.color = "ff0000ff"
        if res.grid.generator.gentype[i] == "hydro":
            name = res.grid.generator.node[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == name:
                    lon = res.grid.node.lon[j]
                    lat = res.grid.node.lat[j]
                    break
            pnt = genhydrofolder.newpoint(coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circleiconstyleurl
            pnt.style.iconstyle.color = "ffc76915"
        if res.grid.generator.gentype[i] == "nuclear":
            name = res.grid.generator.node[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == name:
                    lon = res.grid.node.lon[j]
                    lat = res.grid.node.lat[j]
                    break
            pnt = gennuclearfolder.newpoint(coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circleiconstyleurl
            pnt.style.iconstyle.color = "ff00ffff"
        if res.grid.generator.gentype[i] == "oil":
            name = res.grid.generator.node[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == name:
                    lon = res.grid.node.lon[j]
                    lat = res.grid.node.lat[j]
                    break
            pnt = genoilfolder.newpoint(coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circleiconstyleurl
            pnt.style.iconstyle.color = "ff0000ff"
        if res.grid.generator.gentype[i] == "solar_csp":
            name = res.grid.generator.node[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == name:
                    lon = res.grid.node.lon[j]
                    lat = res.grid.node.lat[j]
                    break
            pnt = gensolarcspfolder.newpoint(coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circleiconstyleurl
            pnt.style.iconstyle.color = "ff00ff00"
        if res.grid.generator.gentype[i] == "unknown":
            name = res.grid.generator.node[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == name:
                    lon = res.grid.node.lon[j]
                    lat = res.grid.node.lat[j]
                    break
            pnt = genunknownfolder.newpoint(coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circleiconstyleurl
            pnt.style.iconstyle.color = "ffccc6bc"
        if res.grid.generator.gentype[i] == "wind":
            name = res.grid.generator.node[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == name:
                    lon = res.grid.node.lon[j]
                    lat = res.grid.node.lat[j]
                    break
            pnt = genwindfolder.newpoint(coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circleiconstyleurl
            pnt.style.iconstyle.color = "ff00ff00"
    ## Show line
    colorbgr = ["ffff6666","ffffff66","ff66ff66","ff66ffff","f6666fff"] 
    # Find range of branch flow
    absbranchflow = [abs(foo) for foo in res.branchFlow[timestep]]
    maxabsbranchflow =  max(absbranchflow)
    minabsbranchflow =  min(absbranchflow)
    steprange = (maxabsbranchflow - minabsbranchflow) / 5
    ranges = []
    for i in xrange(5):
        if i == 0:
            minrange = minabsbranchflow
            maxrange = minrange + steprange
            ranges.append([minrange,maxrange])
        else:
            minrange = ranges[i-1][1]
            maxrange = minrange + steprange
            ranges.append([minrange,maxrange])
        ranges[i].append(colorbgr[i])
    ## Show Line
    # Create folder for line
    branchfolder = kml.newfolder(name="Branch")
    branchcount = len(res.grid.branch.node_from)
    ## Create sub-folder according to branch flow level
    branchrange = ranges[0]
    branchlevel1folder = branchfolder.newfolder(\
        name="Flow <= %s"%(str(branchrange[1])))
    branchrange = ranges[1]
    branchlevel2folder = branchfolder.newfolder(\
        name="Flow <= %s"%(str(branchrange[1])))
    branchrange = ranges[2]
    branchlevel3folder = branchfolder.newfolder(\
        name="Flow <= %s"%(str(branchrange[1])))
    branchrange = ranges[3]
    branchlevel4folder = branchfolder.newfolder(\
        name="Flow <= %s"%(str(branchrange[1])))
    branchrange = ranges[4]
    branchlevel5folder = branchfolder.newfolder(\
        name="Flow <= %s"%(str(branchrange[1])))
    # Get and arrange branch flow
    branchflowmatrix = res.branchFlow[timestep]
    for i in xrange(branchcount):
        branchflow = branchflowmatrix[i]
        if abs(branchflow) >= ranges[0][0] and abs(branchflow) <= ranges[0][1]:
            color = ranges[0][2]
            startbus = res.grid.branch.node_from[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == startbus:
                    startbuslon = res.grid.node.lon[j]
                    startbuslat = res.grid.node.lat[j]
                    break
            endbus = res.grid.branch.node_to[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == endbus:
                    endbuslon = res.grid.node.lon[j]
                    endbuslat = res.grid.node.lat[j]
                    break
            lin = branchlevel1folder.newlinestring(\
                  coords=[(startbuslon,startbuslat),(endbuslon,endbuslat)])
            lin.style.linestyle.color = color
            lin.style.linestyle.width = 1.5    
        if abs(branchflow) > ranges[1][0] and abs(branchflow) <= ranges[1][1]:
            color = ranges[1][2]
            startbus = res.grid.branch.node_from[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == startbus:
                    startbuslon = res.grid.node.lon[j]
                    startbuslat = res.grid.node.lat[j]
                    break
            endbus = res.grid.branch.node_to[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == endbus:
                    endbuslon = res.grid.node.lon[j]
                    endbuslat = res.grid.node.lat[j]
                    break
            lin = branchlevel2folder.newlinestring(\
                  coords=[(startbuslon,startbuslat),(endbuslon,endbuslat)])
            lin.style.linestyle.color = color
            lin.style.linestyle.width = 1.5
        if abs(branchflow) > ranges[2][0] and abs(branchflow) <= ranges[2][1]:
            color = ranges[2][2]
            startbus = res.grid.branch.node_from[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == startbus:
                    startbuslon = res.grid.node.lon[j]
                    startbuslat = res.grid.node.lat[j]
                    break
            endbus = res.grid.branch.node_to[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == endbus:
                    endbuslon = res.grid.node.lon[j]
                    endbuslat = res.grid.node.lat[j]
                    break
            lin = branchlevel3folder.newlinestring(\
                  coords=[(startbuslon,startbuslat),(endbuslon,endbuslat)])
            lin.style.linestyle.color = color
            lin.style.linestyle.width = 1.5
        if abs(branchflow) > ranges[3][0] and abs(branchflow) <= ranges[3][1]:
            color = ranges[3][2]
            startbus = res.grid.branch.node_from[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == startbus:
                    startbuslon = res.grid.node.lon[j]
                    startbuslat = res.grid.node.lat[j]
                    break
            endbus = res.grid.branch.node_to[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == endbus:
                    endbuslon = res.grid.node.lon[j]
                    endbuslat = res.grid.node.lat[j]
                    break
            lin = branchlevel4folder.newlinestring(\
                  coords=[(startbuslon,startbuslat),(endbuslon,endbuslat)])
            lin.style.linestyle.color = color
            lin.style.linestyle.width = 1.5
        if abs(branchflow) > ranges[4][0] and abs(branchflow) <= ranges[4][1]:
            color = ranges[4][2]
            startbus = res.grid.branch.node_from[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == startbus:
                    startbuslon = res.grid.node.lon[j]
                    startbuslat = res.grid.node.lat[j]
                    break
            endbus = res.grid.branch.node_to[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == endbus:
                    endbuslon = res.grid.node.lon[j]
                    endbuslat = res.grid.node.lat[j]
                    break
            lin = branchlevel5folder.newlinestring(\
                  coords=[(startbuslon,startbuslat),(endbuslon,endbuslat)])
            lin.style.linestyle.color = color
            lin.style.linestyle.width = 1.5
    ## Show nodal price
    maxnodalprice = max(res.sensitivityNodePower[timestep])
    minnodalprice = min(res.sensitivityNodePower[timestep])
    steprange = (maxnodalprice - minnodalprice) / 5
    ranges = []
    for i in xrange(5):
        if i == 0:
            minrange = minnodalprice
            maxrange = minrange + steprange
            ranges.append([minrange,maxrange])
        else:
            minrange = ranges[i-1][1]
            maxrange = minrange + steprange
            ranges.append([minrange,maxrange])
        ranges[i].append(colorbgr[i])
    # Create folder for nodal price
    nodalpricefolder = kml.newfolder(name="Nodal price")
    nodalpricerange = ranges[0]
    nodalpricelevel1folder = nodalpricefolder.newfolder(\
        name="Price <= %s"%(str(nodalpricerange[1])))
    nodalpricerange = ranges[1]
    nodalpricelevel2folder = nodalpricefolder.newfolder(\
        name="Price <= %s"%(str(nodalpricerange[1])))
    nodalpricerange = ranges[2]
    nodalpricelevel3folder = nodalpricefolder.newfolder(\
        name="Price <= %s"%(str(nodalpricerange[1])))
    nodalpricerange = ranges[3]
    nodalpricelevel4folder = nodalpricefolder.newfolder(\
        name="Price <= %s"%(str(nodalpricerange[1])))
    nodalpricerange = ranges[4]
    nodalpricelevel5folder = nodalpricefolder.newfolder(\
        name="Price <= %s"%(str(nodalpricerange[1])))
    # Get and arrange nodal price
    nodalpricematrix = res.sensitivityNodePower[timestep]
    for i in xrange(nodecount):
        nodalprice = nodalpricematrix[i]
        if nodalprice >= ranges[0][0] and nodalprice <= ranges[0][1]:
            color = ranges[0][2]
            name = res.grid.node.name[i]
            lon = res.grid.node.lon[i]
            lat = res.grid.node.lat[i]
            pnt = nodalpricelevel1folder.newpoint(coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circleiconstyleurl
            pnt.style.iconstyle.color = color
    for i in xrange(nodecount):
        nodalprice = nodalpricematrix[i]
        if nodalprice >= ranges[1][0] and nodalprice <= ranges[1][1]:
            color = ranges[1][2]
            name = res.grid.node.name[i]
            lon = res.grid.node.lon[i]
            lat = res.grid.node.lat[i]
            pnt = nodalpricelevel2folder.newpoint(coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circleiconstyleurl
            pnt.style.iconstyle.color = color
    for i in xrange(nodecount):
        nodalprice = nodalpricematrix[i]
        if nodalprice >= ranges[2][0] and nodalprice <= ranges[2][1]:
            color = ranges[2][2]
            name = res.grid.node.name[i]
            lon = res.grid.node.lon[i]
            lat = res.grid.node.lat[i]
            pnt = nodalpricelevel3folder.newpoint(coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circleiconstyleurl
            pnt.style.iconstyle.color = color
    for i in xrange(nodecount):
        nodalprice = nodalpricematrix[i]
        if nodalprice >= ranges[3][0] and nodalprice <= ranges[3][1]:
            color = ranges[3][2]
            name = res.grid.node.name[i]
            lon = res.grid.node.lon[i]
            lat = res.grid.node.lat[i]
            pnt = nodalpricelevel4folder.newpoint(coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circleiconstyleurl
            pnt.style.iconstyle.color = color
    for i in xrange(nodecount):
        nodalprice = nodalpricematrix[i]
        if nodalprice >= ranges[4][0] and nodalprice <= ranges[4][1]:
            color = ranges[4][2]
            name = res.grid.node.name[i]
            lon = res.grid.node.lon[i]
            lat = res.grid.node.lat[i]
            pnt = nodalpricelevel5folder.newpoint(coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circleiconstyleurl
            pnt.style.iconstyle.color = color
    # Save kml file
    kml.save("result.kml")

