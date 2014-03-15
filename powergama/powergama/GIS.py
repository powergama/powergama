# -*- coding: utf-8 -*-
"""
Visualization of results using Google Earth
"""

import simplekml

def makekml(res,timestep):
    kml = simplekml.Kml()
    kml.document.name = "Results"
    circle = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"
    hydrogen = "http://maps.google.com/mapfiles/kml/shapes/water.png"
    thermalgen = "http://maps.google.com/mapfiles/kml/shapes/firedept.png"
    star = "http://maps.google.com/mapfiles/kml/shapes/star.png"
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
            description = """ 
            Busname .. %s           <br/>
            Fuel .. Biomass         <br/>
            Lon .. %s, Lat .. %s    <br/>
            """%(name,str(lon),str(lat))
            pnt = genbiomassfolder.newpoint(name=name,description=description,\
                                            coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circle
            pnt.style.iconstyle.color  = "ff0000ff"
            pnt.style.labelstyle.color = "00000000"
        if res.grid.generator.gentype[i] == "coal":
            name = res.grid.generator.node[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == name:
                    lon = res.grid.node.lon[j]
                    lat = res.grid.node.lat[j]
                    break
            description = """
            Busname .. %s           <br/>
            Fuel .. Coal            <br/>
            Lon .. %s, Lat .. %s    <br/>
            """%(name,str(lon),str(lat))
            pnt = gencoalfolder.newpoint(name=name,description=description,\
                                         coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circle
            pnt.style.iconstyle.color  = "ff0000ff"
            pnt.style.labelstyle.color = "00000000"
        if res.grid.generator.gentype[i] == "coal_lgn":
            name = res.grid.generator.node[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == name:
                    lon = res.grid.node.lon[j]
                    lat = res.grid.node.lat[j]
                    break
            description = """
            Busname .. %s           <br/>
            Fuel .. Lignite coal    <br/>
            Lon .. %s, Lat .. %s    <br/>
            """%(name,str(lon),str(lat))
            pnt = gencoallgnfolder.newpoint(name=name,description=description,\
                                            coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circle
            pnt.style.iconstyle.color  = "ff0000ff"
            pnt.style.labelstyle.color = "00000000"
        if res.grid.generator.gentype[i] == "gas":
            name = res.grid.generator.node[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == name:
                    lon = res.grid.node.lon[j]
                    lat = res.grid.node.lat[j]
                    break
            description = """
            Busname .. %s           <br/>
            Fuel .. Gas             <br/>
            Lon .. %s, Lat ..%s     <br/>
            """%(name,str(lon),str(lat))
            pnt = gengasfolder.newpoint(name=name,description=description,\
                                        coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circle
            pnt.style.iconstyle.color  = "ff0000ff"
            pnt.style.labelstyle.color = "00000000"
        if res.grid.generator.gentype[i] == "hydro":
            name = res.grid.generator.node[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == name:
                    lon = res.grid.node.lon[j]
                    lat = res.grid.node.lat[j]
                    break
            description = """
            Busname .. %s           <br/>
            Fuel .. Hydro           <br/>
            Lon .. %s, Lat .. %s    <br/>
            """%(name,str(lon),str(lat))
            pnt = genhydrofolder.newpoint(name=name,description=description,\
                                          coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circle
            pnt.style.iconstyle.color  = "ffff8000"
            pnt.style.labelstyle.color = "00000000"
        if res.grid.generator.gentype[i] == "nuclear":
            name = res.grid.generator.node[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == name:
                    lon = res.grid.node.lon[j]
                    lat = res.grid.node.lat[j]
                    break
            description = """
            Busname .. %s           <br/>
            Fuel .. Nuclear         <br/>
            Lon .. %s, Lat .. %s    <br/>
            """%(name,str(lon),str(lat))
            pnt = gennuclearfolder.newpoint(name=name,description=description,\
                                            coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circle
            pnt.style.iconstyle.color  = "ff00ffff"
            pnt.style.labelstyle.color = "00000000"
        if res.grid.generator.gentype[i] == "oil":
            name = res.grid.generator.node[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == name:
                    lon = res.grid.node.lon[j]
                    lat = res.grid.node.lat[j]
                    break
            description = """
            Busname .. %s           <br/>
            Fuel .. Oil             <br/>
            Lon .. %s, Lat .. %s    <br/>
            """%(name,str(lon),str(lat))
            pnt = genoilfolder.newpoint(name=name,description=description,\
                                        coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circle
            pnt.style.iconstyle.color  = "ff0000ff"
            pnt.style.labelstyle.color = "00000000"
        if res.grid.generator.gentype[i] == "solar_csp":
            name = res.grid.generator.node[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == name:
                    lon = res.grid.node.lon[j]
                    lat = res.grid.node.lat[j]
                    break
            description = """
            Busname .. %s           <br/>
            Fuel .. Solar CSP       <br/>
            Lon .. %s, Lat .. %s    <br/>
            """%(name,str(lon),str(lat))
            pnt = gensolarcspfolder.newpoint(name=name,description=description,\
                                             coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circle
            pnt.style.iconstyle.color  = "ff00ff00"
            pnt.style.labelstyle.color = "00000000"
        if res.grid.generator.gentype[i] == "unknown":
            name = res.grid.generator.node[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == name:
                    lon = res.grid.node.lon[j]
                    lat = res.grid.node.lat[j]
                    break
            description = """
            Busname .. %s           <br/>
            Fuel .. Unknown         <br/>
            Lon .. %s, Lat .. %s    <br/>
            """%(name,str(lon),str(lat))
            pnt = genunknownfolder.newpoint(name=name,description=description,\
                                            coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circle
            pnt.style.iconstyle.color  = "ffccc6bc"
            pnt.style.labelstyle.color = "00000000"
        if res.grid.generator.gentype[i] == "wind":
            name = res.grid.generator.node[i]
            for j in xrange(nodecount):
                if res.grid.node.name[j] == name:
                    lon = res.grid.node.lon[j]
                    lat = res.grid.node.lat[j]
                    break
            description = """
            Busname .. %s           <br/>
            Fuel .. Wind            <br/>
            Lon .. %s, Lat .. %s    <br/>
            """%(name,str(lon),str(lat))
            pnt = genwindfolder.newpoint(name=name,description=description,\
                                         coords=[(lon,lat)])
            pnt.style.iconstyle.icon.href = circle
            pnt.style.iconstyle.color  = "ff00ff00"
            pnt.style.labelstyle.color = "00000000"
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
            capacity = res.grid.branch.capacity[i]
            reactance = res.grid.branch.reactance[i]
            susceptance = res.grid.branch._susceptance[i]
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
            name = "%s==%s"%(startbus,endbus)
            description = """
            Startbus .. %s          <br/>
            Lon .. %s, Lat .. %s    <br/>
            Endbus .. %s            <br/>
            Lon .. %s, Lat .. %s    <br/>
            Capacity .. %s          <br/>
            Reactance .. %s         <br/>
            Susceptance .. %s       <br/>
            """%(startbus,startbuslon,startbuslat,endbus,endbuslon,endbuslat,\
                 capacity,reactance,susceptance)
            lin = branchlevel1folder.newlinestring(name=name,\
                  description = description,\
                  coords=[(startbuslon,startbuslat),(endbuslon,endbuslat)])
            lin.style.linestyle.color = color
            lin.style.linestyle.width = 1.5    
        if abs(branchflow) > ranges[1][0] and abs(branchflow) <= ranges[1][1]:
            color = ranges[1][2]
            capacity = res.grid.branch.capacity[i]
            reactance = res.grid.branch.reactance[i]
            susceptance = res.grid.branch._susceptance[i]
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
            name = "%s==%s"%(startbus,endbus)
            description = """
            Startbus .. %s          <br/>
            Lon .. %s, Lat .. %s    <br/>
            Endbus .. %s            <br/>
            Lon .. %s, Lat .. %s    <br/>
            Capacity .. %s          <br/>
            Reactance .. %s         <br/>
            Susceptance .. %s       <br/>
            """%(startbus,startbuslon,startbuslat,endbus,endbuslon,endbuslat,\
                 capacity,reactance,susceptance)
            lin = branchlevel2folder.newlinestring(name=name,\
                  description = description,\
                  coords=[(startbuslon,startbuslat),(endbuslon,endbuslat)])
            lin.style.linestyle.color = color
            lin.style.linestyle.width = 1.5
        if abs(branchflow) > ranges[2][0] and abs(branchflow) <= ranges[2][1]:
            color = ranges[2][2]
            capacity = res.grid.branch.capacity[i]
            reactance = res.grid.branch.reactance[i]
            susceptance = res.grid.branch._susceptance[i]
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
            name = "%s==%s"%(startbus,endbus)
            description = """
            Startbus .. %s          <br/>
            Lon .. %s, Lat .. %s    <br/>
            Endbus .. %s            <br/>
            Lon .. %s, Lat .. %s    <br/>
            Capacity .. %s          <br/>
            Reactance .. %s         <br/>
            Susceptance .. %s       <br/>
            """%(startbus,startbuslon,startbuslat,endbus,endbuslon,endbuslat,\
                 capacity,reactance,susceptance)
            lin = branchlevel3folder.newlinestring(name=name,\
                  description = description,\
                  coords=[(startbuslon,startbuslat),(endbuslon,endbuslat)])
            lin.style.linestyle.color = color
            lin.style.linestyle.width = 1.5
        if abs(branchflow) > ranges[3][0] and abs(branchflow) <= ranges[3][1]:
            color = ranges[3][2]
            capacity = res.grid.branch.capacity[i]
            reactance = res.grid.branch.reactance[i]
            susceptance = res.grid.branch._susceptance[i]
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
            name = "%s==%s"%(startbus,endbus)
            description = """
            Startbus .. %s          <br/>
            Lon .. %s, Lat .. %s    <br/>
            Endbus .. %s            <br/>
            Lon .. %s, Lat .. %s    <br/>
            Capacity .. %s          <br/>
            Reactance .. %s         <br/>
            Susceptance .. %s       <br/>
            """%(startbus,startbuslon,startbuslat,endbus,endbuslon,endbuslat,\
                 capacity,reactance,susceptance)
            lin = branchlevel4folder.newlinestring(name=name,\
                  description = description,\
                  coords=[(startbuslon,startbuslat),(endbuslon,endbuslat)])
            lin.style.linestyle.color = color
            lin.style.linestyle.width = 1.5
        if abs(branchflow) > ranges[4][0] and abs(branchflow) <= ranges[4][1]:
            color = ranges[4][2]
            capacity = res.grid.branch.capacity[i]
            reactance = res.grid.branch.reactance[i]
            susceptance = res.grid.branch._susceptance[i]
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
            name = "%s==%s"%(startbus,endbus)
            description = """
            Startbus .. %s          <br/>
            Lon .. %s, Lat .. %s    <br/>
            Endbus .. %s            <br/>
            Lon .. %s, Lat .. %s    <br/>
            Capacity .. %s          <br/>
            Reactance .. %s         <br/>
            Susceptance .. %s       <br/>
            """%(startbus,startbuslon,startbuslat,endbus,endbuslon,endbuslat,\
                 capacity,reactance,susceptance)
            lin = branchlevel5folder.newlinestring(name=name,\
                  description = description,\
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
            description = """
            Busname .. %s           <br/>
            Lon .. %s, Lat .. %s    <br/>
            Price .. %s             <br/>
            """%(name,lon,lat,nodalprice)
            pnt = nodalpricelevel1folder.newpoint(name=name,\
                  description=description,coords=[(lon,lat)])
            pnt.style.labelstyle.color = "00000000"
            pnt.style.iconstyle.icon.href = circle
            pnt.style.iconstyle.color = color
    for i in xrange(nodecount):
        nodalprice = nodalpricematrix[i]
        if nodalprice >= ranges[1][0] and nodalprice <= ranges[1][1]:
            color = ranges[1][2]
            name = res.grid.node.name[i]
            lon = res.grid.node.lon[i]
            lat = res.grid.node.lat[i]
            description = """
            Busname .. %s           <br/>
            Lon .. %s, Lat .. %s    <br/>
            Price .. %s             <br/>
            """%(name,lon,lat,nodalprice)
            pnt = nodalpricelevel2folder.newpoint(name=name,\
                  description=description,coords=[(lon,lat)])
            pnt.style.labelstyle.color = "00000000"
            pnt.style.iconstyle.icon.href = circle
            pnt.style.iconstyle.color = color
    for i in xrange(nodecount):
        nodalprice = nodalpricematrix[i]
        if nodalprice >= ranges[2][0] and nodalprice <= ranges[2][1]:
            color = ranges[2][2]
            name = res.grid.node.name[i]
            lon = res.grid.node.lon[i]
            lat = res.grid.node.lat[i]
            description = """
            Busname .. %s           <br/>
            Lon .. %s, Lat .. %s    <br/>
            Price .. %s             <br/>
            """%(name,lon,lat,nodalprice)
            pnt = nodalpricelevel3folder.newpoint(name=name,\
                  description=description,coords=[(lon,lat)])
            pnt.style.labelstyle.color = "00000000"
            pnt.style.iconstyle.icon.href = circle
            pnt.style.iconstyle.color = color
    for i in xrange(nodecount):
        nodalprice = nodalpricematrix[i]
        if nodalprice >= ranges[3][0] and nodalprice <= ranges[3][1]:
            color = ranges[3][2]
            name = res.grid.node.name[i]
            lon = res.grid.node.lon[i]
            lat = res.grid.node.lat[i]
            description = """
            Busname .. %s           <br/>
            Lon .. %s, Lat .. %s    <br/>
            Price .. %s             <br/>
            """%(name,lon,lat,nodalprice)
            pnt = nodalpricelevel4folder.newpoint(name=name,\
                  description=description,coords=[(lon,lat)])
            pnt.style.labelstyle.color = "00000000"
            pnt.style.iconstyle.icon.href = circle
            pnt.style.iconstyle.color = color
    for i in xrange(nodecount):
        nodalprice = nodalpricematrix[i]
        if nodalprice >= ranges[4][0] and nodalprice <= ranges[4][1]:
            color = ranges[4][2]
            name = res.grid.node.name[i]
            lon = res.grid.node.lon[i]
            lat = res.grid.node.lat[i]
            description = """
            Busname .. %s           <br/>
            Lon .. %s, Lat .. %s    <br/>
            Price .. %s             <br/>
            """%(name,lon,lat,nodalprice)
            pnt = nodalpricelevel5folder.newpoint(name=name,\
                  description=description,coords=[(lon,lat)])
            pnt.style.labelstyle.color = "00000000"
            pnt.style.iconstyle.icon.href = circle
            pnt.style.iconstyle.color = color
    # Save kml file
    kml.save("result.kml")

