# -*- coding: utf-8 -*-
'''
Module for creating different PowerGAMA scenarios by scaling grid model parameters 
according to specified input.
'''

import csv
import powergama.constants as const

_QUOTINGTYPE=csv.QUOTE_MINIMAL

def saveScenario(base_grid_data, scenario_file):
    '''
    Saves the data in the current grid model to a scenario file of the 
    format used to create new scenarios
    
    Arguments:
        base_grid_data (GridData): PowerGAMA GridData object
        scenario_file (str): name of scenario (CSV) file
    
    Returns:
        None, but data is written to file
        
    '''
    
    areas_grid = base_grid_data.getAllAreas()
    consumers = base_grid_data.getConsumersPerArea()
    generators = base_grid_data.getGeneratorsPerAreaAndType()
    gentypes_grid = base_grid_data.getAllGeneratorTypes()
    data = {}
    data["demand_annual"]={}
    data["demand_profile"]={}

    for co in areas_grid:
        print("CO="+co)
        
        # Demand
        if consumers.has_key(co):
            loads_this_area = [base_grid_data.consumer.load[i] 
                for i in consumers[co]]
            load_profile = [base_grid_data.consumer.load_profile[i] 
                for i in consumers[co]]
            demand_sum_MW = float(sum(loads_this_area))
            demandprofiles_set = set(load_profile)
            demand_ref = " ".join(str(x) for x in demandprofiles_set)

            print("  demand_avg={0:12.2f} <> demand_ref={1:1s}"
                    .format(demand_sum_MW,demand_ref))
            # avg demand is in MW, whereas input file is GWh
            data["demand_annual"][co] = (
                demand_sum_MW*const.hoursperyear/const.MWh_per_GWh  )
            data["demand_profile"][co] = demand_ref
        else:
            print("  -- no consumers -- ")
            data["demand_annual"][co] = None
            data["demand_profile"][co] = None
            
        # Generation inflow, capacity and costs
        for gentype in gentypes_grid:            
            if generators.has_key(co) and generators[co].has_key(gentype):
                        
                inflow_this_area = [base_grid_data.generator.inflow_factor[i] 
                    for i in generators[co][gentype]]
                inflow_profile = [base_grid_data.generator.inflow_profile[i] 
                    for i in generators[co][gentype]]
                inflowprofiles_set = set(inflow_profile)
                inflow_ref = " ".join(str(x) for x in inflowprofiles_set)
                if len(inflow_this_area)>0:
                    inflow_avg = float(sum(inflow_this_area))/len(inflow_this_area)
                else:
                    inflow_avg = None

                storagelevel_this_area = [base_grid_data.generator.storagelevel_init[i] for i in generators[co][gentype]]
                if len(storagelevel_this_area)>0:
                    storagelevel_avg = float(sum(storagelevel_this_area))/len(storagelevel_this_area)
                else:
                    storagelevel_avg = None
					
                storval_filling_refs = [base_grid_data.generator.storagevalue_profile_filling[i] 
                    for i in generators[co][gentype]]
                storval_filling_refs_set = set(storval_filling_refs)
                storval_filling_ref = " ".join(str(x) for x in storval_filling_refs_set)
                storval_time_refs = [base_grid_data.generator.storagevalue_profile_time[i] for i in generators[co][gentype]]
                storval_time_refs_set = set(storval_time_refs)
                storval_time_ref = " ".join(str(x) for x in storval_time_refs_set)

                gencap_this_area = [base_grid_data.generator.prodMax[i] 
                    for i in generators[co][gentype]]
                gencap_MW = float(sum(gencap_this_area))
                
                storagecap_this_area = [base_grid_data.generator.storage[i] 
                    for i in generators[co][gentype]]
                storagecap_MWh = float(sum(storagecap_this_area))                

                gencost_this_area = [base_grid_data.generator.fuelcost[i] 
                    for i in generators[co][gentype]]
                gencost_avg = float(sum(gencost_this_area))/len(gencost_this_area)
                                
                storval_this_area = [
                    base_grid_data.generator.storagevalue_abs[i] 
                    for i in generators[co][gentype]
                    if base_grid_data.generator.storagevalue_abs[i] != 0]
                if not storval_this_area:
                    storval_avg = None
                else:
                    storval_avg = float(sum(storval_this_area))/len(storval_this_area)
                
                print (("  {0:1s}: cap={1:6.0f}, storage={2:1.0f}"+
                        ", fuelcost_avg={3:6.2f},\n    storval_avg={4:1s}"+
                        ", inflow_fac={5:6.2f}, inflow_ref={6:1s}"+
                        ", \n    storval_fill={7:1s} & _time={8:1s}")
                        .format(gentype,gencap_MW,storagecap_MWh,
                                gencost_avg, str(storval_avg),
                                inflow_avg,inflow_ref,
                                storval_filling_ref,storval_time_ref)
                        )
                    
                # Create (empty) elements if they haven't already been created.
                if not data.has_key("gencap_%s"%gentype):
                    data["gencap_%s"%gentype]={}
                if not data.has_key("fuelcost_%s"%gentype):
                    data["fuelcost_%s"%gentype]={}
                if not data.has_key("storage_price_%s"%gentype):
                    data["storage_price_%s"%gentype]={}
                if not data.has_key("inflow_profile_%s"%gentype):
                    data["inflow_profile_%s"%gentype]={}
                if not data.has_key("inflow_factor_%s"%gentype):
                    data["inflow_factor_%s"%gentype]={}
                if not data.has_key("storagecap_%s"%gentype):
                    data["storagecap_%s"%gentype]={}
                if not data.has_key("storage_ini_%s"%gentype):
                    data["storage_ini_%s"%gentype]={}
                if not data.has_key("storval_filling_ref_%s"%gentype):
                    data["storval_filling_ref_%s"%gentype]={}
                if not data.has_key("storval_time_ref_%s"%gentype):
                    data["storval_time_ref_%s"%gentype]={}
                    
                data["gencap_%s"%gentype][co] = gencap_MW
                data["fuelcost_%s"%gentype][co] = gencost_avg
                data["inflow_profile_%s"%gentype][co] = inflow_ref
                data["inflow_factor_%s"%gentype][co] = inflow_avg
                data["storagecap_%s"%gentype][co] = storagecap_MWh
                data["storage_ini_%s"%gentype][co] = storagelevel_avg
                data["storage_price_%s"%gentype][co] = storval_avg
                data["storval_filling_ref_%s"%gentype][co] = storval_filling_ref
                data["storval_time_ref_%s"%gentype][co] =  storval_time_ref
            else:
                print("  {0:1s}:None".format(gentype))
				
        # end collecting data
                
    # print to file
    fieldnames = data.keys()
    fieldnames.sort()
    headers = areas_grid
    headers.sort()
    headers.insert(0,"PARAMETER")
    with open(scenario_file,'wb') as csvfile:
        datawriter = csv.DictWriter(csvfile, delimiter=',',fieldnames=headers,\
                            quotechar='"', quoting=_QUOTINGTYPE)
        datawriter.writerow(dict((fn,fn) for fn in headers))
        for fn in fieldnames: 
            datarow = data[fn]
            datarow["PARAMETER"]=fn
            datawriter.writerow(datarow)

    return






def newScenario(base_grid_data, scenario_file, newfile_prefix):
    '''
    Create new input data files by up- and down-scaling data based 
    on additional input data given per area.
    
    Parameters
    ----------
        base_grid_data : GridData object        
        scenario_file : Name of scenario file (CSV)        
        newfiles_prefix : prefix used when creating new files
    
    Returns
    -------
        None, but GridData object is modified
    '''
    

    areas_grid = base_grid_data.getAllAreas()
    gentypes_grid = base_grid_data.getAllGeneratorTypes()
    gentypes_data = []
    
    with open(scenario_file,'rb') as csvfile:
        datareader = csv.DictReader(csvfile,delimiter=',',quoting=_QUOTINGTYPE)           
 
        areas_data = datareader.fieldnames[:]
        del areas_data[areas_data.index("PARAMETER")]
        areas_nodata = list(set(areas_grid).difference(areas_data))
        areas_notingrid = list(set(areas_data).difference(areas_grid))
        areas_update = list(set(areas_grid).intersection(areas_data))
        print("These areas have no scenario data (will use base values):\n"+
                str(areas_nodata))
        print("These areas (with data) are not present in the model"+
                "(will be ignored):\n" + str(areas_notingrid))
        print("Data for these areas will be updated:\n"+ str(areas_update))

        consumers = base_grid_data.getConsumersPerArea()
        generators = base_grid_data.getGeneratorsPerAreaAndType()

        # copy existing parameters
        load_new = base_grid_data.consumer.load[:]        
        loadprofiles_new = base_grid_data.consumer.load_profile[:]   
        inflow_new = base_grid_data.generator.inflow_factor[:]
        inflowprofiles_new = base_grid_data.generator.inflow_profile[:]
        gencap_new = base_grid_data.generator.prodMax[:]
        gencost_new = base_grid_data.generator.fuelcost[:]
        storagecap_new = base_grid_data.generator.storage[:]
        storagelevel_new = base_grid_data.generator.storagelevel_init[:]
        storval_basevalue_new = base_grid_data.generator.storagevalue_abs[:]
        storval_filling_ref_new = base_grid_data.generator.storagevalue_profile_filling[:]
        storval_time_ref_new =  base_grid_data.generator.storagevalue_profile_time[:]
        
        
        for row in datareader:
            parameter = row.pop('PARAMETER',None)
            
            if parameter == "demand_annual":
                print("Annual demand (GWh)")
                row = {k:(parseNum(x) if x!='' else None) 
                        for k,x in row.iteritems() if k in areas_update}
                load_new = scaleDemand(load_new,row,areas_update,consumers)
                    
            elif parameter == "demand_profile":
                row = {k:(parseId(x) if x!='' else None) 
                        for k,x in row.iteritems() if k in areas_update}
                print("Demand profile references")
                loadprofiles_new = updateDemandProfileRef(loadprofiles_new,row,areas_update,consumers)
                
            #elif parameter[:14] == "inflow_annual_":
            elif parameter[:14] == "inflow_factor_":
                gentype = parameter[14:]
                row = {k:(parseNum(x) if x!='' else None) 
                        for k,x in row.iteritems() if k in areas_update}
                print("Inflow factor for "+str(gentype))
                inflow_new = updateInflowFactor(inflow_new,row,areas_update,generators,gentype)
                
            elif parameter[:15] == "inflow_profile_":
                gentype = parameter[15:]
                row = {k:(parseId(x) if x!='' else None) 
                        for k,x in row.iteritems() if k in areas_update}
                print("Inflow profile references for "+str(gentype))
                inflowprofiles_new = updateGenProfileRef(
                    inflowprofiles_new,row,areas_update,generators,gentype)

            elif parameter[:7] == "gencap_":
                gentype = parameter[7:]
                row = {k:(parseNum(x) if x!='' else None) 
                        for k,x in row.iteritems() if k in areas_update}
                print("Generation capacities for "+str(gentype))
                if not gentype in gentypes_grid:
                    print("OBS: Generation type is not present in grid model:",
                          gentype)
                else:
                    gentypes_data.append(gentype)
                    gencap_new = scaleGencap(
                        gencap_new,row,areas_update,generators,gentype)

            elif parameter[:9] == "fuelcost_":
                gentype = parameter[9:]
                row = {k:(parseNum(x) if x!='' else None) 
                        for k,x in row.iteritems() if k in areas_update}
                print("Generation fuel costs for "+str(gentype))
                gencost_new = updateGenCost(
                    gencost_new,row,areas_update,generators,gentype)
            
            elif parameter[:14] == "storage_price_":
                gentype = parameter[14:]
                row = {k:(parseNum(x) if x!='' else None) 
                        for k,x in row.iteritems() if k in areas_update}
                print("Storage base price for "+str(gentype))
                storval_basevalue_new = updateGenCost(
                    storval_basevalue_new,row,areas_update,generators,gentype)

            elif parameter[:11] == "storagecap_":
                gentype = parameter[11:]
                row = {k:(parseNum(x) if x!='' else None) 
                        for k,x in row.iteritems() if k in areas_update}
                print("Storage capacities for "+str(gentype))
                if not gentype in gentypes_grid:
                    print("OBS: Generation type is not present in grid model:"+
                          str(gentype))
                else:
                    gentypes_data.append(gentype)
                    storagecap_new = scaleStoragecap(
                        storagecap_new,row,areas_update,generators,gentype)

            elif parameter[:12] == "storage_ini_":
                gentype = parameter[12:]
                row = {k:(parseNum(x) if x!='' else None) 
                        for k,x in row.iteritems() if k in areas_update}
                print("Initial storage filling level for "+str(gentype))
                if not gentype in gentypes_grid:
                    print("OBS: Generation type is not present in grid model:"+
                          str(gentype))
                else:
                    gentypes_data.append(gentype)
                    storagelevel_new = updateStorageLevel(
                        storagelevel_new,row,areas_update,
                        generators,gentype)

            elif parameter[:20] == "storval_filling_ref_":
                gentype = parameter[20:]
                row = {k:(parseId(x) if x!='' else None) 
                        for k,x in row.iteritems() if k in areas_update}
                print("Storage value filling level profile reference for "+
                      str(gentype))
                storval_filling_ref_new = updateGenProfileRef(
                    storval_filling_ref_new,row,areas_update,
                    generators,gentype)

            elif parameter[:17] == "storval_time_ref_":
                gentype = parameter[17:]
                row = {k:(parseId(x) if x!='' else None) 
                        for k,x in row.iteritems() if k in areas_update}
                print("Storage value time profile reference for "+
                       str(gentype))
                storval_time_ref_new =  updateGenProfileRef(
                    storval_time_ref_new,row,areas_update,generators,gentype)

            elif parameter[:6] == "IGNORE":
                print("Ignoring parameter "+parameter)
            else:
                print("WARNING! Unknown parameter: "+parameter)
                raise Exception("Unknown parameter: %s"%parameter)
        
        gentypes_nodata = list(set(gentypes_grid).difference(gentypes_data))      
        print("These generator types have no scenario data "+
                "(using base values):"+str(gentypes_nodata))


        # Updating variables
        base_grid_data.consumer.load[:] = load_new[:]
        base_grid_data.consumer.load_profile[:] = loadprofiles_new[:]
        base_grid_data.generator.inflow_factor[:] = inflow_new[:]
        base_grid_data.generator.inflow_profile[:] = inflowprofiles_new[:]
        base_grid_data.generator.prodMax[:] = gencap_new[:]
        base_grid_data.generator.fuelcost[:] = gencost_new[:]
        base_grid_data.generator.storage[:] = storagecap_new[:]
        base_grid_data.generator.storagelevel_init[:] = storagelevel_new[:]
        base_grid_data.generator.storagevalue_abs[:] = storval_basevalue_new[:]
        base_grid_data.generator.storagevalue_profile_filling[:] = storval_filling_ref_new[:]
        base_grid_data.generator.storagevalue_profile_time[:] = storval_time_ref_new[:]
        
        base_grid_data.writeGridDataToFiles(prefix=newfile_prefix)
        return



        

def scaleDemand(demand,datarow,areas_update,consumers):
    # Find all loads in this area

    print("  Total average demand before = "+str(sum(demand)))
    for co in areas_update:
        if not datarow[co] is None:
            demand_annual_GWh = datarow[co]
            demand_avg_MW = demand_annual_GWh*const.MWh_per_GWh/const.hoursperyear
            
            if not consumers.has_key(co):
                consumers[co] = []
    
            loads_this_area = [demand[i] for i in consumers[co]]
            demand_avg_MW_before = float(sum(loads_this_area))
    
            if demand_avg_MW==0:
                scalefactor = 0
            elif len(loads_this_area)==0:
                # There are no loads in this area
                print("  WARNING There are no loads to scale in area "+
                        str(co))
            elif demand_avg_MW_before==0:
                print("  WARNING Zero demand, cannot scale up in area "+
                        str(co))
                scalefactor = 1
            else:
                scalefactor = demand_avg_MW / float(demand_avg_MW_before)
                #print "Scale factor in %s is %g" % (co,scalefactor)
                
            for i in consumers[co]:
                demand[i] = demand[i]*scalefactor
            
    print("  Total average demand after = "+str(sum(demand)))
    return demand


def updateDemandProfileRef(load_profile,datarow,areas_update,consumers):

    for co in areas_update:
        if not datarow[co] is None:
            load_profile_new = datarow[co]
            for i in consumers[co]:
                load_profile[i] = load_profile_new            

    return load_profile

            
def scaleGencap(gencap,datarow,areas_update,generators,gentype):

    for co in areas_update:
        if not datarow[co] is None:
            gencap_MW_new = datarow[co]
            
            if not generators.has_key(co):
                generators[co] = {}
            if not generators[co].has_key(gentype):
                generators[co][gentype]=[]
            gencap_this_area = [gencap[i] for i in generators[co][gentype]]
            gencap_MW_before = float(sum(gencap_this_area))
    
            if gencap_MW_new==0:
                scalefactor = 0
            elif len(gencap_this_area)==0:
                print("  WARNING No generators of type %s in area %s."+
                        " Cannot scale." %(gentype,co))
            elif gencap_MW_before==0:
                print("  WARNING Zero capacity for generator type %s in "+
                        "area %s. Cannot scale." %(gentype,co))
                scalefactor = 1
            else:
                scalefactor = gencap_MW_new / float(gencap_MW_before)
                #print "Scale factor for %s in %s = %g" % (gentype,co,scalefactor)
                
            for i in generators[co][gentype]:
                gencap[i] = gencap[i]*scalefactor
            
    return gencap


def scaleStoragecap(storagecap,datarow,areas_update,generators,gentype):

    for co in areas_update:
        if not datarow[co] is None:
            storagecap_MW_new = datarow[co]
            
            if not generators.has_key(co):
                generators[co] = {}
            if not generators[co].has_key(gentype):
                generators[co][gentype]=[]
            storagecap_this_area = [storagecap[i] for i in generators[co][gentype]]
            storagecap_MW_before = float(sum(storagecap_this_area))
    
            if storagecap_MW_new==0:
                scalefactor = 0
            elif len(storagecap_this_area)==0:
                print("  WARNING No generators of type %s in area %s."+
                        " Cannot scale." %(gentype,co))
            elif storagecap_MW_before==0:
                print("  WARNING Zero capacity for generator type %s "+
                        "in area %s. Cannot scale." %(gentype,co))
                scalefactor = 1
            else:
                scalefactor = storagecap_MW_new / storagecap_MW_before
                #print "Scale factor for %s in %s = %g" % (gentype,co,scalefactor)
                
            for i in generators[co][gentype]:
                storagecap[i] = storagecap[i]*scalefactor            
    return storagecap


def updateInflowFactor(inflowfactor,datarow,areas_update,generators,gentype):
    '''Update inflow factors per type and area'''
    
    for co in areas_update:        
        if not datarow[co] is None:
            if generators.has_key(co) and generators[co].has_key(gentype):
                for i in generators[co][gentype]:
                    # all generators of this type and area are given 
                    # same inflow factor
                    inflowfactor[i] = datarow[co]
        
    return inflowfactor


def updateStorageLevel(storagelevel,datarow,areas_update,generators,gentype):
    '''Update initial storage level per type and area'''
    
    for co in areas_update:        
        if not datarow[co] is None:
            if generators.has_key(co) and generators[co].has_key(gentype):
                for i in generators[co][gentype]:
                    # all generators of this type and area are given same 
                    # inflow factor
                    storagelevel[i] = datarow[co]        
    return storagelevel


def updateGenCost(gencost,datarow,areas_update,generators,gentype):

    for co in areas_update:
        if not datarow[co] is None:
            gencost_new = datarow[co]           
            if generators.has_key(co) and generators[co].has_key(gentype):
                # and there are generators of this type
                for i in generators[co][gentype]:
                    gencost[i] = gencost_new
            
    return gencost

def NOTNEEDED_updateStoragePrice(storageprice,datarow,areas_update,generators,gentype):

    for co in areas_update:
        if not datarow[co] is None:
            storageprice_new = datarow[co]           
            if generators.has_key(co) and generators[co].has_key(gentype):
                # and there are generators of this type
                for i in generators[co][gentype]:
                    storageprice[i] = storageprice_new
            
    return storageprice


##def updateInflowProfileRef(inflow_profile,datarow,areas_update,generators,gentype):
##
##    #inflow_profile = griddata.generator.inflow_profile[:]
##
##    for co in areas_update:
##        if generators.has_key(co) and generators[co].has_key(gentype):
##            for i in generators[co][gentype]:
##                inflow_profile[i] = datarow[co]            
##
##    return inflow_profile

def updateGenProfileRef(profile_ref,datarow,areas_update,generators,gentype):
    '''Update profile per area and generator type'''
    for co in areas_update:
        if not datarow[co] is None:
            if generators.has_key(co) and generators[co].has_key(gentype):
                for i in generators[co][gentype]:
                    profile_ref[i] = datarow[co]            
    
    return profile_ref


#for debugging:
#global data
#saveScenario(data,"../scenario1/scenario_base.csv")
#newScenario(data,'../examples/scenario_new.csv','../examples/newscenario')


def parseId(num):
    '''parse ID string/integer and return a string'''    
    
    # This method is used when reading input data in order to not interpret 
    # an integer node id o e.g. 100 as "100.0", but always as "100"
    try:
        d = int(num)
    except ValueError:
        d=num
    return str(d)

def parseNum(num):
    '''parse number and return a float'''
    return float(num)

