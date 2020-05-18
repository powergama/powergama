# -*- coding: utf-8 -*-
'''
Module for creating different PowerGAMA scenarios by scaling grid model parameters 
according to specified input.
'''

import pandas as pd
import powergama.constants as const
import numpy as np

_EMPTY = np.nan

def saveScenario(base_grid_data, scenario_file,verbose=True):
    '''
    Saves the data in the current grid model to a scenario file of the 
    format used to create new scenarios
    
    Parameters
    ----------
        base_grid_data :  GridData
            PowerGAMA GridData object
        scenario_file : string
            name of new scenario (CSV) file
    
        
    '''
    def printV(*args,**kwargs):
        if verbose:
            print(*args,**kwargs)
    
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
        if co in consumers:
            loads_this_area = [base_grid_data.consumer.demand_avg[i] 
                for i in consumers[co]]
            load_profile = [base_grid_data.consumer.demand_ref[i] 
                for i in consumers[co]]
            demand_sum_MW = float(sum(loads_this_area))
            demandprofiles_set = set(load_profile)
            demand_ref = " ".join(str(x) for x in demandprofiles_set)

            printV("  demand_avg={0:12.2f} <> demand_ref={1:1s}"
                    .format(demand_sum_MW,demand_ref))
            # avg demand is in MW, whereas input file is GWh
            data["demand_annual"][co] = (
                demand_sum_MW*const.hoursperyear/const.MWh_per_GWh  )
            data["demand_profile"][co] = demand_ref
        else:
            printV("  -- no consumers -- ")
            data["demand_annual"][co] = ''
            data["demand_profile"][co] = ''
            
        # Generation inflow, capacity and costs
        for gentype in gentypes_grid:            
            if co in generators and gentype in generators[co]:
                        
                inflow_this_area = [base_grid_data.generator.inflow_fac[i] 
                    for i in generators[co][gentype]]
                inflow_profile = [base_grid_data.generator.inflow_ref[i] 
                    for i in generators[co][gentype]]
                inflowprofiles_set = set(inflow_profile)
                inflow_ref = " ".join(str(x) for x in inflowprofiles_set)
                if len(inflow_this_area)>0:
                    inflow_avg = float(sum(inflow_this_area))/len(inflow_this_area)
                else:
                    inflow_avg = None

                storagelevel_this_area = [base_grid_data.generator.storage_ini[i] for i in generators[co][gentype]]
                if len(storagelevel_this_area)>0:
                    storagelevel_avg = float(sum(storagelevel_this_area))/len(storagelevel_this_area)
                else:
                    storagelevel_avg = None
					
                storval_filling_refs = [base_grid_data.generator.storval_filling_ref[i] 
                    for i in generators[co][gentype]]
                storval_filling_refs_set = set(storval_filling_refs)
                storval_filling_ref = " ".join(str(x) for x in storval_filling_refs_set)
                storval_time_refs = [base_grid_data.generator.storval_time_ref[i] for i in generators[co][gentype]]
                storval_time_refs_set = set(storval_time_refs)
                storval_time_ref = " ".join(str(x) for x in storval_time_refs_set)

                gencap_this_area = [base_grid_data.generator.pmax[i] 
                    for i in generators[co][gentype]]
                gencap_MW = float(sum(gencap_this_area))
                
                storagecap_this_area = [base_grid_data.generator.storage_cap[i] 
                    for i in generators[co][gentype]]
                storagecap_MWh = float(sum(storagecap_this_area))                

                gencost_this_area = [base_grid_data.generator.fuelcost[i] 
                    for i in generators[co][gentype]]
                gencost_avg = float(sum(gencost_this_area))/len(gencost_this_area)
                                
                storval_this_area = [
                    base_grid_data.generator.storage_price[i] 
                    for i in generators[co][gentype]
                    if base_grid_data.generator.storage_price[i] != 0]
                if not storval_this_area:
                    storval_avg = None
                else:
                    storval_avg = (float(sum(storval_this_area))
                                    /len(storval_this_area))

                deadband_this_area = [
                    base_grid_data.generator.pump_deadband[i]
                    for i in generators[co][gentype]
                    if base_grid_data.generator.pump_deadband[i] != 0]
                if not deadband_this_area:
                    pump_deadband = None
                else:
                    pump_deadband = (float(sum(deadband_this_area))
                                        /len(deadband_this_area))
                pumpcap_this_area = [base_grid_data.generator.pump_cap[i] 
                    for i in generators[co][gentype]]
                pump_capacity = float(sum(pumpcap_this_area))                

                pumpeff_this_area = [base_grid_data.generator.pump_efficiency[i] 
                    for i in generators[co][gentype]
                    if base_grid_data.generator.pump_efficiency[i] != 0]
                if not pumpeff_this_area:
                    pump_efficiency = None
                else:
                    pump_efficiency = (float(sum(pumpeff_this_area))
                                        /len(pumpeff_this_area))
                    
                printV(("  {0:1s}: cap={1:6.0f}, storage={2:1.0f}"+
                        ", fuelcost_avg={3:6.2f},\n    storval_avg={4:1s}"+
                        ", inflow_fac={5:6.2f}, inflow_ref={6:1s}"+
                        ", \n    storval_fill={7:1s} & _time={8:1s}")
                        .format(gentype,gencap_MW,storagecap_MWh,
                                gencost_avg, str(storval_avg),
                                inflow_avg,inflow_ref,
                                storval_filling_ref,storval_time_ref)
                        )
                    
                # Create (empty) elements if they haven't already been created.
                if not ("gencap_%s"%gentype) in data:
                    data["gencap_%s"%gentype]={}
                if not ("fuelcost_%s"%gentype) in data:
                    data["fuelcost_%s"%gentype]={}
                if not ("storage_price_%s"%gentype) in data:
                    data["storage_price_%s"%gentype]={}
                if not ("inflow_profile_%s"%gentype) in data:
                    data["inflow_profile_%s"%gentype]={}
                if not ("inflow_factor_%s"%gentype) in data:
                    data["inflow_factor_%s"%gentype]={}
                if not ("storagecap_%s"%gentype) in data:
                    data["storagecap_%s"%gentype]={}
                if not ("storage_ini_%s"%gentype) in data:
                    data["storage_ini_%s"%gentype]={}
                if not ("storval_filling_ref_%s"%gentype) in data:
                    data["storval_filling_ref_%s"%gentype]={}
                if not ("storval_time_ref_%s"%gentype) in data:
                    data["storval_time_ref_%s"%gentype]={}
                if not ("pump_deadband_%s"%gentype) in data:
                    data["pump_deadband_%s"%gentype]={}
                if not ("pump_capacity_%s"%gentype) in data:
                    data["pump_capacity_%s"%gentype]={}
                if not ("pump_efficiency_%s"%gentype) in data:
                    data["pump_efficiency_%s"%gentype]={}
                    
                data["gencap_%s"%gentype][co] = gencap_MW
                data["fuelcost_%s"%gentype][co] = gencost_avg
                data["inflow_profile_%s"%gentype][co] = inflow_ref
                data["inflow_factor_%s"%gentype][co] = inflow_avg
                data["storagecap_%s"%gentype][co] = storagecap_MWh
                data["storage_ini_%s"%gentype][co] = storagelevel_avg
                data["storage_price_%s"%gentype][co] = storval_avg
                data["storval_filling_ref_%s"%gentype][co] = storval_filling_ref
                data["storval_time_ref_%s"%gentype][co] =  storval_time_ref
                data["pump_deadband_%s"%gentype][co] = pump_deadband
                data["pump_capacity_%s"%gentype][co] = pump_capacity
                data["pump_efficiency_%s"%gentype][co] = pump_efficiency
            else:
                printV("  {0:1s}:None".format(gentype))
				
        # end collecting data
                
    # remove empty lines
    keys_delete=[]
    for k in data:
        #isPresent = bool([a for a in data[k].values() if a != None])
        isPresent = any(data[k].values())
        if not isPresent:
            printV("DOES NOT HAVE ",k)
            keys_delete.append(k)
    for k in keys_delete:
        printV("ignoring ",k)
        del(data[k])

    # print to file   
    df = pd.DataFrame.from_dict(data,orient='index')
    df.to_csv(scenario_file,sep=',')
    
    return data






def newScenario(base_grid_data, scenario_file, newfile_prefix=None):
    '''
    Create new dataset by modifying grid model according to scenario file
    
    This method replaces generator and consumer data according to
    information given in scenario file. Information that should not
    be replaced should be omitted from or have an empty value in the 
    scenario file; the default is to keep existing value.
    
    Parameters
    ----------
        base_grid_data : GridData
            PowerGAMA grid model object used as basis for modifications 
        scenario_file : string
            Name of scenario file (CSV)
        newfiles_prefix : string
            Prefix used when creating new files. New files will be 
            the same as old files with this additional prefix
    
    '''
    

    areas_grid = base_grid_data.getAllAreas()
    gentypes_grid = base_grid_data.getAllGeneratorTypes()
    gentypes_data = []
    
    df = pd.read_csv(scenario_file,index_col=0)
    areas_data = list(df.columns.values)
    datadict = df.transpose().to_dict()
    
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
    load_new = base_grid_data.consumer.demand_avg.copy()      
    loadprofiles_new = base_grid_data.consumer.demand_ref.copy()
    inflow_new = base_grid_data.generator.inflow_fac.copy()
    inflowprofiles_new = base_grid_data.generator.inflow_ref.copy()
    gencap_new = base_grid_data.generator.pmax.copy()
    gencost_new = base_grid_data.generator.fuelcost.copy()
    storagecap_new = base_grid_data.generator.storage_cap.copy()
    storagelevel_new = base_grid_data.generator.storage_ini.copy()
    storval_basevalue_new = base_grid_data.generator.storage_price.copy()
    storval_filling_ref_new = base_grid_data.generator.storval_filling_ref.copy()
    storval_time_ref_new =  base_grid_data.generator.storval_time_ref.copy()
    
        
    for parameter in datadict:
        row = datadict[parameter]
        #print("parameter={}  row.items={}".format(parameter,row.items()))
        #print(row)
        
        if parameter == "demand_annual":
            print("Annual demand (GWh)")
            row = {k:(_parseNum(x) if x!='' else None) 
                    for k,x in row.items() if k in areas_update}
            load_new = _scaleDemand(load_new,row,areas_update,consumers)
                
        elif parameter == "demand_profile":
            row = {k:(_parseId(x) if x!='' else None) 
                    for k,x in row.items() if k in areas_update}
            print("Demand profile references")
            loadprofiles_new = _updateDemandProfileRef(loadprofiles_new,row,areas_update,consumers)
            
        #elif parameter[:14] == "inflow_annual_":
        elif parameter[:14] == "inflow_factor_":
            gentype = parameter[14:]
            row = {k:(_parseNum(x) if x!='' else None) 
                    for k,x in row.items() if k in areas_update}
            print("Inflow factor for "+str(gentype))
            inflow_new = _updateInflowFactor(inflow_new,row,areas_update,generators,gentype)
            
        elif parameter[:15] == "inflow_profile_":
            gentype = parameter[15:]
            row = {k:(_parseId(x) if x!='' else None) 
                    for k,x in row.items() if k in areas_update}
            print("Inflow profile references for "+str(gentype))
            inflowprofiles_new = _updateGenProfileRef(
                inflowprofiles_new,row,areas_update,generators,gentype)

        elif parameter[:7] == "gencap_":
            gentype = parameter[7:]
            row = {k:(_parseNum(x) if x!='' else None) 
                    for k,x in row.items() if k in areas_update}
            print("Generation capacities for "+str(gentype))
            if not gentype in gentypes_grid:
                print("OBS: Generation type is not present in grid model:",
                      gentype)
            else:
                gentypes_data.append(gentype)
                gencap_new = _scaleGencap(
                    gencap_new,row,areas_update,generators,gentype)

        elif parameter[:9] == "fuelcost_":
            gentype = parameter[9:]
            row = {k:(_parseNum(x) if x!='' else None) 
                    for k,x in row.items() if k in areas_update}
            print("Generation fuel costs for "+str(gentype))
            gencost_new = _updateGenCost(
                gencost_new,row,areas_update,generators,gentype)
        
        elif parameter[:14] == "storage_price_":
            gentype = parameter[14:]
            #print("row.items={}".format(row.items()))
            row = {k:(_parseNum(x) if x!='' else None) 
                    for k,x in row.items() if k in areas_update}
            print("Storage base price for "+str(gentype))
            storval_basevalue_new = _updateGenCost(
                storval_basevalue_new,row,areas_update,generators,gentype)

        elif parameter[:11] == "storagecap_":
            gentype = parameter[11:]
            row = {k:(_parseNum(x) if x!='' else None) 
                    for k,x in row.items() if k in areas_update}
            print("Storage capacities for "+str(gentype))
            if not gentype in gentypes_grid:
                print("OBS: Generation type is not present in grid model:"+
                      str(gentype))
            else:
                gentypes_data.append(gentype)
                storagecap_new = _scaleStoragecap(
                    storagecap_new,row,areas_update,generators,gentype)

        elif parameter[:12] == "storage_ini_":
            gentype = parameter[12:]
            row = {k:(_parseNum(x) if x!='' else None) 
                    for k,x in row.items() if k in areas_update}
            print("Initial storage filling level for "+str(gentype))
            if not gentype in gentypes_grid:
                print("OBS: Generation type is not present in grid model:"+
                      str(gentype))
            else:
                gentypes_data.append(gentype)
                storagelevel_new = _updateStorageLevel(
                    storagelevel_new,row,areas_update,
                    generators,gentype)

        elif parameter[:20] == "storval_filling_ref_":
            gentype = parameter[20:]
            row = {k:(_parseId(x) if x!='' else None) 
                    for k,x in row.items() if k in areas_update}
            print("Storage value filling level profile reference for "+
                  str(gentype))
            storval_filling_ref_new = _updateGenProfileRef(
                storval_filling_ref_new,row,areas_update,
                generators,gentype)

        elif parameter[:17] == "storval_time_ref_":
            gentype = parameter[17:]
            row = {k:(_parseId(x) if x!='' else None) 
                    for k,x in row.items() if k in areas_update}
            print("Storage value time profile reference for "+
                   str(gentype))
            storval_time_ref_new =  _updateGenProfileRef(
                storval_time_ref_new,row,areas_update,generators,gentype)

        elif parameter[:14] == "pump_capacity_":
            gentype = parameter[14:]
            row = {k:(_parseNum(x) if x!='' else None) 
                    for k,x in row.items() if k in areas_update}
            print("Pump capacity for "+str(gentype))
            print("** WARNING ** - NOT IMPLEMENTED !!")

        elif parameter[:14] == "pump_deadband_":
            gentype = parameter[14:]
            row = {k:(_parseNum(x) if x!='' else None) 
                    for k,x in row.items() if k in areas_update}
            print("Pump deadband for "+str(gentype))
            print("** WARNING ** - NOT IMPLEMENTED !!")

        elif parameter[:16] == "pump_efficiency_":
            gentype = parameter[16:]
            row = {k:(_parseNum(x) if x!='' else None) 
                    for k,x in row.items() if k in areas_update}
            print("Pump efficiency for "+str(gentype))
            print("** WARNING ** - NOT IMPLEMENTED !!")
        
        elif parameter[:6] == "IGNORE":
            print("Ignoring parameter "+parameter)
        else:
            print("WARNING! Unknown parameter: "+parameter)
            raise Exception("Unknown parameter: %s"%parameter)
    

    gentypes_nodata = list(set(gentypes_grid).difference(gentypes_data))      
    print("These generator types have no scenario data "+
            "(using base values):"+str(gentypes_nodata))


    # Updating variables
    base_grid_data.consumer.demand_avg = load_new[:]
    base_grid_data.consumer.demand_ref = loadprofiles_new[:]
    base_grid_data.generator.inflow_fac = inflow_new[:]
    base_grid_data.generator.inflow_ref = inflowprofiles_new[:]
    base_grid_data.generator.pmax = gencap_new[:]
    base_grid_data.generator.fuelcost = gencost_new[:]
    base_grid_data.generator.storage_cap = storagecap_new[:]
    base_grid_data.generator.storage_ini = storagelevel_new[:]
    base_grid_data.generator.storage_price = storval_basevalue_new[:]
    base_grid_data.generator.storval_filling_ref = storval_filling_ref_new[:]
    base_grid_data.generator.storval_time_ref = storval_time_ref_new[:]
    
    if newfile_prefix is not None:
        base_grid_data.writeGridDataToFiles(prefix=newfile_prefix)
    return base_grid_data



        

def _scaleDemand(demand,datarow,areas_update,consumers):
    # Find all loads in this area

    print("  Total average demand before = "+str(sum(demand)))
    for co in areas_update:
        if not datarow[co] is _EMPTY:
            demand_annual_GWh = datarow[co]
            demand_avg_MW = demand_annual_GWh*const.MWh_per_GWh/const.hoursperyear
            
            if not co in consumers:
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


def _updateDemandProfileRef(load_profile,datarow,areas_update,consumers):

    for co in areas_update:
        if not datarow[co] is _EMPTY:
            load_profile_new = datarow[co]
            if not co in consumers:
                consumers[co] = []
                print("  WARNING No consumers in {}".format(co))
                print(datarow)
            for i in consumers[co]:
                load_profile[i] = load_profile_new

    return load_profile

            
def _scaleGencap(gencap,datarow,areas_update,generators,gentype):

    for co in areas_update:
        if not datarow[co] is _EMPTY:
            gencap_MW_new = datarow[co]
            
            if not co in generators:
                generators[co] = {}
            if not gentype in generators[co]:
                generators[co][gentype]=[]
            gencap_this_area = [gencap[i] for i in generators[co][gentype]]
            gencap_MW_before = float(sum(gencap_this_area))
    
            if gencap_MW_new==0:
                scalefactor = 0
            elif len(gencap_this_area)==0:
                print(("  WARNING No generators of type {0} in area {1}."+
                        " Cannot scale.").format(gentype,co))
            elif gencap_MW_before==0:
                print(("  WARNING Zero capacity for generator type {0:1s} in "+
                        "area {1:1s}. Cannot scale.").format(gentype,co))
                scalefactor = 1
            else:
                scalefactor = gencap_MW_new / float(gencap_MW_before)
                #print "Scale factor for %s in %s = %g" % (gentype,co,scalefactor)
                
            for i in generators[co][gentype]:
                gencap[i] = gencap[i]*scalefactor
            
    return gencap


def _scaleStoragecap(storagecap,datarow,areas_update,generators,gentype):

    for co in areas_update:
        if not datarow[co] is _EMPTY:
            storagecap_MW_new = datarow[co]
            
            if not co in generators:
                generators[co] = {}
            if not gentype in generators[co]:
                generators[co][gentype]=[]
            storagecap_this_area = [storagecap[i] for i in generators[co][gentype]]
            storagecap_MW_before = float(sum(storagecap_this_area))
    
            if storagecap_MW_new==0:
                scalefactor = 0
            elif len(storagecap_this_area)==0:
                print(("  WARNING No generators of type {} in area {}."+
                        " Cannot scale.").format(gentype,co))
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


def _updateInflowFactor(inflowfactor,datarow,areas_update,generators,gentype):
    '''Update inflow factors per type and area'''
    
    for co in areas_update:        
        if not datarow[co] is _EMPTY:
            if co in generators and gentype in generators[co]:
                for i in generators[co][gentype]:
                    # all generators of this type and area are given 
                    # same inflow factor
                    inflowfactor[i] = datarow[co]
        
    return inflowfactor


def _updateStorageLevel(storagelevel,datarow,areas_update,generators,gentype):
    '''Update initial storage level per type and area'''
    
    for co in areas_update:        
        if not datarow[co] is _EMPTY:
            if co in generators and gentype in generators[co]:
                for i in generators[co][gentype]:
                    # all generators of this type and area are given same 
                    # inflow factor
                    storagelevel[i] = datarow[co]        
    return storagelevel


def _updateGenCost(gencost,datarow,areas_update,generators,gentype):

    for co in areas_update:
        if not datarow[co] is _EMPTY:
            gencost_new = datarow[co]           
            if co in generators and gentype in generators[co]:
                # and there are generators of this type
                for i in generators[co][gentype]:
                    gencost[i] = gencost_new
            
    return gencost

def _NOTNEEDED_updateStoragePrice(storageprice,datarow,areas_update,generators,gentype):

    for co in areas_update:
        if not datarow[co] is _EMPTY:
            storageprice_new = datarow[co]           
            if co in generators and gentype in generators[co]:
                # and there are generators of this type
                for i in generators[co][gentype]:
                    storageprice[i] = storageprice_new
            
    return storageprice


def _updateGenProfileRef(profile_ref,datarow,areas_update,generators,gentype):
    '''Update profile per area and generator type'''
    for co in areas_update:
        if not datarow[co] is _EMPTY:
            if co in generators and gentype in generators[co]:
                for i in generators[co][gentype]:
                    profile_ref[i] = datarow[co]            
    
    return profile_ref


def _parseId(value):
    '''parse ID string/integer and return a string   
    
     This method is used when reading input data in order to not interpret 
     an integer node id o e.g. 100 as "100.0", but always as "100"
     '''
    try:
        #if it is an integer number
        s = str(int(value))
    except ValueError:
        #if not then it must have been a string (or empty)
        s=value
    return s

def _parseNum(num):
    '''parse number and return a float'''
    return float(num)

