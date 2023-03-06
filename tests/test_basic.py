# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 09:38:49 2016

@author: hsven
"""
import matplotlib.pyplot as plt


def testResultGetters(data,res):
    '''
    Test functions for retrieving results
    
    Parameters
    ----------
    data : powergama.GridData object
        object holding grid model
    res : powergama.Results object
        object holding simulation results
    '''
    area=data.getAllAreas()[0]
    gentype = data.getAllGeneratorTypes(sort='fuelcost')[0]
    if res.getAreaPrices(area):
        print("getAreaPrices .. OK")
    if res.getAreaPricesAverage():
        print("getAreaPricesAverage .. OK")
    if res.getAverageBranchFlows():
        print("getAverageBranchFlows .. OK")
    if res.getAverageBranchSensitivity().tolist():
        print("getAverageBranchSensitivity .. OK")
    if res.getAverageEnergyBalance().tolist():
        print("getAverageEnergyBalance .. OK")
    if res.getAverageImportExport(area):
        print("getAverageImportExport .. OK")
    if res.getAverageInterareaBranchFlow():
        print("getAverageInterareaBranchFlow .. OK")
    if res.getAverageNodalPrices().tolist():
        print("getAverageNodalPrices .. OK")
    if res.getAverageUtilisation().tolist():
        print("getAverageUtilisation .. OK")
    if res.getDemandPerArea(area):
        print("getDemandPerArea .. OK")
    if res.getEnergyBalanceInArea(area,spillageGen=[gentype]).size > 0:
        print("getEnergyBalanceInArea .. OK")
    if res.getGeneratorOutputSumPerArea():
        print("getGeneratorOutputSumPerArea .. OK")
    if res.getGeneratorSpilled(0):
        print("getGeneratorSpilled .. OK")
    if res.getGeneratorSpilledSums():
        print("getGeneratorSpilledSums .. OK")
    if res.getGeneratorStorageAll(res.timerange[0]):
        print("getGeneratorStorageAll .. OK")
    if res.getGeneratorStorageValues(res.timerange[0]):
        print("getGeneratorStorageValues .. OK")
    if res.getLoadheddingInArea(area).tolist():
        print("getLoadsheddingInArea .. OK")
    if res.getLoadheddingSums():
        print("getLoadsheddingSums .. OK")
    if res.getLoadsheddingPerNode():
        print("getLoadsheddingPerNode .. OK")
    if res.getNetImport(area):
        print("getNetImport .. OK")
    if res.getNodalPrices(0).tolist():
        print("getNodalPrices .. OK")
    try:
        # list is empty if there is no storage, so can't just check bool(list)
        res.getStorageFillingInAreas([area],gentype)
        print("getStorageFillingInAreas .. OK")
    except:
        raise
    if res.getSystemCost():
        print("getSystemCostFast .. OK")
    
    
def testPlots(data,res):
    '''
    Test PowerGAMA plotting functions
    
    Parameters
    ----------
    data : powergama.GridData object
        object holding grid model
    res : powergama.Results object
        object holding simulation results
    '''
    plt.close('all')
    area=data.getAllAreas()[0]
    
    # Try plots
    res.plotAreaPrice([area])    
    res.plotDemandAtLoad(0)    
    res.plotDemandPerArea([area])
    res.plotEnergyMix([area])
    res.plotEnergySpilled([area])
    indices = data.getIdxConsumersWithFlexibleLoad()
    if indices:
        res.plotFlexibleLoadStorageValues(indices[0])
    res.plotGenerationPerArea(area)
    res.plotGenerationScatter(area)
    res.plotGeneratorOutput(0)
    res.plotMapGrid(nodetype='nodalprice',branchtype='sensitivity',
                    dotsize=40,show_node_labels=False,filter_branch=[0,1])
    res.plotNodalPrice(0)                
    res.plotRelativeGenerationCapacity(tech=data.getAllGeneratorTypes()[0])
    res.plotRelativeLoadDistribution()
    indices = data.getIdxGeneratorsWithStorage()
    if indices:
        res.plotStorageFilling(indices[0])
        res.plotStorageValues(indices[0])
    res.plotStoragePerArea(area)  
    res.plotTimeseriesColour(areas=[area],value='nodalprice')

def testKmlExport(data,res,filename):
    '''
    Test export of Google Earth file
    
    Parameters
    ----------
    data : powergama.GridData object
        object holding grid model
    res : powergama.Results object
        object holding simulation results
    filename : string
        name of KML file to be created
    '''
    import powergama.GIS
    powergama.GIS.makekml(filename,grid_data=data,res=res,
                          nodetype='nodalprice',branchtype='flow')   


if __name__=='__main__':
    try:
        data
        res
        testResultGetters(data,res)
        testPlots(data,res)
        testKmlExport('testfile.kml',data,res)
        
    except NameError:
        print("Must first define GridData and Results objects 'data' and 'res'")
        
    
