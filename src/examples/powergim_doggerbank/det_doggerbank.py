# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 12:51:18 2016

@author: hsven
"""

import powergama
import powergama.powergim as pgim
import powergama.GIS
import pyomo.environ as pyo
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as rnd


plt.close('all')

# INPUT DATA

#scenario=''
scenario='4_1'
#realisation='4_1'
realisation=None
rnd.seed(2016) #fixed seed  to be able to recreate results - debugging

grid_data = powergama.GridData()
gridf='data'
if realisation is not None:
    print("Investigating specified scenario realisation: {}".format(realisation))
    gridf='{}_{}'.format(gridf,realisation)
grid_data.readSipData(nodes = gridf+"/dog_nodes.csv",
                  branches = gridf+"/dog_branches.csv",
                  generators = gridf+"/dog_generators.csv",
                  consumers = gridf+"/dog_consumers.csv")

# Profiles:
if True:
    print("\n<> Loading time-series sample...")
    samplesize = 100
    grid_data.readProfileData(filename= "data/timeseries_sample_100_rnd2016.csv",
                              timerange=range(samplesize), timedelta=1.0)
if False:
    # create new sample
    grid_data.readProfileData(filename = "data/timeseries_doggerbank.csv",
                              timerange = range(8760), timedelta = 1.0)

    print("\n<> Sampling...\n")
    samplingmethod = 'kmeans'
    samplesize = 100
    profiledata_sample = pgim.sampleProfileData(data=grid_data,
                                                samplesize=samplesize,
                                                sampling_method=samplingmethod)
    profiledata_sample.to_csv("data/timeseries_sample_100_rnd2016.csv")
    grid_data.timerange = range(profiledata_sample.shape[0])
    grid_data.profiles = profiledata_sample

sip = pgim.SipModel()
dict_data = sip.createModelData(grid_data,
                                datafile='data/dog_data_irpwind.xml',
                                maxNewBranchNum=5,maxNewBranchCap=5000)
if scenario=="":
    pass
elif scenario=="4_3":
    dict_data['powergim']['genCapacity2'][9] = 0 #9,600,1200
elif scenario=="4_2":
    dict_data['powergim']['genCapacity2'][9] = 600 #9,600,1200
elif scenario=="4_1":
   dict_data['powergim']['genCapacity2'][9] = 1200 #9,600,1200

elif scenario=="3_1":
    dict_data['powergim']['stage2TimeDelta'][None] = 2
    dict_data['powergim']['genCostAvg'][2] = 1.0
elif scenario=="3_2":
    dict_data['powergim']['stage2TimeDelta'][None] = 2
    dict_data['powergim']['genCostAvg'][2] = 1.2
elif scenario=="3_3":
    dict_data['powergim']['stage2TimeDelta'][None] = 2
    dict_data['powergim']['genCostAvg'][2] = 1.4
else:
    raise Exception("Unknown scenario")
    
model = sip.createConcreteModel(dict_data) 

print("\n<> Solving deterministic problem...\n")
opt = pyo.SolverFactory('gurobi',solver_io='python')
#model.pprint('output/det_model{}.txt'.format(scenario))

results = opt.solve(model, 
                    tee=True, #stream the solver output
                    keepfiles=False, #print the LP file for examination
                    symbolic_solver_labels=True) # use human readable names

if realisation is not None:
    powergama.GIS.makekml("output/det_result{}_{}_input.kml".format(scenario,realisation),
                          grid_data=grid_data,
                          nodetype='powergim_type',branchtype='powergim_type',
                          res=None,title='DET input {}_{}'.format(scenario,realisation))
    sip.saveDeterministicResults(model=model,
                     excel_file='output/det_result{}_{}.xlsx'
                     .format(scenario,realisation))
    grid_res2 = sip.extractResultingGridData(grid_data,model=model,stage=2)
    powergama.GIS.makekml("output/det_result{}_{}_optimal2.kml".format(scenario,realisation),
                          grid_data=grid_res2,
                          nodetype='powergim_type',branchtype='powergim_type',
                          res=None,title='DET result stage2 {}'.format(scenario))

else:
    sip.saveDeterministicResults(model=model,
                     excel_file='output/det_result{}.xlsx'.format(scenario))

    powergama.GIS.makekml("output/det_result{}_input.kml".format(scenario),
                          grid_data=grid_data,
                          nodetype='powergim_type',branchtype='powergim_type',
                          res=None,title='DET input {}'.format(scenario))
    grid_res = sip.extractResultingGridData(grid_data,model=model)
    powergama.GIS.makekml("output/det_result{}_optimal.kml".format(scenario),
                          grid_data=grid_res,
                          nodetype='powergim_type',branchtype='powergim_type',
                          res=None,title='DET result {}'.format(scenario))
    grid_res2 = sip.extractResultingGridData(grid_data,model=model,stage=2)
    powergama.GIS.makekml("output/det_result{}_optimal2.kml".format(scenario),
                          grid_data=grid_res2,
                          nodetype='powergim_type',branchtype='powergim_type',
                          res=None,title='DET result stage2 {}'.format(scenario))

        

    
    
    