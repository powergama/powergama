# -*- coding: utf-8 -*-
"""
Illustrative example based on the Nordic power system

Note: The case setup is NOT intended to be realistic. 
This example is for illustration of the PowerGAMA tool only
"""

import powergama
from powergama import makekml
import time

datapath= "data/"
timerange=range(0,24*3)
# timerange=range(0,3)

data = powergama.GridData()

data.readGridData(nodes=datapath+"nodes.csv",
                  ac_branches=datapath+"branches.csv",
                  dc_branches=None,
                  generators=datapath+"generators.csv",
                  consumers=datapath+"consumers.csv")
data.readProfileData(inflow=datapath+"profiles_inflow.csv",
            demand=datapath+"profiles_demand.csv",
            storagevalue_filling=datapath+"profiles_storval_filling.csv",
            storagevalue_time=datapath+"profiles_storval_time.csv",
            timerange=timerange, 
            timedelta=1.0)

lp = powergama.LpProblem(data)
start_time = time.time()
res = lp.solve()
end_time = time.time()
print end_time - start_time, "seconds"
makekml(res,timestep=0)

# Save results to file (for later analysis)
#import pickle
#import cPickle as pickle
#with open('saved_2w.pickle','wb') as f:
#    pickle.dump({'data':data,'lp':lp,'res':res},f)

## Load previously generated results from file
#import powergama
#with open('scen1.pickle','rb') as f:
#    pg = pickle.load(f)
#res = pg['res']
#data = pg['data']
#lp = pg['lp']

