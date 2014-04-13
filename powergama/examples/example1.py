# -*- coding: utf-8 -*-
"""
Illustrative example based on the Nordic power system

Note: The case setup is NOT intended to be realistic. 
This example is for illustration of the PowerGAMA tool only
"""

import powergama
import time
import matplotlib.pyplot as plt
import os

datapath= "data/"
timerange=range(24*2,24*3)

resultfile = 'example1.sqlite3'

# Remove the file if it already exists. BE CAREFUL!
if os.path.isfile(resultfile):
    os.remove(resultfile)

plt.close('all')

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
res = powergama.Results(data,resultfile)
res = lp.solve(res)
end_time = time.time()
print end_time - start_time, "seconds"

#Make some plots (do more from the command)
res.plotMapGrid(nodetype='nodalprice',branchtype='sensitivity',dcbranchtype='',
                    show_node_labels=False,latlon=None,timeMaxMin=None,
                    dotsize=100)
#res.plotStorageValues(2, timeMaxMin=None)
#res.plotDemandPerArea(['NO','SE','FI'],timeMaxMin=None)
#res.plotGenerationPerArea('FI',timeMaxMin=None)
#res.plotStorageFilling(1, timeMaxMin=[50,54])
#res.plotGeneratorOutput(2)
#res.plotStoragePerArea('SE',absolute=False,timeMaxMin=None)

