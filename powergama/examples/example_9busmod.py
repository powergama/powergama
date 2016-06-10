# -*- coding: utf-8 -*-
"""
Illustrative example based on the IEEE 9 bus system


linearised optimal power flow does not seem to give a very good approximation
"""

import powergama
import time

datapath= "data/9busmod_"
timerange=range(24*100,24*101)

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
res = powergama.Results(data,'example_9busmod.sqlite')

start_time = time.time()
lp.solve(res)
end_time = time.time()


