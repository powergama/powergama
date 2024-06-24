# -*- coding: utf-8 -*-
'''
PowerGAMA
=========

Created on Fri Nov 01 13:09:04 2013
by Harald G Svendsen

Licence: `The MIT License (MIT) <http://opensource.org/licenses/MIT>`_

Classes and modules
-------------------
GridData (class)
    power grid model (nodes, branches, consumers, generators) and power inflow
    and storage value 
LpProblem (class)
    LP problem, with PuLP interface to external solver (COIN-CBC)
Results (class)
    Ouptut of optimisation, methods for analysis and plotting
constants
    global constants
scenarios
    methods to manipulate grid scenarios (generation and consumption)

Example
=======

>>> import powergama

Specify time range, i.e. which rows in the time series files to use, and 
number of hours per timestep

>>> timerange = range(0,8760)
>>> timedelta = 1.0

Import grid model and time series data from files

>>> data = powergama.GridData()
>>> data.readGridData(nodes="nodes.csv",
>>>					  ac_branches="branches.csv",
>>>                   dc_branches="hvdc.csv",
>>>                   generators="generators.csv",
>>>                   consumers="consumers.csv")
>>> data.readProfileData(inflow="inflow.csv",
>>>                      demand="demand.csv",
>>>                      storagevalue_filling="profile_storval_filling.csv",
>>>                      storagevalue_time="profile_storval_time.csv",
>>>                      timerange=timerange,
>>>                      timedelta=timedelta)                      

Initiate results and LP problem objects, including name of database file
(OBS: Will replace any existing file with the same name)

>>> lp = powergama.LpProblem(data)
>>> res = powergama.Results(data,"results.sqlite3")

Solve the problem, timestep by timestep, and save results in the results 
object

>>> lp.solve(res)

Plot results

>>> res.plotMapGrid(nodetype="nodalprice")

'''

from .GridData import GridData
from .LpProblemPyomo import LpProblem
from .Results import Results, ResultsBaseClass
