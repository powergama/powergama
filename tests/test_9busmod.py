"""
Integration test: IEEE 9 bus system
"""

import powergama
import time
from pathlib import Path

TEST_DATA_ROOT_PATH = Path(__file__).parent / "test_data"

datapath= TEST_DATA_ROOT_PATH/"data_9bus"
timerange=range(24*100,24*101)

data = powergama.GridData()

data.readGridData(nodes=datapath/"9busmod_nodes.csv",
                  ac_branches=datapath/"9busmod_branches.csv",
                  dc_branches=None,
                  generators=datapath/"9busmod_generators.csv",
                  consumers=datapath/"9busmod_consumers.csv")
data.readProfileData(filename=datapath/"9busmod_profiles.csv",
            storagevalue_filling=datapath/"9busmod_profiles_storval_filling.csv",
            storagevalue_time=datapath/"9busmod_profiles_storval_time.csv",
            timerange=timerange, 
            timedelta=1.0)

lp = powergama.LpProblem(data)
res = powergama.Results(data,'example_9busmod.sqlite')

start_time = time.time()
lp.solve(res,solver="cbc")
end_time = time.time()

# TODO check if result is as expected


