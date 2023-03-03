from pathlib import Path
import powergama
import powergama.scenarios
import powergama.plots
import time
import matplotlib.pyplot as plt

TEST_DATA_ROOT_PATH = Path(__file__).parent / "test_data"

timerange=range(0,12)

data = powergama.GridData()

datapath= TEST_DATA_ROOT_PATH/"data_europe2014/"
resultpath= ""
scenarioPrefix = "2014_"
rerun = True
sqlfile = "example_europe2014.sqlite3"

data.readGridData(nodes=datapath/(scenarioPrefix + "nodes.csv"),
                  ac_branches=datapath/(scenarioPrefix + "branches.csv"),
                  dc_branches=datapath/(scenarioPrefix+ "hvdc.csv"),
                  generators=datapath/(scenarioPrefix + "generators.csv"),
                  consumers=datapath/(scenarioPrefix + "consumers.csv"))
data.readProfileData(filename=datapath/"profiles.csv",
            storagevalue_filling=datapath/"profiles_storval_filling.csv",
            storagevalue_time=datapath/"profiles_storval_time.csv",
            timerange=timerange, 
            timedelta=1.0)

lp = powergama.LpProblem(data)
if rerun:
    res = powergama.Results(data,resultpath+sqlfile,replace=True)
    start_time = time.time()
    lp.solve(res)
    end_time = time.time()
    print("\nExecution time = "+str(end_time - start_time)+"seconds")
else:
    res = powergama.Results(data,resultpath+sqlfile,replace=False)

# SOME PLOTS:
m=powergama.plots.plotMap(data,res,
                          nodetype="nodalprice",branchtype="utilisation")
m.save(resultpath+"2014Europe_results_map.html")

#res.plotMapGrid(nodetype="nodalprice",branchtype="",
#                show_node_labels=False, dotsize=10, draw_par_mer=False
#                show_Title=False)
#plt.gcf().set_size_inches(7.5,4)
#plt.savefig(resultpath+"2030_map2.pdf", bbox_inches = 'tight')

#powergama.GIS.makekml(resultpath+"2014Europe.kml",data,res=res,
#                      nodetype="nodalprice",branchtype="flow",
#                      title="2014 Europe")

#res.plotGenerationPerArea('MA',fill=True)
#res.plotEnergyMix(relative=True,showTitle=False)
#plt.gcf().set_size_inches(7.5,4)
#plt.savefig(resultpath+"2030_energymix.pdf", bbox_inches = 'tight')

#res.plotEnergySpilled(gentypes=['wind','wind_offshore','solar_pv','solar_csp','hydro'],showTitle=False)
#plt.gcf().set_size_inches(7.5,4)
#plt.savefig(resultpath+"2030_energyspilled.pdf", bbox_inches = 'tight')

# Plot production from a CSP plant with storage (Andasol, ES, node='E-187')
#genindx=data.generator.index[(data.generator.desc=='Andasol')][0]
#timemaxmin=timerange
#res.plotGeneratorOutput(generator_index=genindx,timeMaxMin=timemaxmin,
#                        showTitle=False)
#
# Plot nodal price at associated node
#res.plotStorageValues(genindx=genindx,timeMaxMin=timemaxmin,showTitle=False)
