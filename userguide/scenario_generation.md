# Scenario generation tool

In order to simplify the process of generating scenarios with different
generation mixes and demands, the package includes a scenario module that
can be used to save the loaded grid model to a scenario file. This is a
summary file that includes demand per area, generation capacities per
type and area etc. Exporting the grid model to a scenario file can be
very useful for checking that the dataset is consistent with the
scenario being studied.

The key functions are :

-   powergama.scenarios.saveScenario(\...)

-   powergama.scenarios.newScenario(\...)

## Saving grid model to scenario file

To load an existing grid model and export a scenario file
("scenario.csv"), run code similar to the following:

    >>>import powergama
    >>>import powergama.scenarios

    >>>gridmodel = powergama.GridData()
    >>>gridmodel.readGridData(nodes="nodes.csv",
                              ac_branches="branches.csv",
                              dc_branches="hvdc.csv",
                              generators="generators.csv",
                              consumers="consumers.csv")
    >>>gridmodel.readProfileData(filename="profiles.csv",
                   storagevalue_filling="profiles_storval_filling.csv",
                   storagevalue_time="profiles_storval_time.csv",
                   timerange=range(0,8760), 
                   timedelta=1.0)

    >>>powergama.scenarios.saveScenario(gridmodel, 
                                        scenario_file= "scenario.csv")

## Create modified dataset using scenario file

In order to create a scenario file, the simplest way is to save an
existing grid model to a scenario file as shown above. Then, it can be
opened in a spreadsheet editor and modified according to the
specifications of the new scenario. Values that should not be modified
should be left blank. Irrelevant rows can be removed. In general, the
newScenario function only modifies data where information is provided in
the scenario file.

Be careful with profile reference data, as the output created by
saveScenario will join together all references present for each country,
and cannot be used directly when creating new scenarios. The information
may be useful in validating the dataset, but is not useful for creating
new scenarios. If a single reference is used for the country (e.g. all
wind generators in a country), then it is ok to include this (e.g.
demand reference), but for generator inflow and storage value
references, it may be necessary to modify these values directly in the
data file (if modification is required). If no modifications are
required, these rows should be removed. This concerns the following rows
in the scenario file:

-   demand_ref

-   inflow_ref\_\<gentype\>

-   storval_time_ref \_\<gentype\>

-   storval_filling_ref\_\<gentype\>

Once a modified scenario file has been created ("scenario_new.csv"), run
code similar to the following in order to create new input data files:

    >>>import powergama
    >>>import powergama.scenarios

    >>>gridmodel = powergama.GridData()
    >>>gridmodel.readGridData(nodes="nodes.csv",
                              ac_branches="branches.csv",
                              dc_branches="hvdc.csv",
                              generators="generators.csv",
                              consumers="consumers.csv")
    >>>gridmodel.readProfileData(filename="profiles.csv",
                   storagevalue_filling="profiles_storval_filling.csv",
                   storagevalue_time="profiles_storval_time.csv",
                   timerange=range(0,8760), 
                   timedelta=1.0)
                
    >>>powergama.scenarios.newScenario(gridmodel, 
                                       scenario_file="scenario_new.csv", 
                                       newfile_prefix="new_")

The new input files will have the same names as the original, but with
the prefix "new\_", e.g. "new_nodes.csv".

Now, you can run a new simulation with these new input files instead of
the original ones.
