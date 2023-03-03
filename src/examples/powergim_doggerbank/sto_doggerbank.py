import powergama
import powergama.powergim as pgim
import numpy.random as rnd
import math

sampling = 'kmeans'
samplesize = 100 #50
rnd.seed(2016) #fixed seed  to be able to recreate results - debugging
res_file='ef.csv'

scenario_numberof=3
scenario_probability = [0.50,0.25,0.25]


# INPUT DATA
grid_data = powergama.GridData()
grid_data.readSipData(nodes = "data/dog_nodes.csv",
                  branches = "data/dog_branches.csv",
                  generators = "data/dog_generators.csv",
                  consumers = "data/dog_consumers.csv")

# Profiles:
samplesize = 100
grid_data.readProfileData(filename= "data/timeseries_sample_100_rnd2016.csv",
                          timerange=range(samplesize), timedelta=1.0)

sip = pgim.SipModel()
dict_data = sip.createModelData(grid_data,
                                datafile='data/dog_data_irpwind.xml',
                                maxNewBranchNum=5,maxNewBranchCap=5000)


def scenario_data(scenario_name):
    """Modify data according to scenario"""
    global dict_data
    global grid_data

    # Uncertain wind farm capacity (coming in stage 2)
    # deterministic: 1200
    # Deterministicv solution connects NO-GB via Teeside C (indxC)
    #indxC = 8 #node 50 (Teeside C)
    indxD = 9 #node 60 (Teeside D)
    #capC = {'Scenario1': 1200,
    #        'Scenario2': 600,
    #        'Scenario3': 0}
    capD = {'Scenario1': 1200,
            'Scenario2': 600,
            'Scenario3': 0}
    #dict_data['powergim']['genCapacity2'][indxC] = capC[scenario_name]
    dict_data['powergim']['genCapacity2'][indxD] = capD[scenario_name]

    return dict_data


# STOCHASTIC MODEL SPECIFICS:
def pysp_scenario_tree_model_callback():
    '''call-back function to create scenario tree model
    (alternative to using ScenarioStructure.dat file input)

    The existence of the pysp_scenario_tree_model_callback in
    the model file indicates to PySP that no separate scenario
    tree structure file is required (e.g., ScenarioStructure.dat).
    '''

    print("Creating scenario tree model")
    stm = sip.createScenarioTreeModel(num_scenarios=scenario_numberof,
        probabilities=scenario_probability)
    print("Done scenario tree model.")
    return stm

def pysp_instance_creation_callback(scenario_name, node_names):
    '''call-back function to create model instance
    (alternative to using scenario .dat file input)
    '''
    print("Creating model instance, scenario={}"
            .format(scenario_name))
    #print("Node names = {}".format(str(node_names)))

    #Modify dict_data according to scenario:
    dict_data = scenario_data(scenario_name)

    instance = sip.createConcreteModel(dict_data=dict_data)
    #instance.write('sto_model_{}.lp'.format(scenario_name))
    instance.pprint('sto_prob_{}.txt'.format(scenario_name))
    return instance
