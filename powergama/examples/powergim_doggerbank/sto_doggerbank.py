# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 12:51:18 2016

@author: hsven
"""

"""
INSTRUCTIONS:
1. Run the command line sto_runme.bat file (which invokes this script)
2. Once completed (20 hours?), run this script directly to produce .kml 
   output files (reading from the ph.csv file)
3. Copy/rename the ph.csv result file so it is not overwritten
   by the next case study.

Specify which case to investigate below:
"""
caseID = 4

import powergama
import powergama.powergim as pgim
import powergama.GIS
import numpy.random as rnd
import math

sampling = 'kmeans'
samplesize = 100 #50
rnd.seed(2016) #fixed seed  to be able to recreate results - debugging
res_file='ef.csv'

if caseID==1:
    scenario_numberof=4
    scenario_probability = [0.25,0.25,0.25,0.25]
elif caseID==2:
    scenario_numberof=4
    scenario_probability = [0.25,0.25,0.25,0.25]
elif caseID==3:
    scenario_numberof=3
    scenario_probability = [0.334,0.333,0.333]
elif caseID==4:
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
    global caseID
    global dict_data
    global grid_data
    
    if caseID==1:
        #DC max transmission capacity uncertainty
        #deterministic: 1000
        # Not a very good case as in principle this is information known
        cap_dc = {'Scenario1': 1148.0, 
                  'Scenario2': 500.0, 
                  'Scenario3': 2000.0,
                  'Scenario4': 700.0}
        
            
        dict_data['powergim']['branchtypeMaxCapacity']['dcdirect'] \
                = cap_dc[scenario_name]
    elif caseID==2:        
        # Uncertain capacity of future GB-NO connection (coming in stage 2)
        # deterministic: 0
        # Not a very good case as cable does not affect prices
        cap_gb_no = {'Scenario1': 0.0, 
                     'Scenario2': 0.0, 
                     'Scenario3': 1000.0,
                     'Scenario4': 2000.0}
        indx = grid_data.branch[(grid_data.branch['node_from']=='7') 
                                & (grid_data.branch['node_to']=='4')].index[0]
        dict_data['powergim']['branchExistingCapacity2'][indx] = cap_gb_no
    elif caseID==3:
        # Uncertain electricity market prices (scale up/down avg price in UK)
        # deterministic: 1 (scale factor)
        scaleNO = {'Scenario1': 1.0,
                   'Scenario2': 1.2,
                   'Scenario3': 1.4}
        indx = 2 #NO
        dict_data['powergim']['genCostAvg'][indx] = scaleNO[scenario_name]
        dict_data['powergim']['stage2TimeDelta'][None] = 2
        
    elif caseID==4:
        # Uncertain wind farm capacity (coming in stage 2)
        # deterministic: 1200 for both
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
    '''RUNPH call-back function to create scenario tree model
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
    '''RUNPH call-back function to create model instance
    (alternative to using scenario .dat file input)
    '''
    global caseID

    print("Creating model instance (case {}), scenario={}"
            .format(caseID,scenario_name))
    #print("Node names = {}".format(str(node_names)))
    
    #Modify dict_data according to scenario:
    dict_data = scenario_data(scenario_name)
        
    instance = sip.createConcreteModel(dict_data=dict_data)  
    #instance.write('sto_model_{}.lp'.format(scenario_name))
    instance.pprint('sto_{}_prob_{}.txt'.format(caseID,scenario_name))
    return instance


def computeSTOcosts(grid_data,dict_data,generation=None,include_om=True):
    """
    Compute costs as in objective function of the optimisation
    """
    
    # BRANCHES
    distances = grid_data.branchDistances()
    f_rate = dict_data['powergim']['financeInterestrate'][None]
    f_years = dict_data['powergim']['financeYears'][None]
    stage2delta = dict_data['powergim']['stage2TimeDelta'][None]
    cost={(1,'invest'):0.0, 
          (1,'op'):0.0, 
          (2,'invest'):0.0,
          (2,'op'):0.0}
    for count in range(grid_data.branch.shape[0]):
        br_indx = grid_data.branch.index[count]
        b = grid_data.branch.loc[br_indx]
        br_dist = distances[count]
        br_type=b['type']
        br_cap = b['capacity']
        br_num = math.ceil(br_cap/
            dict_data['powergim']['branchtypeMaxCapacity'][br_type])
        ar = 0
        salvagefactor=0
        discount_t0=1
        if b['expand']==1:
            ar = pgim.annuityfactor(f_rate,f_years)
        elif b['expand']==2:
            ar = (pgim.annuityfactor(f_rate,f_years)
                  -pgim.annuityfactor(f_rate, stage2delta))
            salvagefactor = (stage2delta/f_years)*(
                            1/((1+f_rate)**(f_years-stage2delta)))
            discount_t0 = (1/((1+f_rate)**stage2delta))
                 
        b_cost = 0
        b_cost += (dict_data['powergim']['branchtypeCost'][(br_type,'B')]
                    *br_num)
        b_cost += (dict_data['powergim']['branchtypeCost'][(br_type,'Bd')]
                    *br_dist*br_num)
        b_cost += (dict_data['powergim']['branchtypeCost'][(br_type,'Bdp')]
                    *br_dist*br_cap)
        
        #endpoints offshore (N=1) or onshore (N=0)
        N1 = dict_data['powergim']['branchOffshoreFrom'][br_indx]
        N2 = dict_data['powergim']['branchOffshoreTo'][br_indx]
        #print(br_indx,N1,N2,br_num,br_cap,br_type,b['expand'])
        for N in [N1,N2]:
            b_cost += N*(dict_data['powergim']['branchtypeCost']
                                    [(br_type,'CS')]*br_num
                        +dict_data['powergim']['branchtypeCost']
                                    [(br_type,'CSp')]*br_cap)            
            b_cost += (1-N)*(dict_data['powergim']['branchtypeCost']
                                    [(br_type,'CL')]*br_num
                        +dict_data['powergim']['branchtypeCost']
                                    [(br_type,'CLp')]*br_cap)
    
        b_cost = dict_data['powergim']['branchCostScale'][br_indx]*b_cost                

        # discount  back to t=0        
        b_cost = b_cost*discount_t0
        # O&M        
        omcost = 0     
        if include_om:
            omcost =dict_data['powergim']['omRate'][None] * ar * b_cost
        
        # subtract salvage value and add om cost
        b_cost = b_cost*(1-salvagefactor) + omcost
        
        if b['expand'] in [1,2]:
            cost[(b['expand'],'invest')] += b_cost
        
    # NODES (not yet)
        
    # GENERATION (op cost)
    if not generation is None:
        df_gen1=generation[0]
        df_gen2=generation[1]
        ar1 = pgim.annuityfactor(f_rate,stage2delta)
        ar2 = (pgim.annuityfactor(f_rate,f_years)
                  -pgim.annuityfactor(f_rate, stage2delta))
        for count in range(grid_data.generator.shape[0]):
            g_indx = grid_data.generator.index[count]
            gen1 = sum(df_gen1[(df_gen1['gen']==g_indx) 
                                & (df_gen1['time']==t)]['value'].iloc[0]
                        *dict_data['powergim']['genCostAvg'][g_indx]
                        *dict_data['powergim']['genCostProfile'][(g_indx,t)] 
                         for t in grid_data.timerange)
            gen2 = sum(df_gen2[(df_gen2['gen']==g_indx) 
                                & (df_gen2['time']==t)]['value'].iloc[0]
                        *dict_data['powergim']['genCostAvg'][g_indx]
                        *dict_data['powergim']['genCostProfile'][(g_indx,t)] 
                         for t in grid_data.timerange)
            #if model.curtailmentCost.value >0:
            #    expr += sum(curt[g,t].value*model.curtailmentCost for t in model.TIME)
            #expr += sum(gen[g,t].value*
            #                    model.genTypeEmissionRate[model.genType[g]]*model.CO2price
            #                    for t in model.TIME)
            # lifetime cost
            samplefactor = dict_data['powergim']['samplefactor'][None]
            cost[(2,'op')] += (gen1*samplefactor*ar1 
                                    + gen2*samplefactor*ar2)
        
    
    return cost
    

if __name__ == '__main__':    
    #read result data if exist:

    grid_res = sip.extractResultingGridData(grid_data=grid_data,
                                        file_ph=res_file,stage=1)
    grid_res.writeGridDataToFiles("sto_{}_".format(caseID))
    powergama.GIS.makekml("sto_{}_result_input.kml".format(caseID),
                          grid_data=grid_data,
                          nodetype='powergim_type',branchtype='powergim_type',
                          res=None,title='STO input {}'.format(caseID))
    powergama.GIS.makekml("sto_{}_result_optimal.kml".format(caseID),
                          grid_data=grid_res,
                          nodetype='powergim_type',branchtype='powergim_type',
                          res=None,title='STO result {}'.format(caseID))

    print(computeSTOcosts(grid_res,dict_data))
    for s in range(scenario_numberof):
        dict_data = scenario_data('Scenario{}'.format(s+1))
        #case 4:
        grid_data.generator.set_value(9,'pmax2',
                      value=dict_data['powergim']['genCapacity2'][9])
        
        grid_res = sip.extractResultingGridData(grid_data=grid_data,
                                            file_ph=res_file,stage=2,
                                            scenario=s+1)
        grid_res.writeGridDataToFiles("sto_{}_{}_".format(caseID,s+1))
        powergama.GIS.makekml(
            "sto_{}_result_optimal_s{}.kml".format(caseID,s+1),
            grid_data=grid_res,
            nodetype='powergim_type', branchtype='powergim_type',
            res=None, title='STO result {}/s{}'.format(caseID,s+1))
        print('Scenario {}'.format(s+1))
        print(computeSTOcosts(grid_res,dict_data,
                              generation=grid_res.generation))