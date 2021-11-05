import math
import pandas as pd
from .utils import annuityfactor


def computeSTOcosts(grid_data, dict_data, generation=None, include_om=True):
    """
    Compute costs as in objective function of the optimisation
    This function is used to analyse optimisation results.

    PARAMETERS
    ==========
    grid_data : powergama.grid_data
        grid object
    dict_data : dict
        dictionary holding the optimisation input data (as dictionary)
    generation : list of dataframes, one per stage
        generator operational costs, dataframe with columns ['gen','time','value']
    """

    print("Warning! powergim.computeSTOcosts is not fully implemented yet.")

    # BRANCHES
    distances = grid_data.branchDistances()
    f_rate = dict_data["powergim"]["financeInterestrate"][None]
    f_years = dict_data["powergim"]["financeYears"][None]
    stage2delta = dict_data["powergim"]["stage2TimeDelta"][None]
    cost = {(1, "invest"): 0.0, (1, "op"): 0.0, (2, "invest"): 0.0, (2, "op"): 0.0}
    for count in range(grid_data.branch.shape[0]):
        br_indx = grid_data.branch.index[count]
        b = grid_data.branch.loc[br_indx]
        br_dist = distances[count]
        br_type = b["type"]
        br_cap = b["capacity"]
        br_num = math.ceil(
            br_cap / dict_data["powergim"]["branchtypeMaxCapacity"][br_type]
        )
        ar = 0
        salvagefactor = 0
        discount_t0 = 1
        # default is that branch has not been expanded (b_stage=0)
        b_stage = 0
        if b["expand"] == 1:
            b_stage = 1
            ar = annuityfactor(f_rate, f_years)
        elif b["expand2"] == 1:
            b_stage = 2
            ar = annuityfactor(f_rate, f_years) - annuityfactor(f_rate, stage2delta)
            salvagefactor = (stage2delta / f_years) * (
                1 / ((1 + f_rate) ** (f_years - stage2delta))
            )
            discount_t0 = 1 / ((1 + f_rate) ** stage2delta)

        b_cost = 0
        b_cost += dict_data["powergim"]["branchtypeCost"][(br_type, "B")] * br_num
        b_cost += (
            dict_data["powergim"]["branchtypeCost"][(br_type, "Bd")] * br_dist * br_num
        )
        b_cost += (
            dict_data["powergim"]["branchtypeCost"][(br_type, "Bdp")] * br_dist * br_cap
        )

        # endpoints offshore (N=1) or onshore (N=0)
        N1 = dict_data["powergim"]["branchOffshoreFrom"][br_indx]
        N2 = dict_data["powergim"]["branchOffshoreTo"][br_indx]
        # print(br_indx,N1,N2,br_num,br_cap,br_type,b['expand'])
        for N in [N1, N2]:
            b_cost += N * (
                dict_data["powergim"]["branchtypeCost"][(br_type, "CS")] * br_num
                + dict_data["powergim"]["branchtypeCost"][(br_type, "CSp")] * br_cap
            )
            b_cost += (1 - N) * (
                dict_data["powergim"]["branchtypeCost"][(br_type, "CL")] * br_num
                + dict_data["powergim"]["branchtypeCost"][(br_type, "CLp")] * br_cap
            )

        b_cost = dict_data["powergim"]["branchCostScale"][br_indx] * b_cost

        # discount  back to t=0
        b_cost = b_cost * discount_t0
        # O&M
        omcost = 0
        if include_om:
            omcost = dict_data["powergim"]["omRate"][None] * ar * b_cost

        # subtract salvage value and add om cost
        b_cost = b_cost * (1 - salvagefactor) + omcost

        if b_stage > 0:
            cost[(b_stage, "invest")] += b_cost

    # NODES (not yet)

    # GENERATION (op cost)
    if not generation is None:
        df_gen1 = generation[0]
        df_gen2 = generation[1]
        ar1 = annuityfactor(f_rate, stage2delta)
        ar2 = annuityfactor(f_rate, f_years) - annuityfactor(f_rate, stage2delta)
        samplefactor = pd.Series(dict_data["powergim"]["samplefactor"])
        for count in range(grid_data.generator.shape[0]):
            g_indx = grid_data.generator.index[count]
            gen1 = sum(
                df_gen1[(df_gen1["gen"] == g_indx) & (df_gen1["time"] == t)][
                    "value"
                ].iloc[0]
                * dict_data["powergim"]["genCostAvg"][g_indx]
                * dict_data["powergim"]["genCostProfile"][(g_indx, t)]
                * samplefactor[t]
                for t in grid_data.timerange
            )
            gen2 = sum(
                df_gen2[(df_gen2["gen"] == g_indx) & (df_gen2["time"] == t)][
                    "value"
                ].iloc[0]
                * dict_data["powergim"]["genCostAvg"][g_indx]
                * dict_data["powergim"]["genCostProfile"][(g_indx, t)]
                * samplefactor[t]
                for t in grid_data.timerange
            )
            # if model.curtailmentCost.value >0:
            #    expr += sum(curt[g,t].value*model.curtailmentCost for t in model.TIME)
            # expr += sum(gen[g,t].value*
            #                    model.genTypeEmissionRate[model.genType[g]]*model.CO2price
            #                    for t in model.TIME)
            # lifetime cost

            # samplefactor = dict_data['powergim']['samplefactor'][None]
            cost[(2, "op")] += gen1 * ar1 + gen2 * ar2
    return cost
