import copy
import os

import numpy as np
import pandas as pd

from .specify_storage import load_table_from_res

"""
Functions to get statistics on load shedding from a simulation.
"""


def get_load_shedding_data(res_file, data, verbose=False, timeminmax=None, res_nodes=None, node_subset=None):
    if res_nodes is None:
        res_nodes = load_table_from_res(res_file, "Res_Nodes")

    if node_subset is not None:
        res_nodes = res_nodes[res_nodes.indx.isin(node_subset)]
        node_list = node_subset
    else:
        node_list = list(range(data.numNodes()))

    N_steps = data.timerange[-1] + 1  # Assumes we start at 0
    summary_stats = []
    for idx in node_list:
        if verbose:
            print(f"\nNode: {idx}")
        res_idx = res_nodes[res_nodes.indx == idx]
        if timeminmax is not None:
            N_steps = timeminmax[1] - timeminmax[0] + 1
            res_idx = res_idx[(res_idx.timestep <= timeminmax[1]) & (res_idx.timestep >= timeminmax[0])]
        if not (len(res_idx) == N_steps):
            raise Exception("Programming error:     Wrong length")
        sum_shed = np.sum(res_idx.loadshed)
        if verbose:
            print(f"Sum shed: {sum_shed}")
        mean_shed = sum_shed / N_steps
        if verbose:
            print(f"Mean shed: {mean_shed}")
        num_hours_shed = np.count_nonzero(res_idx.loadshed)
        if verbose:
            print(f"Hours shed: {num_hours_shed}")
        shed_probability = num_hours_shed / N_steps
        if verbose:
            print(f"Probability of shedding: {shed_probability}")
        summary_stats.append(
            {
                "indx": idx,
                "sum_shed": sum_shed,
                "mean_shed": mean_shed,
                "hours_shed": num_hours_shed,
                "shed_prob": shed_probability,
                "yearly_hours_shed": shed_probability * 8760,
            }
        )
    shed_data = pd.DataFrame(summary_stats)

    timestep_shed = res_nodes.groupby("timestep")["loadshed"].agg(np.sum)
    fraction_hours_with_shedding = np.sum(timestep_shed > 0) / len(timestep_shed)
    return fraction_hours_with_shedding, shed_data


def get_sum_demand(consumers, full_profiles, node_list=None):
    if node_list is not None:
        consumers = consumers[consumers.node.isin(node_list)]
    sum_demand = 0
    for ii, row in consumers.iterrows():
        ii_tot_demand = row["demand_avg"] * full_profiles.get_multiplier_demand()[row["demand_ref"]]
        sum_demand += ii_tot_demand
    return sum_demand


def get_summary_stats(
    fhwue, unserved_data, consumers, full_profiles, node_list=None, sdigits=4, verbose=True, years=30
):
    # fhwue: fraction_hours_with_undelivered_energy
    fhwue_year = fhwue * 24 * 365
    nfhwu = np.mean(unserved_data["shed_prob"])
    nfhwu_year = nfhwu * 24 * 365
    sum_demand = get_sum_demand(consumers=consumers, full_profiles=full_profiles, node_list=node_list)
    undelivered_fraction = np.sum(unserved_data["sum_shed"]) / sum_demand
    undelivered = np.sum(unserved_data["sum_shed"]) / (1000 * years)  # GWh per year
    if verbose:
        print("Fraction hours with underdelivery: %s" % np.round(fhwue, sdigits))
        print("Average hours with underdelivery per year: %s" % np.round(fhwue_year, sdigits))
        print("Average over nodes fraction hours with underdelivery: %s" % np.round(nfhwu, sdigits))
        print("Average over nodes hours with underdelivery per year: %s" % np.round(nfhwu_year, sdigits))
        print(
            "Fraction energy underdelivered: %s" % np.round(undelivered_fraction, sdigits)
        )  # unserved energy / tot energy demand
        print("Total energy undelivered (GWh per year): %s" % np.round(undelivered, sdigits))
    res = [fhwue, fhwue_year, nfhwu, nfhwu_year, undelivered_fraction, undelivered]
    return res


def preprocess_failure_simulation(res_nodes_base, failure_sim_directory):
    # Start with the base case array
    # Whenever there is a failure situation, and the load shedding is above zero,
    #     replace those timesteps in the base case
    csv_file = f"{failure_sim_directory}.csv"
    if os.path.isfile(csv_file):
        print("Reading preprocessed simulation from file")
        failure_case = pd.read_csv(csv_file)
        return failure_case
    else:
        print(f"Preprocessing failure simulation {failure_sim_directory}")

    failure_case = copy.deepcopy(res_nodes_base)

    num_situations = len(os.listdir(failure_sim_directory))

    for ii in range(num_situations):
        res_table = load_table_from_res(
            f"{failure_sim_directory}/failure_{ii}/failure_case_combined.sqlite3", "Res_Nodes"
        )
        if len(res_table[res_table["loadshed"] > 0]) > 0:

            min_ts = res_table["timestep"].min()
            max_ts = res_table["timestep"].max()

            start_idx = failure_case[failure_case["timestep"] == min_ts].index.min()
            end_idx = failure_case[failure_case["timestep"] == max_ts].index.max()

            failure_case[start_idx : end_idx + 1] = res_table.drop(columns="fault_start")
    failure_case.to_csv(csv_file, index=False)
    return failure_case
