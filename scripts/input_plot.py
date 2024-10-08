import powergama
import pathlib
import powergama.plots as pgplot
import folium
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import gurobipy
import os

from tests.test_9busmod import datapath

# This script reads and plots input data for the PowerGAMA model. The plots are saved in the folder input_plots.


year=2020

def read_data(year):
    scenario_power_syst_data = pathlib.Path(f"C:/Users/einar/OneDrive - NTNU/NTNU/9. semester/PowerGAMA/scripts/nordic/nordic/data_{year}")
    file_storval_filling = pathlib.Path(f"C:/Users/einar/OneDrive - NTNU/NTNU/9. semester/PowerGAMA/scripts/nordic/nordic/data_storagevalues/profiles_storval_filling.csv")
    file_30y_profiles = pathlib.Path(f"C:/Users/einar/OneDrive - NTNU/NTNU/9. semester/PowerGAMA/scripts/nordic/nordic/data_timeseries/timeseries_profiles/timeseries_profiles.csv")

    # Direkte fra raw filen over powerplants fra powerplantmatching. Brukt for å plotte powerplants på kartet med riktig lat long posisjon.
    powerplants = pd.read_csv(f"C:/Users/einar/OneDrive - NTNU/NTNU/9. semester/PowerGAMA/plots/data/All_powerplant_powerplantmatching.csv")
    nordic_powerplants = powerplants[powerplants['Country'].isin(['Norway', 'Sweden', 'Finland'])]

    data = powergama.GridData()
    data.readGridData(nodes=scenario_power_syst_data/"node.csv",
                        ac_branches=scenario_power_syst_data/"branch.csv",
                        dc_branches=scenario_power_syst_data/"dcbranch.csv",
                        generators=scenario_power_syst_data/"generator.csv",
                        consumers=scenario_power_syst_data/"consumer.csv")

    return scenario_power_syst_data, data, nordic_powerplants, file_storval_filling, file_30y_profiles
scenario_power_syst_data, data, nordic_powerplants, file_storval_filling, file_30y_profiles = read_data(year)


# Nodes
node_file = scenario_power_syst_data / "node.csv"
nodes_df = pd.read_csv(node_file)

mean_lat_nodes = nodes_df['lat'].mean()
mean_lon_nodes = nodes_df['lon'].mean()
map_nodes = folium.Map(location=[mean_lat_nodes, mean_lon_nodes], zoom_start=5)

for idx, row in nodes_df.iterrows():
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=f"Node: {row['id']}",
    ).add_to(map_nodes)
map_nodes.save(os.path.join('../input_plots', 'nodes_map.html'))  # Save the map as an HTML file


# Producers
mean_lat_prod = nordic_powerplants['lat'].mean()
mean_lon_prod = nordic_powerplants['lon'].mean()
map_prod = folium.Map(location=[mean_lat_prod, mean_lon_prod], zoom_start=5)

for idx, row in nordic_powerplants.iterrows():
    # Create blue circles for producers with size depending on capacity
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=row['Capacity'] / 100,  # Scale the radius by capacity (adjust if necessary)
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.6,
        popup=f"Producer Capacity: {row['Capacity']} MW"
    ).add_to(map_prod)
map_prod.save(os.path.join('../input_plots', 'powergrid_producers_map.html'))                                           # Save the map as an HTML file


# Consumers
consumers_file = scenario_power_syst_data / "consumer.csv"
consumers_df = pd.read_csv(consumers_file)
consumers_with_location = pd.merge(consumers_df, nodes_df[['id', 'lat', 'lon']], left_on='node', right_on='id')         # Merge with nodes to get location
map_consumers = folium.Map(location=[mean_lat_nodes, mean_lon_nodes], zoom_start=5)


for idx, row in consumers_with_location.iterrows():
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=f"Average demand: {row['demand_avg']}",
    ).add_to(map_consumers)
map_consumers.save(os.path.join('../input_plots', 'consumers_map.html'))





# # Create a marker for each consumer
# for idx, row in consumers_df.iterrows():
#     folium.Marker(
#         location=[row['lat'], row['lon']],
#         popup=f"Consumer: {row['demand_avg']}",
#     ).add_to(map_consumers)
#
# map_consumers.save(os.path.join('../input_plots', 'consumers_map.html'))  # Save the map as an HTML file





