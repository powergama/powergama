import powergama
import pathlib
import powergama.plots as pgplot
import folium
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import gurobipy
import pdfkit

year=2050


# datapaths
datapath= pathlib.Path(f"C:/Users/einar/OneDrive - NTNU/NTNU/9. semester/PowerGAMA/scripts/nordic/nordic/data_{year}")
file_storval_filling = pathlib.Path(f"C:/Users/einar/OneDrive - NTNU/NTNU/9. semester/PowerGAMA/scripts/nordic/nordic/data_storagevalues/profiles_storval_filling.csv")
file_30y_profiles = pathlib.Path(f"C:/Users/einar/OneDrive - NTNU/NTNU/9. semester/PowerGAMA/scripts/nordic/nordic/data_timeseries/timeseries_profiles/timeseries_profiles.csv")
powerplants = pd.read_csv(f"C:/Users/einar/OneDrive - NTNU/NTNU/9. semester/PowerGAMA/plots/data/All_powerplant_powerplantmatching.csv")
nordic_powerplants = powerplants[powerplants['Country'].isin(['Norway', 'Sweden', 'Finland'])]


data = powergama.GridData()
data.readGridData(nodes=datapath/"node.csv",
                    ac_branches=datapath/"branch.csv",
                    dc_branches=datapath/"dcbranch.csv",
                    generators=datapath/"generator.csv",
                    consumers=datapath/"consumer.csv")

node_file = datapath / "node.csv"
nodes_df = pd.read_csv(node_file)

mean_lat_nodes = nodes_df['lat'].mean()
mean_lon_nodes = nodes_df['lon'].mean()
map_nodes = folium.Map(location=[mean_lat_nodes, mean_lon_nodes], zoom_start=5)

# Create map centered in the Nordic region
mean_lat_prod = powerplants['lat'].mean()
mean_lon_prod = powerplants['lon'].mean()
map_prod = folium.Map(location=[mean_lat_prod, mean_lon_prod], zoom_start=5)




for idx, row in nodes_df.iterrows():
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=f"Node: {row['id']}",
    ).add_to(map_nodes)

map_nodes.save('nodes_map.html')  # Save the map as an HTML file


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

# Save the map
map_prod.save('powergrid_producers_map.html')  # Save the map as an HTML file

