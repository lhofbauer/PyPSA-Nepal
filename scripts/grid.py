"""
Script deriving the grid data for PyPSA-Nepal

Copyright (C) 2026 PyPSA-Nepal contributors, licensed under a MIT license
"""

import logging

import numpy as np
import pandas as pd
import geopandas as gpd

from geopy.distance import geodesic

pd.set_option('future.no_silent_downcasting', True)

logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



if __name__ == "__main__":

#%% levers and macros
    # FIXME: load from config file
    path_data = "../data/network/"
    path_shapefile = "../data/raw_data/boundaries/geoBoundaries-NPL-ADM1-all/geoBoundaries-NPL-ADM1_simplified.shp"
    path_tra_lines = "../data/raw_data/grid/transmission_lines_data.csv"
    
    
    # if to set all lines to extendable
    all_extendable=False
    # if to remove max from extendable lines
    remove_max=False
    
    # FIXME: include option to change spatial resolution/representation
    
#%% Load and process transmission grid data
  
    # load transmission grid data
    # FIXME: crosscheck and improve these data
    lines_df = pd.read_csv(path_tra_lines)
    
    # load your shapefile
    areas = gpd.read_file(path_shapefile)
    areas.loc[:,"shapeName"] = areas.loc[:,"shapeName"].replace({"Province 1":"Koshi",
                                                                 "Province 2":"Madhesh", 
                                                                 "Sudurpaschim":"Sudurpashchim"})
    areas = areas.to_crs("EPSG:4326")
    # calculate centroids
    areas['centroid'] = areas.geometry.centroid
    areas['lat'] = areas.centroid.y
    areas['lon'] = areas.centroid.x
    

    
    # create distance matrix
    provinces = areas['shapeName'].tolist()
    distance_matrix = {}
    
    for i, prov1 in enumerate(provinces):
        lat1, lon1 = areas.iloc[i]['lat'], areas.iloc[i]['lon']
        for j, prov2 in enumerate(provinces):
            lat2, lon2 = areas.iloc[j]['lat'], areas.iloc[j]['lon']
            if i != j:
                distance_km = geodesic((lat1, lon1), (lat2, lon2)).kilometers
                distance_matrix[f"{prov1}_0-{prov2}_0"] = round(distance_km)

    # apply distances
    lines_df.loc[:,"length"] = 1
    for idx, row in lines_df.iterrows():
        connection = f"{row['bus0']}-{row['bus1']}"
        if connection in distance_matrix:
            lines_df.at[idx, 'length'] = distance_matrix[connection]
        # also check reverse direction
        reverse_connection = f"{row['bus1']}-{row['bus0']}"
        if reverse_connection in distance_matrix:
            lines_df.at[idx, 'length'] = distance_matrix[reverse_connection]
    
    # set all to extendable if required
    if all_extendable:
        lines_df.loc[:,"p_nom_extendable"] = True
        
    # remove max if required
    if remove_max:
        lines_df.loc[lines_df["p_nom_extendable"]==True,"p_nom_max"] = np.nan
        
    # save to file
    # FIXME: this is currently implemented as links – likely to be reimplemented
    # as lines once issues that cause issues in the line implementation are fixed
    lines_df.to_csv(path_data + "links.csv", index=False)
    
    # create file with bus info
    buses = pd.DataFrame({"name":(areas["shapeName"]+"_0"),
                          "x":areas["lon"],
                          "y":areas["lat"],
                          "location":areas["shapeName"],
                          "v_nom":380}).set_index("name")
    buses.to_csv(path_data+"buses.csv")