"""
Script deriving the demand for PyPSA-Nepal

Copyright (C) 2026 PyPSA-Nepal contributors, licensed under a MIT license
"""

import os
import logging
import re

import pandas as pd
import numpy as np
import geopandas as gpd

pd.set_option('future.no_silent_downcasting', True)

import fitz  # import PyMuPDF
if not hasattr(fitz.Page, "find_tables"):
    raise RuntimeError("This PyMuPDF version does not support the table feature")

import nepali_datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



if __name__ == "__main__":

#%% levers and macros
    # FIXME: read from config file
    
    path_data = "../data/network/"
    path_dem = "../data/raw_data/demand/"
    path_proj = "../data/raw_data/others/elec_demand_proj.csv" 
    path_pop = "../data/raw_data/others/population.csv"
    path_pop_proj = "../data/raw_data/others/Population Projections for Nepal.pdf"
    path_processed = "../data/processed_data/"
    
    


    path_bus_regions = "bus_regions.shp"
    snapshot_years = [2020, 2030, 2040, 2050] 

#%% Load and process demand data

    # get list of files
    files = os.listdir(path_dem)
    
    # list of to store dataframe with monthly data
    year_data = []

    # load monthly demand files
    for f in files:
        # read csv file with (half)hourly demand data for one month
        dem = pd.read_csv(path_dem+f,index_col=0)
    
        # rearrange from wide to long format and rename columns
        dem_stacked = dem.stack(future_stack=True)
        dem_stacked = dem_stacked.reset_index()
        dem_stacked.columns = ['Day', 'Time', 'Load']
    
        # strip whitespace and replace empty strings with NaN
        dem_stacked['Load'] = dem_stacked['Load'].astype(str).str.strip()
        dem_stacked['Load'] = dem_stacked['Load'].replace('', np.nan)
        
        # convert to numeric:
        dem_stacked['Load'] = pd.to_numeric(dem_stacked['Load'],
                                            errors='coerce')
    
        # replace zero values with NaN first, then interpolate
        dem_stacked["Load"] = dem_stacked["Load"].replace(0, np.nan)
    
        # linear interpolation
        dem_stacked["Load"] = dem_stacked["Load"].interpolate(method='linear')
    
        # extract date information from filename:
        # split the string by hyphen, underscore or dot
        date_info = re.split(r"[-_.]+", f)
    
        nepali_datetime_year = date_info[0]
        nepali_datetime_month = date_info[2]
    
        # create new column with nepali datetime object
        # FIXME: this currently assumes data are from Nepali year 2074
        # instead of 2073, as the number of days per month seems to not match
        # for 2073 based on nepali_datetime – to be explored
        dem_stacked['NDate'] = ("2074"+"-" #nepali_datetime_year
                                +nepali_datetime_month+"-"
                                +dem_stacked['Day'].astype(str).str.zfill(2))
        
        
        # aggregate to hourly
        # extract hour from time column, with half hourly data averaged into
        # the value for the full hour
        dem_stacked['Time'] = dem_stacked['Time'].str[:-3].astype(int)
        # groupby day and time
        dem_stacked_h = dem_stacked.groupby(['Day','Time']).agg({
            'Load': 'mean',
            'NDate': 'first' 
        }).reset_index()

        # add the processed data to yearly data
        year_data.append(dem_stacked_h)

    
    # concatenate all monthly data into a yearly dataset
    year_df = pd.concat(year_data, ignore_index=True)
    
    # convert to gregorian calendar
    dates = year_df["NDate"].apply(nepali_datetime.datetime.strptime,args=('%Y-%m-%d',)).to_list()
    dates = [str(d.to_datetime_date().month).zfill(2)+"-"+str(d.to_datetime_date().day).zfill(2)
             for d in dates]
    year_df["GDate"] = dates

    
    # sort
    hourly_dem_df_sorted = year_df.sort_values(["GDate","Time"]).reset_index(drop=True)
    
    # replace outliers that are assumed faulty data points
    q1 = hourly_dem_df_sorted["Load"].quantile(0.05)
    q3 = hourly_dem_df_sorted["Load"].quantile(0.95)
    iqr = q3 - q1
    
    hourly_dem_df_sorted.loc[(hourly_dem_df_sorted["Load"] < (q1*0.75)),
                             "Load"] = q1
    hourly_dem_df_sorted.loc[(hourly_dem_df_sorted["Load"] > (q3*1.25)),
                             "Load"] = q3
        
    # make yearly demand into a pandas series to feed into growth factor section of code:
    hourly_dem = pd.Series(hourly_dem_df_sorted["Load"], dtype='float64')
    
    ### project demand
    # total demand of demand profile in GWh (converting from MWh)
    baseline_annual_demand = hourly_dem.sum() / 1000

    
    # read in demand projections csv
    projections = pd.read_csv(path_proj)
    
    # extract data of interest from demand projection csv:
    # scenario can be chosen here - options are: BAU, High, Low, Electrification
    scen = "Low"
    years = projections["Year"]
    # FIXME: choose scenario based on config
    demands = projections[scen]
    proj = projections.loc[:,["Year",scen]]
    proj = proj.set_index("Year")


    # add snapshot years by interpolation if not part of years given in
    # projection
    mi = [y for y in snapshot_years if y not in proj.index]
    if len(mi)>0:
        for y in mi:
            # add rows for missing years to dataframe
            proj.loc[y] = np.nan
            
        proj = proj.sort_index()
        # interpolate to get data for missing years
        proj.loc[:,scen] = proj.loc[:,scen].interpolate(method="index",
                                                          limit_direction="both")
        
    # filter
    proj = proj.loc[snapshot_years]
    
    # calculating growth factors:
    proj.loc[:,"SF"] = proj.loc[:,scen] / baseline_annual_demand
    
    hourly_dem = pd.concat([hourly_dem*f for f in proj["SF"]],
                    ignore_index=True)
    hourly_dem = hourly_dem.to_frame()
    
    
    ### split demand across buses
    
    # FIXME: Temporarily removed to add flexible but basic estimate based on
    # area (to handle flexible spatial resolutions). This should use geospatial
    # demand data to disaggregate national or regional totals.
    # # load population data
    # doc = fitz.open(path_pop_proj)
    
    # # lists to hold data extracted from population pdf
    # names = []
    # population = []
    
    # # FIXME: take into account projections, not only single year,
    # # iterating through pages of interest to extract population info
    # for i in range(203,210):
    #     page = doc[i]
    #     page.set_rotation(90) #table is sideways in pdf
    #     try:
    #         tabs = page.find_tables()
    #         tab = tabs[0]
    #         intermediary_df = tab.to_pandas()
    #         name = intermediary_df["Province\nand age\ngroups"][1]
    #         names.append(name+"_0") 
    #         pop = intermediary_df["2041"][2]
    #         pop = float(pop.replace(',', ''))
    #         population.append(pop)
    #     except:
    #         logger.error(i)
    
    # # zip lists into population dataframe
    # pop_df = pd.DataFrame(zip(names,population), columns= ['Bus','Population'])
    # pop_df = pop_df.set_index("Bus")
    # # convert to proportions (percentages of total population)
    # pop_df = pop_df/pop_df.sum()
    
    # simply take area to split demand (see fixme above)
    areas = gpd.read_file(path_processed+path_bus_regions)
    areas.loc[:,"area"] = areas.area
    
    pop_df = areas.loc[:,["id","area"]]
    pop_df.columns = ["id","Population"]
    pop_df = pop_df.set_index("id")
    pop_df = pop_df/pop_df.sum()
    
    ### calculate demand for each province as fraction of total
    for p in pop_df.index:
        hourly_dem.loc[:,p+"_load"] = hourly_dem.loc[:,"Load"]*pop_df.loc[p,"Population"]
    hourly_dem = hourly_dem.drop("Load",axis=1)
    
    # save to file
    hourly_dem.to_csv(path_data+"loads-p_set.csv")
    
    # create file with load to bus mapping
    loads = pd.DataFrame({"name":hourly_dem.columns,
                          "bus": hourly_dem.columns.str[:-5],
                          "carrier":"AC"}).set_index("name")
    loads.to_csv(path_data+"loads.csv")
