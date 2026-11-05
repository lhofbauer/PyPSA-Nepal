"""
Script deriving the demand for PyPSA-Nepal

Copyright (C) 2026 PyPSA-Nepal contributors, licensed under a MIT license
"""

import os
import logging
import re
from functools import cache

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as raster_mask


pd.set_option('future.no_silent_downcasting', True)


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
    
    
    
    path_dem_shapefile = None  # Filepath or None
    path_subn_dem_data = None # None or path to file with respective demand for one or several years
    path_raster_gdp = "../data/raw_data/gdp/rast_gdpTot_1990_2024_5arcmin.tif"
    paths_raster_pop = {
                    year: (
                        "../data/raw_data/population/"
                        f"npl_pop_{year}_CN_100m_R2025A_v1.tif"
                        )
                    for year in range(2020, 2031)
                    }
    gdp_weight = 0.6
    subnational_name_column = "shapeName"
    
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
    
    hourly_dem = pd.concat(
        {year: hourly_dem * factor for year, factor in proj["SF"].items()},
        names=["year", "hour"])
    hourly_dem = hourly_dem.to_frame(name="Load")
    
    
    ### split demand across buses
    
    ### load bus regions and optional subnational demand
    bus_regions = gpd.read_file(path_processed + path_bus_regions)[["id", "geometry"]]
    bus_regions["id"] = bus_regions["id"].astype(str)
    
    if path_subn_dem_data is not None and path_dem_shapefile is not None:
        # load and process subnational shapefile
        # FIXME: ensure that province names are handled consistently across
        # files (here and elsewhere), add checks where useful
        subnational = gpd.read_file(path_dem_shapefile).to_crs(bus_regions.crs)
        subnational = subnational[[subnational_name_column, "geometry"]].rename(
            columns={subnational_name_column: "region"}
        )
        subnational["region"] = subnational["region"].replace({
            "Province 1": "Koshi",
            "Province 2": "Madhesh",
            "Sudurpaschim": "Sudurpashchim",
        })
        
        pieces = gpd.overlay(bus_regions, subnational, how="intersection")
        
        # load subnational demand
        regional_demand = pd.read_csv(path_subn_dem_data, index_col=0).T
        regional_demand.index = regional_demand.index.astype(int)
        regional_demand = (
            regional_demand
            .reindex(regional_demand.index.union(snapshot_years))
            .sort_index()
            .interpolate(method="index", limit_direction="both")
            .loc[snapshot_years]
        )
        regional_shares = regional_demand.div(
            regional_demand.sum(axis=1), axis=0
        )
    else:
        pieces = bus_regions.assign(region="Nepal")
        regional_shares = pd.DataFrame(
            1.0, index=snapshot_years, columns=["Nepal"]
        )
    
    ### sum raster values within each bus/demand region intersection
    
    # cache results so repeated raster years are only processed once
    @cache
    def raster_totals(path, band=1):
        totals = []

        with rasterio.open(path) as src:
            for geometry in pieces.to_crs(src.crs).geometry:
                values, _ = raster_mask(
                    src,
                    [geometry],
                    indexes=band,
                    crop=True,
                    all_touched=True,
                    filled=False,
                )
                totals.append(values.filled(0).sum())

        return np.asarray(totals)
    
    # get list of bands
    with rasterio.open(path_raster_gdp) as src:
        gdp_bands = {
            int(description.rsplit("_", 1)[-1]): band
            for band, description in enumerate(src.descriptions, start=1)
        }
    
    weights_by_year = {}
    
    for year in snapshot_years:
        gdp_year = min(gdp_bands, key=lambda y: abs(y - year))
        pop_year = min(paths_raster_pop, key=lambda y: abs(y - year))
    
        pieces["gdp"] = raster_totals(path_raster_gdp, gdp_bands[gdp_year])
        pieces["pop"] = raster_totals(paths_raster_pop[pop_year])
    
        # normalize within Nepal or within each subnational region.
        totals = pieces.groupby("region")[["gdp", "pop"]].transform("sum")
        fractions = pieces[["gdp", "pop"]].div(totals)
    
        pieces["weight"] = (
            gdp_weight * fractions["gdp"]
            + (1 - gdp_weight) * fractions["pop"]
        )
        # calculate overall weight taking into account bus region (piece)
        # and subnational region's fraction of demand
        pieces["weight"] *= pieces["region"].map(regional_shares.loc[year])
    
        # recombine pieces belonging to the same bus region
        weights_by_year[year] = (
            pieces.groupby("id")["weight"]
            .sum()
            .reindex(bus_regions["id"], fill_value=0.0)
        )
    
    bus_weights = pd.DataFrame(weights_by_year).T
    
    
    ### apply weights to timeseries
    
    # apply the corresponding year's spatial weights to each hourly total
    hourly_weights = bus_weights.reindex(
        hourly_dem.index.get_level_values("year")
    )
    hourly_weights.index = hourly_dem.index
    hourly_dem = (
        hourly_weights.mul(hourly_dem["Load"], axis=0)
        .add_suffix("_load")
        .reset_index(drop=True)
    )
    
    ### save files
    
    # save to file
    hourly_dem.to_csv(path_data+"loads-p_set.csv")
    
    # create file with load to bus mapping
    loads = pd.DataFrame({"name":hourly_dem.columns,
                          "bus": hourly_dem.columns.str[:-5],
                          "carrier":"AC"}).set_index("name")
    loads.to_csv(path_data+"loads.csv")
