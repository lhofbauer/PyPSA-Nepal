"""
Script deriving timeseries data for generators and storage inflow


Copyright (C) 2026 PyPSA-Nepal contributors, licensed under a MIT license
"""

import atlite
import pandas as pd
import geopandas as gpd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    
    #%% Levers and macros
    
    path_data = "../data/network/"
    path_cd = "../data/raw_data/climate/"
    path_hyd_coords = "../data/processed_data/all_hydro_coords.csv"
    path_processed = "../data/processed_data/"
    
    # FIXME: levers to be implemented, currently not working/tested
    # lever to adjust resolution
    buses = "provinces" # districts or provinces

    #%% Get cutouts and shapefiles
    
    # get cutout with reanalysis data via atlite
    # FIXME: consider temporal (year) alignment with demand curve
    # 1 Baisakh (01) 2073 = 13 April 2016 & 31 Chaitra (12) 2073 = 13 April 2017
    cutout = atlite.Cutout(
        path=path_cd+"nepal-2022_all.nc",
        module="era5",
        x=slice(80.0, 88.5),
        y=slice(26.0, 31),
        time=slice("2022-01-01","2022-12-31"),
        #time=slice("2021-01-01","2021-01-31"),
    )
    
    cutout.prepare(features=["wind","influx","temperature","height","runoff"],
                    monthly_requests=True,
                    concurrent_requests=True)
    
    
    
    # get shapefiles
    
    if buses == "provinces":
        areas = gpd.read_file("../data/raw_data/boundaries/geoBoundaries-NPL-ADM1-all/geoBoundaries-NPL-ADM1_simplified.shp")
        areas.loc[:,"shapeName"] = areas.loc[:,"shapeName"].replace({"Province 1":"Koshi",
                                                                     "Province 2":"Madhesh",
                                                                     "Sudurpaschim":"Sudurpashchim"})
        areas.loc[:,"shapeName"] = areas.loc[:,"shapeName"]+"_0" #Is this necessary?
    elif buses == "districts": 
        #Not used for now, but could be used in future for more detailed resolution
        areas = gpd.read_file("../data/raw_data/boundaries/geoBoundaries-NPL-ADM2-all/geoBoundaries-NPL-ADM2_simplified.shp")
    else:
        raise ValueError(f"'{buses}' not defined as value for resolution.")
        
    basins = gpd.read_file("../data/raw_data/hydrobasins/hybas_as_lev01-12_v1c/hybas_as_lev08_v1c.shp")
    
    
    
    # load generators and storage to get names of relevant pps
    pps = pd.read_csv(path_data+"generators.csv")
    stor = pd.read_csv(path_data+"storage_units.csv")
    gen_hyd = pps.loc[pps["carrier"]=="hydro","name"]
    stor_hyd = stor.loc[pps["carrier"]=="hydro","name"]               
    gen_pv = pps.loc[pps["carrier"]=="solar","name"]
    gen_wind = pps.loc[pps["carrier"]=="wind","name"]
    

    #%% Derive timeseries for hydro
    
    # load all hydro power plant coordinates (saved before thresholding)
    all_hydro_coords = pd.read_csv(path_processed+"all_hydro_coords.csv",
                                   index_col="name")
    
    logger.info(f"Loaded {len(all_hydro_coords)} hydro plant coordinates.")
    
    # filter out coordinates not within country (these should be handled
    # earlier, but still exist, so filtering here)
    
    # convert coordinates to GeoDataFrame
    coords_gdf = gpd.GeoDataFrame(
        all_hydro_coords,
        geometry=gpd.points_from_xy(all_hydro_coords.lon, all_hydro_coords.lat),
        crs='EPSG:4326'
    )
    
    within_provinces = gpd.sjoin(coords_gdf, areas,
                                 how='inner', predicate='dwithin',
                                 distance=2)
    plants_within = within_provinces.index.unique()
    
    # Filter out invalid coordinates
    final_coords = all_hydro_coords.loc[plants_within].copy()
    excluded_coords = all_hydro_coords.loc[~all_hydro_coords.index.isin(plants_within)]
    
    if len(excluded_coords) > 0:
        logger.warning(f"Excluded {len(excluded_coords)} plants that fall outside province boundaries.")
        for idx in excluded_coords.index[:5]:  # Show first 5
            row = excluded_coords.loc[idx]
            logger.warning(f"  {idx}: lat={row['lat']:.4f}, lon={row['lon']:.4f}")
        if len(excluded_coords) > 5:
            logger.warning(f"  ... and {len(excluded_coords)-5} more")
    
    logger.info(f"Final dataset: {len(final_coords)} plants within Nepal for atlite calculation")
    
    if len(final_coords) == 0:
        raise ValueError("No plants found within Nepal.")
    
    # calculate hydro inflows for plants using atlite
    logger.info("Calculating hydro patterns for valid hydro plants...")
    all_hyd = cutout.hydro(plants=final_coords[["lat","lon"]],
                           hydrobasins=basins,
                           show_progress=True)
    all_hyd = all_hyd.to_pandas().T # unit: m^3/h
    
    # calculate environmental flow as 10% of minimum monthly average flow
    # FIXME: this is likely incorrect, check
    eflow = all_hyd.groupby(by=[all_hyd.index.month]).mean().min()*0.10
    
    # calculate remaining flow
    all_hyd_use = all_hyd-eflow
    
    if all_hyd_use.min().min()<0:
        raise ValueError("Calculations led to negative hydro inflows.")
    
    # calculate capacity factors assuming Q45
    # FIXME: this is using a simple estimate based on design flows – improve
    # e.g., calibrate with actual flow/generation data
    q_all = all_hyd_use.quantile(q=0.45)
    all_hyd_cf = (all_hyd_use/q_all).clip(upper=1)
    
    # calculate energy inflows
    # FIXME: this is using a default head and simple calculation – to be improved
    # e.g., calibrate with actual flow/generation data
    head_default = 150.0 # unit: m 
    g = 9.81 # unit: m/s^2
    # 9.81 m/s2 * 1m * 1 m^3/s*1000kg/m^3 = X J/s
    rho = 1000
    all_hyd_inf = all_hyd_use/3600 *rho * g * head_default / 10**6 # unit: MW(h)
    
    #%% Process for individually represented and aggregate plants
    
    # now extract patterns for large individual plants (above threshold)
    
    # only extract patterns for large plants that were successfully processed
    ind_plants = [p for p in gen_hyd if "agg" not in p]
    ind_stors = [p for p in stor_hyd if "agg" not in p]
    # extract patterns for individually modelled plants
    hyd_cf = all_hyd_cf[ind_plants].copy()  
    hyd_inf = all_hyd_inf[ind_stors].copy() 
    
    logger.info(f"Extracted patterns for {len(hyd_cf.columns)} individual hydro plants without storage.")
    logger.info(f"Extracted patterns for {len(hyd_inf.columns)} individual hydro plants with storage.")
    
    # For missing (aggregate) hydro generators, use regional averages from ALL plants
    mi = ([a for a in gen_hyd if a not in hyd_cf.columns]
          +[a for a in stor_hyd if a not in hyd_inf.columns])
    
    logger.info(f"Processing {len(mi)} missing/aggregated hydro generators (with or without storage) using regional patterns...")
    
    for m in mi:
        # Extract region from generator name
        # For aggregated generators like "Bagmati_0 hydro True agg pp"
        if "agg" in m:  
            bus = m.split()[0]  # e.g., "Bagmati_0"
            region_base = bus.split("_")[0]  # e.g., "Bagmati"
        # For individual generators, find their bus region
        # FIXME: with the current setup, this should not be necessary,
        # as all ind. plants are expected to have their inflows/CFs
        else:  
            gen_bus = pps.loc[pps["name"] == m, "bus"].iloc[0] if len(pps.loc[pps["name"] == m]) > 0 else None
            if gen_bus:
                region_base = gen_bus.split("_")[0]  # e.g., "Bagmati"
            else:
                region_base = None
        
        if region_base:
            # find all valid hydro plants in the same region (from final_coords)
            regional_plants = final_coords[final_coords["bus"].str.startswith(region_base)]
            regional_plant_names = regional_plants.index.tolist()
            
            if regional_plant_names:
                if "storage" not in m:
                    # Use average of all hydro plants in same region
                    regional_pattern = all_hyd_cf[regional_plant_names].mean(axis=1)
                    hyd_cf.loc[:, m] = regional_pattern
                    logger.info(f"  {m}: Used regional average from {len(regional_plant_names)} plants in {region_base}")
                else:
                    # Use sum of all storage plants in the same region
                    # FIXME: this does double count, should only use the storage
                    # plants that were aggregate together for each aggregate 
                    # plant
                    regional_pattern = all_hyd_inf[[ s for s in regional_plant_names
                                                    if s not in ind_stors]].sum(axis=1)
                    hyd_inf.loc[:, m] = regional_pattern
                    logger.info(f"  {m}: Used regional average from {len(regional_plant_names)} plants in {region_base}")
            else:
                logger.warning(f"  No valid plants found in region {region_base} for {m}, using country-wide average")
                
                if "storage" not in m:
                    # fall back to country average
                    country_mean = all_hyd_cf.mean(axis=1)
                    hyd_cf.loc[:, m] = country_mean
                else:
                    # fall back to 1/7 of country sum (after deducting inflow
                    # for ind. plants)
                    country_pattern = (all_hyd_inf.sum(axis=1)-hyd_inf.sum(axis=1))/7
                    hyd_inf.loc[:, m] = country_pattern
                
        else:
            logger.warning(f"  No region found for {m}, using country average")
            if "storage" not in m:
                # fall back to country average
                country_mean = all_hyd_cf.mean(axis=1)
                hyd_cf.loc[:, m] = country_mean
            else:
                # fall back to 1/7 of country sum (after deducting inflow
                # for ind. plants)
                country_pattern = (all_hyd_inf.sum(axis=1)-hyd_inf.sum(axis=1))/7
                hyd_inf.loc[:, m] = country_pattern
    
    logger.info("Hydro capacity factor processing complete")

    #%% Derive wind and solar CF
    
    # FIXME: these seem quite low – to be checked
    # FIXME: this calculates CFs per area, could use pp lat and lon,
    # where applicable
    # calculate wind capacity factors
    wind_cf = cutout.wind(turbine="Vestas_V90_3MW",
                          shapes=areas,
                          index=pd.Index(areas["shapeName"],
                                         name="GEOGRAPHY"),
                          per_unit=True)
    wind_cf = wind_cf.to_pandas()
    
    # map to generators
    for g in gen_wind:
        gen_bus = pps.loc[pps["name"] == g, "bus"].iloc[0] if len(pps.loc[pps["name"] == g]) > 0 else None
        wind_cf.loc[:,g] = wind_cf.loc[:,gen_bus]
        
    wind_cf = wind_cf.loc[:,gen_wind]
    
    # calculate solar capacity factors
    pv_cf = cutout.pv(panel="CSi",
                      orientation="latitude_optimal",
                      shapes=areas,
                      index=pd.Index(areas["shapeName"],
                                      name="GEOGRAPHY"),
                      per_unit=True
                      )
    pv_cf = pv_cf.to_pandas()
    
    # map to generators
    for g in gen_pv:
        gen_bus = pps.loc[pps["name"] == g, "bus"].iloc[0] if len(pps.loc[pps["name"] == g]) > 0 else None
        pv_cf.loc[:,g] = pv_cf.loc[:,gen_bus]
        
    pv_cf = pv_cf.loc[:,gen_pv]

    #%% Aggregate and save all inflows and CFs
    # merge capacity factors
    cfs = pd.concat([wind_cf,
                    pv_cf,
                    hyd_cf],
                    axis=1)
    
    # get number of years
    ny = len(pd.read_csv(path_data+"investment_periods.csv"))
    
    
    # assume same capacity factors apply to all years
    cfs = pd.concat([cfs]*ny,ignore_index=True)
    
    # clip to avoid issues when solving
    for c in cfs.columns:
        cfs.loc[cfs[c]>0,c] = cfs.loc[cfs[c]>0,c].clip(0.001)
    
    # verify the structure before saving
    logger.info(f"CFs shape: {cfs.shape} (time_periods × generators)")
    logger.info(f"Expected generators: {len(pd.read_csv(path_data+'generators.csv'))}")
    logger.info(f"CFs generators: {len(cfs.T)}")
    
    # save to file
    cfs.to_csv(path_data+"generators-p_max_pu.csv")
    logger.info("Saved generators-p_max_pu.csv")
    
    # storage energy inflows
    
    # assume same inflow patterns apply to all years
    hyd_inf = pd.concat([hyd_inf]*ny, ignore_index=True)

    
    logger.info(f"Original inflow shape: {hyd_inf.shape} (time_periods × storage_units)")
  
    # save to file

    hyd_inf.to_csv(path_data + "storage_units-inflow.csv")
    logger.info("Saved storage_units-inflow.csv")
