"""
Script deriving static generator parameters for PyPSA-Nepal


Copyright (C) 2026 PyPSA-Nepal contributors, licensed under a MIT license
"""


import os
from functools import partial

import pandas as pd
import geopandas as gpd
import logging
import numpy as np
import rasterio
from atlite.gis import ExclusionContainer, shape_availability
import nepali_datetime
      
pd.set_option('future.no_silent_downcasting', True)



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



# categorize hydropower plants based on their names
# FIXME: this could be replaced with specific mapping, incorporating also
# technical details for HPPs where relevant
def categorize_plant(name):
    name = name.strip()

    # Rule 1: Exact match for Kulekhani-I
    if name == 'Kulekhani-I':
        return 'storage'
    
    # Rule 2: Pumped storage
    if 'Pumped Storage' in name or 'Pump Storage' in name:
        return 'pumped_storage'
    
    # Rule 3: Other storage (not pumped)
    if 'Storage' in name:
        return 'storage'
    
    # Rule 4: Exact match list for PROR
    pror_exact_names = [
        'Kali Gandaki A', 'Madhya Marsyangdi', 'Marsyangdi', 
        'Trishuli', 'Chameliya Khola', 'PhewaKhola'
    ]
    if name in pror_exact_names:
        return 'pror'

    # Rule 5: PROR-related keywords
    pror_partial_keywords = ['PROR', 'PRoR', 'Pump']
    if any(keyword in name for keyword in pror_partial_keywords):
        return 'pror'

    # Rule 6: Default to ROR
    return 'ror'
    
    
# process build year - handle both YYYY-MM-DD and DD/MM/YYYY formats
def parse_build_year(x):
    if pd.isna(x):
        return 2030  # default fallback
        logger.warning(f"Build year included entry '{x}', replacing with 2030")
    x = str(x)
    if "/" in x:
        # DD/MM/YYYY format
        day, month, year = [int(i) for i in x.split("/")]
        
    else:
        # YYYY-MM-DD format
        year, month, day = [int(i) for i in x.split("-")]
    return nepali_datetime.date(year, month, day).to_datetime_date().year

if __name__ == "__main__":

    #%% Levers and macros ###

    # FIXME: refine and move to config
    path_data = "../data/network/"
    path_pp = "../data/raw_data/power_plants/" 
    path_processed = "../data/processed_data/"
    
    path_bus_regions = "bus_regions.shp"
    
    # lever for power plant aggregation
    thres = 400 # capacity threshold (MW)
    
    # levers for generic solar PV and wind (potential not covered by licenses)
    include_generic_vres = True
    path_solar_potential = "../data/raw_data/others/solar_potential.csv"
    path_wind_potential = "../data/raw_data/others/wind_potential.csv"

    path_landcover = "../data/raw_data/landcover/copernicus_landcover.tif"
    path_protected_areas = None  # optional vector file, ignored if None
    path_slope = None  # optional raster in degrees, required if max_slope is set
    potential_area_crs = "+proj=laea +lat_0=28 +lon_0=84 +datum=WGS84 +units=m +no_defs"
    potential_resolution = 100  # metres
    # power density needs to be provided in MW per eligible km2.
    # FIXME: placeholder config, to be replaced
    potential_grid_settings = {
        "solar": {"capacity_per_sqkm": 4.6,
                  "land_cover_codes": [20, 30, 40, 50, 60, 90, 100],
                  "settlement_buffer": 0,
                  "max_slope": None},
        "wind": {"capacity_per_sqkm": 3.0,
                 "land_cover_codes": [20, 30, 40, 60, 100,
                                      111, 112, 113, 114, 115, 116,
                                      121, 122, 123, 124, 125, 126],
                 "settlement_buffer": 1000,
                 "max_slope": None},
    }
    scale_potentials_to_total = True
    # gross technical totals, including licensed projects: a number specifies MW;
    # "Laupane-0f" / "Laupane-0.75f" takes totals from Laupane et al.
    potential_national_totals = {"solar": "Laupane-0f", "wind": "Laupane-0f"}

    #%% Load and process power plant lists
    
    # load raw power plant lists
    
    files = os.listdir(path_pp)
    csv_files = [f for f in files if f.endswith('.csv')]
    
    logger.info(f"Found {len(csv_files)} CSV files to process.")
    
    l_pps = list()
    for f in csv_files:
        try:
            # Load CSV file
            pps = pd.read_csv(path_pp + f, encoding='MacRoman')
            
            # Extract carrier name from filename
            carrier_name = f.split("_")[0]
            pps.loc[:,"carrier"] = carrier_name
            
            # Process build year assumptions
            # FIXME: this currently makes simple assumptions on built year
            # based on the type of license, to be improved
            if "_construction_" in f:
                pps.loc[:,"C O D"] = "2086-06-01" #2029
            elif "_construction-applied_" in f:
                pps.loc[:,"C O D"] = "2086-06-01" #2029
            elif "_survey_" in f:
                pps.loc[:,"C O D"] = "2096-06-01" #2039
            elif "_survey-applied_" in f:
                pps.loc[:,"C O D"] = "2096-06-01" #2039
            elif "_survey-technical_" in f:
                pps.loc[:,"C O D"] = "2096-06-01" #2039
                
            l_pps.append(pps)
            logger.info(f"✓ Processed {f}: {pps.shape}")
            
        except Exception as e:
            logger.error(f"✗ Error processing {f}: {e}")
    
    if l_pps:
        pps = pd.concat(l_pps, ignore_index=True)
        logger.info(f"Final concatenated shape: {pps.shape}")
    else:
        logger.error("No DataFrames to concatenate!")
    
    # renames and fix data issues
    rename = {"Project":"name",
                "Capacity (MW)":"p_nom",
                #"River":"River",
                #"Isuue Date":"Date_start",
                #"Validity":"Date_end",
                #"Address":"Address",
                "Latitiude N 1":"Lat_0",
                "Latitiude N 2":"Lat_1",
                "Longitude E 1":"Lon_0",
                "Longitude E 2":"Lon_1",
                #"VDC/District":"District",
                "C O D":"build_year",
                "carrier":"carrier"}
    pps = pps.loc[:,[c for c in rename.keys()]]
    pps = pps.rename(columns=rename)
    
    pps.loc[:,"build_year"] = pps.loc[:,"build_year"] .replace({"0000-00-00":"2080-01-01",
                                                                "\xa0":"2080-01-01",
                                                                "2081-09":"2081-09-01"})
    
    pps.loc[pps["name"]=="Ghunsa Khola HEP","name"] = pps.loc[pps["name"]=="Ghunsa Khola HEP","name"]+ " "+ pps.loc[pps["name"]=="Ghunsa Khola HEP","p_nom"].astype(str)

    
    #%% Clean and process data
    
    # apply categorization (for hydro)
    pps.loc[:,'plant_type'] =  pps.loc[:,'carrier']
    pps.loc[pps['carrier'] == 'hydro','plant_type'] = pps.loc[pps['carrier'] == 'hydro','name'].apply(categorize_plant)
    
    unique_plant_types = pps['plant_type'].unique()
    logger.info("Power plant types existing:"+str(unique_plant_types))
    
    # Clean the coordinate columns first to remove problematic Unicode characters
    for coord in ["Lat_0","Lat_1","Lon_0","Lon_1"]:
        pps.loc[:,coord] = (pps[coord].str.replace('¬†', ' ')
                                        .str.replace('\xa0', ' ')
                                        .str.replace('¬¨‚Ä†', ' ')
                                        .str.replace('¬', ' ')
                                        .str.replace('‚Ä†', ' '))
    
    
    # process coordinates --> take the central point
    # FIXME: consider if this is the best approach, and update if required
    for coord in ["Lat_0","Lat_1","Lon_0","Lon_1"]:
         direction = "N"
         c = pps.loc[:,coord].str.split('[o°\'"]',expand=True)
         c.loc[:,3] = "1"
         c.loc[:,"c"] = (c[0].astype(float) + c[1].astype(float) /60 + c[2].astype(float) /(60*60))* c[3].astype(float)
         pps.loc[:,coord+"_f"] = c.loc[:,"c"]
    pps.loc[:,"lat"] = pps.loc[:,["Lat_0_f","Lat_1_f"]].mean(axis=1)
    pps.loc[:,"lon"] = pps.loc[:,["Lon_0_f","Lon_1_f"]].mean(axis=1)
    
    # FIXME: add cleaning of coordinates, check if within a province, etc.
    
    # convert build year/date to year in Gregorian calendar
    pps.loc[:,"build_year"] = pps.loc[:,"build_year"].apply(parse_build_year).astype(str)
    pps["build_year"] = pps.loc[:,"build_year"].astype(int)
    
    # locate area in model region
    pps_gdf = gpd.GeoDataFrame(pps,geometry=gpd.points_from_xy(pps.lon,pps.lat),
                               crs="EPSG:4326")
    areas = gpd.read_file(path_processed+path_bus_regions)
    
    areas = areas.to_crs("EPSG:4326")
    # spatial join nearest --> each power plant point, it finds the nearest
    # administrative boundary polygon and joins the boundary's attributes
    # to the power plant record
    pps_gdf = pps_gdf.sjoin_nearest(areas, how="left") 
    pps_gdf.loc[:,"bus"] = pps_gdf.loc[:,"id"] # defines bus
    # extract only the required columns 
    pps = pps_gdf.loc[:,["name","p_nom","build_year","lat","lon","bus","carrier","plant_type"]].copy()
    # ensures build_year is an integer for proper calculation
    pps.loc[:,"build_year"] = pps.loc[:,"build_year"].astype(int)

    #%% Adding other parameters (cost, efficiency, etc.)
    
    # Read technology costs from CSV file
    # these are overnight costs, hence the need for the annualization
    tech_costs_per_kw = pd.read_csv("../data/raw_data/others/technology_params.csv",
                                    index_col=0)
    # Nepal-specific discount rate:
    discount_rate = 0.10  # 10% from Nepal guidelines
    
    # base year for calculation, either year of optimization or, if 1+, 2020 (?)
    # FIXME: check if used correctly
    year_of_interest = 2020 
    
    # Set default values first
    pps.loc[:,"lifetime"] = 0
    pps.loc[:,"capital_cost"] = 0
    pps.loc[:,"marginal_cost"] = 0
    pps.loc[:,"efficiency"] = 1.0 #100%
    
    # function to calculate annualized cost (updated for PyPSA units)
    def annualize_cost(overnight_cost_per_kw, discount_rate, lifetime):
        # converts overnight cost ($/kW) to annualized cost ($/MW) for PyPSA
        annualized_per_kw = (overnight_cost_per_kw * discount_rate 
                             / (1 - (1 + discount_rate)**(-lifetime)))
        # convert kW to MW for PyPSA
        return int(annualized_per_kw * 1000) 
    
    # check for duplicates in hydro plants before saving coordinates
    hydro_plants = pps.loc[pps["carrier"]=="hydro"]
    hydro_dupes = hydro_plants[hydro_plants["name"].duplicated(keep=False)]
    
    if not hydro_dupes.empty:
        logger.warning("Duplicate hydro plant names found:")
        logger.warning(hydro_dupes[["name", "lat", "lon", "bus", "p_nom"]].to_string())
        logger.warning(f"\nFound {len(hydro_dupes)} duplicate entries for {len(hydro_dupes['name'].unique())} plant names")
        
        # Keep entry with largest p_nom for each duplicate name
        hydro_plants = hydro_plants.loc[hydro_plants.groupby('name')['p_nom'].idxmax()]
        logger.warning(f"Kept largest capacity plants, resulting in {len(hydro_plants)} unique plants")
    else:
        logger.info("✓ No duplicate hydro plant names found")
        
    #%% Save coordinates of certain types of power plants for later processing
    
    # all hydro pps – this will be used for regional averaging
    all_hydro_coords = hydro_plants[["name","lat","lon","bus"]]
    all_hydro_coords.to_csv(path_processed+"all_hydro_coords.csv", index=False)
    
    # separately save the coordinates for generators and PROR storage that are
    # above threshold - the PROR plants will become storage units
    pror_plant_coords = pps.loc[((pps["p_nom"]>=thres)&
                                      (pps["carrier"]=="hydro")&
                                      (pps["plant_type"]=="PROR")), 
                                     ["name","lat","lon"]]
    pror_plant_coords.to_csv(path_processed+"pror_coords.csv", index=False)
    
    # save coordinates for large hydro generators (above threshold,
    # excluding PROR)
    large_hydro_coords = pps.loc[((pps["p_nom"]>=thres)&
                               (pps["carrier"]=="hydro")&
                               (pps["plant_type"]!="PROR")),
                              ["name","lat","lon"]]
    large_hydro_coords.to_csv(path_processed+"hydro_coords.csv", index=False)
    
    # save coordinates for all wind plants (before any filtering or 
    # aggregation) - used to calculate local capacity factors for  wind projects
    # FIXME: this is currently not actually used by model as the CFs returned
    # by atlite are very small for the wind projects defined
    all_wind_coords = pps.loc[pps["carrier"]=="wind", ["name","lat","lon","bus"]]
    all_wind_coords.to_csv(path_processed+"all_wind_coords.csv", index=False)
    
    # drop "lat","lon" from pps
    pps = pps.drop(columns=["lat","lon"])
    
    #%% separate out power plants with any kind of storage capability
    
    # hydro plants with any kind of storage
    sto_mask = pps['plant_type'].isin(["pror",
                                        "storage",
                                        "pumped_storage"])
    
    # create storage dataframe from PROR plants – implemented as storage
    stor = pps[sto_mask].copy()
    
    # remove them from the main pps dataframe:
    pps = pps[~sto_mask]
    
    
    #%% Process storage plants
    
    # Update costs for storage types
    # FIXME: need to account for cost differences, at least based on type,
    # best for each power plant, where data are available
    pror_mask = (stor['plant_type'] == 'pror')
    stor.loc[pror_mask, "capital_cost"] = annualize_cost(tech_costs_per_kw.loc['Hydro PROR',
                                                                       str(year_of_interest)],
                                                 discount_rate,
                                                 tech_costs_per_kw.loc['Hydro PROR',
                                                                       'Lifetime (years)'])
    stor.loc[pror_mask, "marginal_cost"] = tech_costs_per_kw.loc["Hydro PROR",
                                                         "Marginal Cost ($/MWh)"]  
    stor.loc[pror_mask, "lifetime"] = tech_costs_per_kw.loc["Hydro PROR",
                                                    "Lifetime (years)"]
    # hydro storage plants, including pumped storage
    storage_mask = stor['plant_type'].isin(["storage","pumped_storage"])
    stor.loc[storage_mask, "lifetime"] = tech_costs_per_kw.loc['Hydro Storage',
                                                              'Lifetime (years)']
    stor.loc[storage_mask, "capital_cost"] = annualize_cost(tech_costs_per_kw.loc['Hydro Storage',
                                                                                 str(year_of_interest)],
                                                           discount_rate,
                                                           tech_costs_per_kw.loc['Hydro Storage','Lifetime (years)'])
    stor.loc[storage_mask, "marginal_cost"] = tech_costs_per_kw.loc["Hydro Storage", "Marginal Cost ($/MWh)"]
    stor.loc[storage_mask, "efficiency"] = tech_costs_per_kw.loc["Hydro Storage", "Efficiency"]

    
    # add PyPSA Storage Unit parameters
    stor.loc[:, "p_nom_extendable"] = False
    stor.loc[:, "p_nom_min"] = pd.NA
    stor.loc[:, "p_nom_max"] = pd.NA
    
    # for future storage projects, set as potential capacity
    stor.loc[stor["build_year"] > 2026, "p_nom_extendable"] = True  
    stor.loc[stor["build_year"] > 2026,
             "p_nom_max"] = stor.loc[stor["build_year"] > 2026, "p_nom"]
    stor.loc[stor["build_year"] > 2026, "p_nom"] = 0  
    
    # storage-specific parameters 
    # FIXME: replace with plant specific data, at least for larger storage and
    # pumped storage plants
    # for PROR plants, assume daily/weekly storage
    pror_mask = stor['plant_type'] == 'pror'
    stor.loc[pror_mask, 'max_hours'] = 5
    if pror_mask.any():
        logger.info(f"Set {pror_mask.sum()} PROR plants to 5 hours")

    # seasonal storage plants, including pumped (e.g., Kulekhani-I)
    storage_mask = stor['plant_type'].isin(["storage","pumped_storage"])
    stor.loc[storage_mask, 'max_hours'] = 2160  # 3 months = 90 days * 24 hours
    # stor.loc[storage_mask, 'max_hours'] = 2880  # use this for 4 months instead
    if storage_mask.any():
        logger.info(f"Set {storage_mask.sum()} (pumped or not) storage plants to 2160 hours (3 months)")

    # all other plant types (should be none)– assumed non-storage (safe fallback) 
    other_mask = ~(pror_mask | storage_mask)
    if other_mask.any():
        other_types = stor.loc[other_mask, 'plant_type'].unique()
        stor.loc[other_mask, 'max_hours'] = 0  
        logger.warning(f"Set {other_mask.sum()} non-storage plants ({other_types}) to 0 hours")

    
    # efficiency when pumping/storing water
    stor.loc[:, "efficiency_store"] = 1
    # Same as generation efficiency used: ~86.57% = Average of all the
    # hydropower plants for "Actual Generation/Design Generation ratio (%)"
    # from the NEA 2024 Gen. Report of Hydropower plant (0.87)
    # FIXME: set to 1 for now, need to consider if 0.87 or 1 or else is approp.
    stor.loc[:, "efficiency_dispatch"] = 1
    
    # Initial state and cycling parameters
    # initial state is ignored if cyclic state is True
    #stor.loc[:, "state_of_charge_initial"] = 0.5  # Start at 50% capacity
    stor.loc[:, "cyclic_state_of_charge"] = True  
    stor.loc[:, "cyclic_state_of_charge_per_period"] = True
    
    # Power limits (per unit of p_nom)
    # pumped storage can store at full capacity (negative = charging/pumping)
    # all others cannot store by pumping (only due to inflow defined elsewhere)
    stor.loc[:, "p_min_pu"] = 0
    stor.loc[stor['plant_type'] == 'pumped_storage', "p_min_pu"] = -1.0  
    # can discharge at full capacity (positive = generation)
    stor.loc[:, "p_max_pu"] = 1.0   

    # clean up: remove plant_type column as it's not needed for PyPSA Storage Units
    stor = stor.drop(columns=['plant_type'])
    
    #%% Set parameters for power plants without storage
    # hydro ROR plants:
    ror_mask = pps['plant_type'] == 'ror'
    pps.loc[ror_mask, "lifetime"] = tech_costs_per_kw.loc["Hydro ROR", "Lifetime (years)"]  
    pps.loc[ror_mask, "capital_cost"] = annualize_cost(tech_costs_per_kw.loc['Hydro ROR', str(year_of_interest)], discount_rate, tech_costs_per_kw.loc['Hydro ROR','Lifetime (years)'])
    pps.loc[ror_mask, "marginal_cost"] = tech_costs_per_kw.loc["Hydro ROR", "Marginal Cost ($/MWh)"]
    # ~86.57% = Average of all the hydropower plants for
    # "Actual Generation/Design Generation ratio (%)" from
    # the NEA 2024 Gen. Report of Hydropower plant (0.87)
    # FIXME: set to 1 for now, need to consider if 0.87 or 1 or else is approp.
    pps.loc[ror_mask, "efficiency"] = 1
    

    # solar plants
    solar_mask = pps['plant_type'] == 'solar'
    pps.loc[solar_mask, "capital_cost"] = annualize_cost(tech_costs_per_kw.loc["Solar PV", str(year_of_interest)], discount_rate, tech_costs_per_kw.loc["Solar PV","Lifetime (years)"])
    pps.loc[solar_mask, "marginal_cost"] = tech_costs_per_kw.loc["Solar PV", "Marginal Cost ($/MWh)"]   
    pps.loc[solar_mask, "efficiency"] = tech_costs_per_kw.loc["Solar PV", "Efficiency"]
    pps.loc[solar_mask, "lifetime"] = tech_costs_per_kw.loc["Solar PV", "Lifetime (years)"]
    
    # wind plants
    wind_mask = pps['plant_type'] == 'wind'
    pps.loc[wind_mask, "capital_cost"] = annualize_cost(tech_costs_per_kw.loc["Wind", str(year_of_interest)], discount_rate, tech_costs_per_kw.loc["Wind","Lifetime (years)"])
    pps.loc[wind_mask, "marginal_cost"] = tech_costs_per_kw.loc["Wind", "Marginal Cost ($/MWh)"]
    pps.loc[wind_mask, "efficiency"] = tech_costs_per_kw.loc["Wind", "Efficiency"] 
    pps.loc[wind_mask, "lifetime"] = tech_costs_per_kw.loc["Wind", "Lifetime (years)"]
    
    # cogeneration plants - NOTE: The Thermal plants not in operation acc. to NEA & the cogeneration plants are small-scale bagasse-fired cogeneration plants typical of sugar mills!
    cogen_mask = pps['plant_type'] == 'cogeneration'
    pps.loc[cogen_mask, "capital_cost"] = annualize_cost(tech_costs_per_kw.loc["Cogeneration", str(year_of_interest)], discount_rate, tech_costs_per_kw.loc["Cogeneration","Lifetime (years)"])  # input of function = 2500 $/kW (range: 2000-3000)
    pps.loc[cogen_mask, "marginal_cost"] = tech_costs_per_kw.loc["Cogeneration", "Marginal Cost ($/MWh)"]  # $/kW-year (estimated O&M)
    pps.loc[cogen_mask, "efficiency"] = tech_costs_per_kw.loc["Cogeneration", "Efficiency"]  # 25% efficiency - source: https://onlinelibrary.wiley.com/doi/10.1155/2023/5749122 - Table 1
    pps.loc[cogen_mask, "lifetime"] = tech_costs_per_kw.loc["Cogeneration", "Lifetime (years)"]
    
    # remove plant_type column as it's not needed for PyPSA generators
    pps = pps.drop(columns=["plant_type"])  
    
    
    ### add additional placeholder data 
    pps.loc[:,"p_nom_extendable"] = False
    pps.loc[:,"p_nom_min"] = pd.NA
    pps.loc[:,"p_nom_max"] = pd.NA
    
    # if build year in future, set as potential capacity, not existing capacity
    pps.loc[pps["build_year"]>2026,"p_nom_extendable"] = True 
    # this sets the the maximum buildable capacity to the planned capacity
    pps.loc[pps["build_year"]>2026,"p_nom_max"] = pps.loc[pps["build_year"]>2026,
                                                          "p_nom"]
    # current installed capacity is zero (not built yet)
    pps.loc[pps["build_year"]>2026,"p_nom"] = 0 
    
    
    #%% Aggregate powerplants below the threshold
    
    # identifies plants below the defined threshold for aggregation.
    # Uses p_nom + p_nom_max to include both existing and potential capacity.
    pps_agg = pps.loc[(pps["p_nom"]+pps["p_nom_max"].fillna(0).astype(float))<thres] 
    
    # the function below combines small plants by carrier type 
    # (hydro, solar, etc.), geographic location (bus),
    # Expandability (existing vs. future plants)
    pps_agg = pps_agg.groupby(["carrier",
                               "bus",
                               "p_nom_extendable"]).agg({"name":"sum",
                                                    "p_nom":"sum", #total capacity is summed up
                                                    "p_nom_min":"sum",
                                                    "p_nom_max":"sum",
                                                    "efficiency":"mean", 
                                                    "marginal_cost":"mean",
                                                    "build_year":"mean", 
                                                    "lifetime":"mean", 
                                                    "capital_cost":"mean"}) 
                                                         
    pps_agg.loc[:,"build_year"] = pps_agg.loc[:,"build_year"].astype(int)
    pps_agg = pps_agg.reset_index()
    # create descriptive name for aggregated plants
    pps_agg.loc[:,"name"] = (pps_agg.loc[:,"bus"]
                             +" "
                             + pps_agg.loc[:,"carrier"]
                             +" "
                             + pps_agg.loc[:,"p_nom_extendable"].astype(str)
                             + " agg pp")
    # merges the large individual plants (≥60 MW) - modeled separately for
    # precision & the aggregated small plants - modeled as representative units
    pps = pd.concat([pps.loc[(pps["p_nom"]+pps["p_nom_max"].fillna(0).astype(float))>=thres],
                     pps_agg]) 
    
    
    ### check for duplicates in generator names
    pps_dupes = pps[pps['name'].duplicated(keep=False)]
    if not pps_dupes.empty:
        logger.warning("Duplicate generator names:")
        logger.warning(pps_dupes[['name', 'carrier', 'bus', 'p_nom', 'p_nom_extendable']])
        logger.warning(f"\nDuplicate names: {pps_dupes['name'].unique().tolist()}")
        # FIXME: add error?
        
    #%% Aggregate storage power plants below the threshold
    
    # Aggregate small storage units (same logic as generators)
    stor_agg = stor.loc[(stor["p_nom"]+stor["p_nom_max"].fillna(0).astype(float))<thres] # identifies plants below the defined threshold for aggregation. Uses p_nom + p_nom_max to include both existing and potential capacity.
    
    # Group by carrier, bus, and expandability
    stor_agg = stor_agg.groupby(["carrier",
                                "bus", 
                                "p_nom_extendable"]).agg({
        "name": "sum",
        "p_nom": "sum",
        "p_nom_min": "sum", 
        "p_nom_max": "sum",
        "marginal_cost": "mean",
        "build_year": "mean",
        "lifetime": "mean",
        "capital_cost": "mean",
        # Storage-specific parameters:
        "max_hours": "mean",
        "efficiency_store": "mean",
        "efficiency_dispatch": "mean",
        #"state_of_charge_initial": "mean",
        "cyclic_state_of_charge": "mean",
        "cyclic_state_of_charge_per_period": "mean",
        "p_min_pu": "mean",
        "p_max_pu": "mean",
        "efficiency": "mean"
    })
    
    stor_agg.loc[:, "build_year"] = stor_agg.loc[:, "build_year"].astype(int)
    stor_agg = stor_agg.reset_index()
    
    # create descriptive names for aggregated storage
    stor_agg.loc[:, "name"] = (stor_agg.loc[:, "bus"] + " " + 
                              stor_agg.loc[:, "carrier"] + " " +
                              stor_agg.loc[:, "p_nom_extendable"].astype(str) + 
                              " agg storage")
    
    # merge large individual storage units with aggregated small ones
    stor = pd.concat([stor.loc[(stor["p_nom"]+stor["p_nom_max"].fillna(0).astype(float))>=thres],stor_agg])
    
    # Check for duplicates in storage names
    stor_dupes = stor[stor['name'].duplicated(keep=False)]
    if not stor_dupes.empty:
        logger.warning("Duplicate storage names:")
        logger.warning(stor_dupes[['name', 'carrier', 'bus', 'p_nom', 'p_nom_extendable']])
        logger.warning(f"\nDuplicate names: {stor_dupes['name'].unique().tolist()}")
        
    #%% Add generic solar PV and wind based on resource potential
    
    if include_generic_vres:
        
        # setting up data structures and iterations
        vres = {"solar": ("Solar PV", path_solar_potential),
                "wind": ("Wind", path_wind_potential)}
        periods = pd.read_csv(path_data+"investment_periods.csv")["period"].tolist()
        periods_df = pd.DataFrame({"period": periods})
        potentials = {}
        pp_generic, gc = [], []

        regions = areas.set_index("id").to_crs(potential_area_crs)
        
        # load layers
        protected = (gpd.read_file(path_protected_areas)
                     if path_protected_areas is not None else None)
        if protected is None:
            logger.warning("VRE potentials: protected-area exclusions are disabled.")
        
        landcover = rasterio.open(path_landcover) 

        slope = (rasterio.open(path_slope)
                     if path_slope is not None else None)
        if slope is None:
            logger.warning("VRE potentials: slope is not considered.")
            
        # iterating through wind/solar estimating potentials based on land
        # availability, scaled to totals if triggered
        for carrier, (_, path_potential) in vres.items():
            
            settings = potential_grid_settings[carrier]
            excluder = ExclusionContainer(crs=regions.crs, res=potential_resolution)
            
            # setting included (invert=True) or excluded land cover types/areas 
            # based on config
            excluder.add_raster(landcover, codes=settings["land_cover_codes"], invert=True, nodata=0)
            if settings["settlement_buffer"] > 0:
                excluder.add_raster(landcover, codes=[50], buffer=settings["settlement_buffer"], nodata=0)
            if protected is not None and not protected.empty:
                excluder.add_geometry(protected)
            if settings["max_slope"] is not None:
                if slope is None:
                    raise ValueError(f"{carrier} max_slope requires a slope raster.")
                excluder.add_raster(slope, codes=partial(np.less, settings["max_slope"]), nodata=255)

            
            # calculate capacity potentials
            capacities = []
            for bus in regions.index:
                eligible, transform = shape_availability(regions.loc[[bus]].geometry, excluder)
                area_km2 = eligible.sum() * excluder.res**2 /1e6
                capacities.append(area_km2 * settings["capacity_per_sqkm"])
            potential = pd.DataFrame({"bus": regions.index, "potential_MW": capacities})
            raw_total = potential["potential_MW"].sum()
            logger.info(f"Unscaled technical potential for {carrier}: {raw_total:.2f} MW.")
            
            # scale to total if triggered
            # FIXME: should this take into account the licensed power plants,
            # i.e., scale to total - [capacity of licensed power plants]?
            if scale_potentials_to_total:
                target = potential_national_totals.get(carrier)
                if target is None:
                    logger.warning(f"No national {carrier} potential provided; retaining unscaled grid potential.")
                else:
                    if isinstance(target, str):
                        if not target.startswith("Laupane-") or not target.endswith("f"):
                            raise ValueError(f"Unknown {carrier} potential scenario: {target}; use 'Laupane-Xf'.")
                        level = target.removeprefix("Laupane-")
                        target = pd.read_csv(path_potential)[f"potential_{level}_MW"].sum(skipna=False)
                    if not np.isfinite(target) or target < 0:
                        raise ValueError(f"National {carrier} potential must be finite and nonnegative.")
                    if raw_total == 0 and target > 0:
                        raise ValueError(f"Cannot allocate {target} MW of {carrier}: no eligible area.")
                    factor = target / raw_total if raw_total > 0 else 0.0
                    potential["potential_MW"] *= factor
                    logger.info(f"Grid {carrier}: scaling factor {factor:.6g}, national total {target:.2f} MW.")
                    if factor > 1:
                        logger.warning(f"Scaled {carrier} potential exceeds the initial density-based estimate.")
            potentials[carrier] = potential

        # create generators and cumulative limits
        for carrier, (cost_key, _) in vres.items():
            potential = potentials[carrier]
            vres_pp = pps.loc[pps["carrier"] == carrier]
            existing = vres_pp.loc[~vres_pp["p_nom_extendable"]].groupby("bus")["p_nom"].sum()
            future = vres_pp.loc[vres_pp["p_nom_extendable"]].groupby("bus")["p_nom_max"].sum()
            potential["existing"] = potential["bus"].map(existing).fillna(0.0)
            potential["licensed"] = potential["existing"] + potential["bus"].map(future).fillna(0.0)
            
            # check for conflicts, i.e., licensed higher than calculated potential
            excess = potential["licensed"] > potential["potential_MW"]
            if excess.any():
                logger.warning(f"Licensed {carrier} capacity exceeds technical potential at buses: "
                                 + ", ".join(potential.loc[excess, "bus"].astype(str)))
            # Calculate relevant potentials, note the cumulative expansion limit
            # includes future licenses, but not existing assets.
            potential["pot_generic"] = (potential["potential_MW"] - potential["licensed"]).clip(lower=0)
            potential["pot_constraint"] = (potential["potential_MW"] - potential["existing"]).clip(lower=0)

            lifetime = tech_costs_per_kw.loc[cost_key, "Lifetime (years)"]
            capex_by_period = {
                period: annualize_cost(tech_costs_per_kw.loc[cost_key, str(period)],
                                       discount_rate, lifetime)
                for period in periods
            }
            vintage = potential.merge(periods_df, how="cross")
            # arrange generic power plant data
            pp_generic.append(pd.DataFrame({
                "name": (vintage["bus"] + " " + carrier + " potential "
                         + vintage["period"].astype(str)),
                "p_nom": 0.0, "build_year": vintage["period"],
                "bus": vintage["bus"], "carrier": carrier, "lifetime": lifetime,
                "capital_cost": vintage["period"].map(capex_by_period),
                "marginal_cost": tech_costs_per_kw.loc[cost_key, "Marginal Cost ($/MWh)"],
                "efficiency": tech_costs_per_kw.loc[cost_key, "Efficiency"],
                "p_nom_extendable": True, "p_nom_min": 0.0,
                "p_nom_max": vintage["pot_generic"],
            }))
            # arrange global constraint
            gc.append(pd.DataFrame({
                "name": potential["bus"] + " " + carrier + " potential limit",
                "type": "tech_capacity_expansion_limit", "carrier_attribute": carrier,
                "sense": "<=", "constant": potential["pot_constraint"], "bus": potential["bus"],
            }))
            logger.info(f"Added generic {carrier} potential: {potential['pot_generic'].sum():.2f} MW "
                        f"remaining, cumulative expansion limit {potential['pot_constraint'].sum():.2f} MW.")

        pps = pd.concat([pps, *pp_generic], ignore_index=True)
        pd.concat(gc, ignore_index=True).to_csv(path_data+"global_constraints.csv",
                                                       index=False)
    
    
    #%% Save input data
    pps.to_csv(path_data+"generators.csv",index=False)
    stor.to_csv(path_data+"storage_units.csv",index=False)
    
