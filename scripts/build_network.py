"""
Script building the transmission network and areas

Copyright (C) 2026 PyPSA-Nepal contributors, licensed under a MIT license
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from shapely import MultiPoint, voronoi_polygons

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
    
    # %% Levers and paths
    
# FIXME: load settings from a config file.
root = Path(__file__).resolve().parents[1]
path_adm_shapefile = root / "data/raw_data/boundaries/geoBoundaries-NPL-ADM0-all/geoBoundaries-NPL-ADM0.shp"
path_buses_shapefile = root / "data/raw_data/grid/pre_built/network_buses.shp"
path_lines_shapefile = root / "data/raw_data/grid/pre_built/network_lines.shp"


# FIXME: update these placeholder data 
line_types = {
    132: "243-AL1/39-ST1A 110.0",
    220: "Al/St 240/40 2-bundle 220.0",
}
network_voltage = 220.0  # kV: common voltage of the simplified AC network
line_type = "Al/St 240/40 2-bundle 220.0"
line_cost = 100  # annual currency/(MW*m), interpreted per MVA
line_life = 40  # years
line_extendable = True 
line_s_max_pu = 1.0  



path_output = root / "data/network"
path_processed = root / "data/processed_data"
# FIXME: to be used in future, currently unused
path_prov_shapefile = root / "data/raw_data/boundaries/geoBoundaries-NPL-ADM1-all/geoBoundaries-NPL-ADM1_simplified.shp"


# "full", "clustering" (k-means), or "custom" (provided region polygons).
processing = "full"
    
    #%% Worker functions

def _validate_geometries(frame, label, allowed_types):
    """Check spatial inputs; attribute validation belongs to their preparation."""
    if frame.empty or frame.crs is None:
        raise ValueError(f"{label} must be nonempty and have a CRS.")
    if not frame.geometry.geom_type.isin(allowed_types).all():
        raise ValueError(f"Unexpected or missing geometry types in {label}.")
    if frame.geometry.is_empty.any() or not frame.geometry.is_valid.all():
        raise ValueError(f"Empty or invalid geometries in {label}.")


def build_voronoi_regions(buses, boundary):
    """Create territories for buses already validated by prepare_buses."""
    _validate_geometries(boundary, "boundary", ["Polygon", "MultiPolygon"])
    # Distances are calculated in metres, not longitude/latitude degrees
    metric_crs = boundary.estimate_utm_crs()
    boundary_shape = boundary.to_crs(metric_crs).geometry.iloc[0]
    points = buses.to_crs(metric_crs)
    
    # check for duplicated buses and outside buses.
    if points.geometry.duplicated().any():
        raise ValueError("Bus coordinates must be unique to create Voronoi areas.")
    outside = ~points.geometry.covered_by(boundary_shape)
    if outside.any():
        raise ValueError(
            "Buses outside the provided boundary need review: "
            + ", ".join(points.loc[outside, "id"])
        )

    if len(points) == 1:
        cells = [boundary_shape]
    else:
        cells = list(voronoi_polygons(
            MultiPoint(points.geometry.tolist()),
            extend_to=boundary_shape.envelope,
        ).geoms)

    # match cells spatially
    regions = gpd.GeoDataFrame(geometry=cells, crs=metric_crs)
    regions = regions.sjoin(points[["id", "geometry"]], predicate="covers")
    if len(regions) != len(points) or regions["id"].nunique() != len(points):
        raise ValueError("Could not assign exactly one Voronoi area to every bus.")
        
    regions["geometry"] = regions.geometry.intersection(boundary_shape)
    regions = regions[["id", "geometry"]].sort_values("id").reset_index(drop=True)

    # perform checks on Voronoi areas
    if regions.geometry.is_empty.any() or not regions.geometry.is_valid.all():
        raise ValueError("Voronoi territories contain empty or invalid geometries.")
    covered = regions.geometry.union_all()
    tolerance = boundary_shape.area * 1e-8
    if boundary_shape.symmetric_difference(covered).area > tolerance:
        raise ValueError("Voronoi territories do not cover Nepal's territory.")
    if regions.geometry.area.sum() - covered.area > tolerance:
        raise ValueError("Voronoi territories overlap.")
    return regions.to_crs("EPSG:4326")


def prepare_network(
    buses, lines, line_types, network_voltage, line_type, line_cost,
    line_life, line_extendable, line_s_max_pu,
):
    """Prepare the simplified AC bus and line tables."""
    bus_table = prepare_buses(buses, network_voltage)
    line_table = prepare_lines(
        lines, bus_table, line_types=line_types, line_type=line_type,
        line_cost=line_cost, line_life=line_life,
        line_extendable=line_extendable, line_s_max_pu=line_s_max_pu,
    )

    return bus_table, line_table


def prepare_buses(buses, network_voltage):
    """Prepare buses table"""
    _validate_geometries(buses, "buses", ["Point"])

    if not np.isfinite(network_voltage) or network_voltage <= 0:
        raise ValueError("Network voltage must be finite and positive.")
        
    points = buses.to_crs("EPSG:4326").set_index("id")
    bus_table = pd.DataFrame({
        "x": points.geometry.x, "y": points.geometry.y, "carrier": "AC",
        "v_nom": network_voltage,
    })
    bus_table.index.name = "name"

    return bus_table


def prepare_lines(
    lines, bus_table, line_types, line_type, line_cost,
    line_life, line_extendable, line_s_max_pu,
):
    """Prepare lines table."""
    _validate_geometries(lines, "lines", ["LineString", "MultiLineString"])
    required = ["name", "bus0", "bus1", "volt_kv", "len_km", "circuits"]
    routes = lines[required].copy()

    for column in ["volt_kv", "len_km", "circuits"]:
        routes[column] = pd.to_numeric(routes[column], errors="raise")
        
    settings = np.array([line_cost, line_life, line_s_max_pu], dtype=float)
    if (
        not np.isfinite(settings).all()
        or line_cost < 0 or line_life <= 0
        or not 0 < line_s_max_pu <= 1
    ):
        raise ValueError(
            "Line settings must be finite: lifetime positive, cost "
            "non-negative, and s_max_pu in (0, 1]."
        )
        
    if not isinstance(line_extendable, (bool, np.bool_)):
        raise ValueError("line_extendable must be a boolean.")

    # process line types
    source_types = routes["volt_kv"].map(line_types)
    if source_types.isna().any():
        voltages = sorted(routes.loc[source_types.isna(), "volt_kv"].unique())
        raise ValueError(f"No source line type configured for voltages: {voltages}")
    n = pypsa.Network()
    types_used = pd.Index([*source_types.unique(), line_type]).unique()
    unknown = types_used.difference(n.line_types.index)
    if not unknown.empty:
        raise ValueError(f"Unknown line types: {unknown.tolist()}")
        
    # calculate transmission capacity
    source_current = source_types.map(n.line_types["i_nom"])
    s_nom = np.sqrt(3) * routes["volt_kv"] * source_current * routes["circuits"]
    
    # arrange line data
    # fractional num_parallel describes equivalent circuits at the common voltage
    common_current = n.line_types.at[line_type, "i_nom"]
    routes["num_parallel"] = s_nom / (np.sqrt(3) * network_voltage * common_current)
    routes["s_nom"] = s_nom
    routes["s_nom_min"] = s_nom
    routes["s_nom_extendable"] = line_extendable
    routes["s_max_pu"] = line_s_max_pu
    routes["type"] = line_type
    routes["carrier"] = "AC"
    # Annual cost per MVA, using the unity-power-factor convention for MW costs.
    routes["capital_cost"] = line_cost * routes["len_km"] * 1000
    routes["lifetime"] = line_life
    routes["source_type"] = source_types
    line_table = routes.rename(columns={
        "len_km": "length", "volt_kv": "source_v_nom", "circuits": "source_circuits",
    }).set_index("name")
    line_table = line_table[[
        "bus0", "bus1", "carrier", "type", "length", "num_parallel",
        "s_nom", "s_nom_min", "s_nom_extendable", "s_max_pu", "capital_cost",
        "lifetime", "source_v_nom", "source_type", "source_circuits",
    ]]

    return line_table


if __name__ == "__main__":

    # %% Load the geofiles
    
    # load shapefiles
    buses = gpd.read_file(path_buses_shapefile)
    lines = gpd.read_file(path_lines_shapefile)
    boundary = gpd.read_file(path_adm_shapefile)
    
    # %% Prepare network tables before constructing areas
    bus_table, line_table = prepare_network(
        buses, lines, line_types=line_types, network_voltage=network_voltage,
        line_type=line_type, line_cost=line_cost, line_life=line_life,
        line_extendable=line_extendable, line_s_max_pu=line_s_max_pu,
    )

    # %% Process buses and their areas
    if processing == "full":
        regions = build_voronoi_regions(buses, boundary)
    elif processing == "clustering":
        # TODO: cluster buses with k-means, then create Voronoi territories.
        raise NotImplementedError("K-means clustering is not implemented yet.")
    elif processing == "custom":
        
        # if buses == "provinces":
        #     areas = gpd.read_file("../data/raw_data/boundaries/geoBoundaries-NPL-ADM1-all/geoBoundaries-NPL-ADM1_simplified.shp")
        #     areas.loc[:,"shapeName"] = areas.loc[:,"shapeName"].replace({"Province 1":"Koshi",
        #                                                                  "Province 2":"Madhesh",
        #                                                                  "Sudurpaschim":"Sudurpashchim"})
        # elif buses == "districts":
        #     areas = gpd.read_file("../data/raw_data/boundaries/geoBoundaries-NPL-ADM2-all/geoBoundaries-NPL-ADM2_simplified.shp")
        # else:
        #     raise ValueError(f"'{buses}' not defined as value for resolution.")
        
        
        # TODO: aggregate buses to supplied regions; add/connect missing buses.
        raise NotImplementedError("Custom region clustering is not implemented yet.")
    else:
        raise ValueError(f"Unknown processing option: {processing}")

    # %% Save model files

    output = Path(path_output)
    output.mkdir(parents=True, exist_ok=True)
    proc_path = Path(path_processed)
    regions.to_file(proc_path / "bus_regions.shp", index=False, encoding="utf-8")
    bus_table.to_csv(output / "buses.csv")
    line_table.to_csv(output / "lines.csv")
    
    logger.info("Saved %d buses, %d lines and %d territories to %s",
                len(bus_table), len(line_table), len(regions), output)

